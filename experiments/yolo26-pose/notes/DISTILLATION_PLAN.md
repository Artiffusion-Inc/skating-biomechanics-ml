# Knowledge Distillation Plan: MogaNet-B → YOLO26n

**Date:** 2026-04-16
**Goal:** Train YOLO26n with MogaNet-B quality (AP~0.96) at YOLO speed (588 FPS)

---

## Strategy

### Two-Stage Pipeline

```
Stage 1: Pseudo-labeling (Offline)
MogaNet-B → SkatingVerse (28K videos) → pseudo_labels/
          ├─ COCO JSON format
          ├─ Confidence scores
          └─ Filter: conf > 0.5

Stage 2: Distillation Training
Pseudo-labels + AthletePose3D → YOLO26n train
                           ├─ Knowledge distillation loss
                           ├─ Validation every epoch
                           └─ Compare: student vs teacher
```

---

## Stage 1: Pseudo-labeling SkatingVerse

### Input
- **Source:** SkatingVerse dataset (28K videos, 28 classes)
- **Location:** `/root/data/datasets/skatingverse/`
- **Teacher:** MogaNet-B (AP=0.962)

### Output
- **Format:** COCO JSON (compatible with YOLO)
- **Location:** `/root/data/datasets/skatingverse_pseudo/`
- **Structure:**
  ```
  pseudo_labels/
  ├── annotations/
  │   ├── train.json          # Pseudo-labels for training
  │   └── val.json            # Holdout (20% for validation)
  └── images/
      ├── train/              # Symlinks to original frames
      └── val/
  ```

### Filtering
- **Confidence threshold:** 0.5 (tune based on distribution)
- **Keypoint quality:** Remove predictions with max(heatmap) < 0.3
- **Person detection:** Use MogaNet's confidence score

### Processing
```bash
# Extract frames from videos (10 fps)
python scripts/extract_frames_skatingverse.py

# Pseudo-label with MogaNet-B
python scripts/pseudo_label_skatingverse.py \
    --model-path /root/data/models/athletepose3d/moganet_b_ap2d_384x288.pth \
    --input /root/data/datasets/skatingverse/frames/ \
    --output /root/data/datasets/skatingverse_pseudo/ \
    --conf-thresh 0.5 \
    --workers 8
```

---

## Stage 2: Distillation Training

### Knowledge Distillation Loss

```python
# Standard YOLO loss
loss_detection = yolo_loss(pred, gt_labels)

# Distillation loss (soft targets from teacher)
loss_distill = KL_divergence(
    student_logits,
    teacher_logits  # From MogaNet-B heatmaps
)

# Combined loss
loss = loss_detection + λ * loss_distill
```

**Hyperparameters:**
- `λ` (distillation weight): 0.1 → 0.5 (tune)
- Temperature for softmax: 1.0 (default)
- Optimizer: AdamW (lr=0.001, cosine decay)

### Training Config

```yaml
# YOLO26n distillation config
model: yolo26n-pose.pt
data: skatingverse_pseudo.yaml

# Training
epochs: 100
batch: 64
device: [0,1]  # 2x GPU

# Distillation
teacher: moganet_b_ap2d_384x288.pth
distill_weight: 0.3  # λ
temperature: 1.0

# Validation (CRITICAL!)
val: true
val_period: 1
save_period: 5
patience: 15

# Augmentation
mosaic: 1.0
mixup: 0.15
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
```

---

## Validation Strategy

### Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| **mAP50-95** | Overall accuracy | > 0.85 |
| **mAP50** | Loose threshold | > 0.95 |
| **AP (per keypoint)** | Per-joint accuracy | > 0.80 (worst kp) |
| **AR** | Recall | > 0.90 |
| **Speed (FPS)** | Inference speed | > 500 FPS |
| **Distill Loss** | KL divergence | ↓ Decreasing |

### Validation Sets

1. **AthletePose3D val** (100 images)
   - Ground truth available
   - Compare student vs teacher directly
   - **Target:** Close gap to < 0.05 AP difference

2. **SkatingVerse pseudo val** (20% holdout)
   - No ground truth
   - Monitor student vs teacher consistency
   - **Metric:** Teacher-student agreement

3. **COCO val** (optional)
   - Generalization check
   - Prevent overfitting to skating domain

---

## Expected Results

### Baseline (Before Distillation)

| Model | AP (AthletePose3D) | FPS (4090) |
|-------|-------------------|------------|
| MogaNet-B (teacher) | **0.962** | 15.9 |
| YOLO26n pretrained | 0.712 | 588 |
| **Gap** | **-0.250** | - |

### Target (After Distillation)

| Model | AP (AthletePose3D) | FPS (4090) |
|-------|-------------------|------------|
| MogaNet-B (teacher) | 0.962 | 15.9 |
| **YOLO26n distill** | **> 0.85** | **588** |
| **Gap** | **< 0.12** | - |

**Success Criteria:**
- ✅ YOLO26n AP > 0.85 on AthletePose3D val
- ✅ Speed remains > 500 FPS
- ✅ Gap to teacher reduced by 50%+

---

## Implementation Steps

### Phase 1: Pseudo-labeling (1-2 days)
1. ✅ Extract frames from SkatingVerse videos
2. ✅ Run MogaNet-B inference on all frames
3. ✅ Filter by confidence threshold
4. ✅ Export to COCO JSON format
5. ✅ Split train/val (80/20)

### Phase 2: Distillation Training (3-5 days)
1. ✅ Implement distillation loss in YOLO
2. ✅ Create training config
3. ✅ Train with validation every epoch
4. ✅ Monitor metrics: student vs teacher
5. ✅ Early stopping on val AP

### Phase 3: Evaluation (1 day)
1. ✅ Final evaluation on AthletePose3D test
2. ✅ Speed benchmark (FPS)
3. ✅ Compare with all baselines
4. ✅ Ablation study (λ values)

---

## Files to Create

1. **scripts/extract_frames_skatingverse.py** — Frame extraction from videos
2. **scripts/pseudo_label_skatingverse.py** — MogaNet-B inference pipeline
3. **experiments/exp_distill_moganet_to_yolo26n.py** — Distillation training
4. **ml/src/distillation/yolo_distill.py** — YOLO distillation wrapper
5. **experiments/yolo26-pose/configs/distill_moganet.yaml** — Training config

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pseudo-labels noisy | Student learns bad patterns | Filter by conf > 0.7, use hard negatives |
| Overfitting to SkatingVerse | Poor generalization | Mix with COCO (15%), augment |
| Teacher-student gap too large | Distillation ineffective | Use intermediate fine-tuned model as bridge |
| Training instability | Loss diverges | Warmup, gradient clipping, lower λ |

---

## Timeline Estimate

- **Pseudo-labeling:** 1-2 days (28K videos → ~300K frames at 10 fps)
- **Training setup:** 0.5 day (config, loss implementation)
- **Training:** 2-3 days (100 epochs on 2x RTX 4090)
- **Evaluation:** 0.5 day

**Total:** ~4-6 days

---

## References

- Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
- YOLOv8-pose distillation guide
- MMPose knowledge distillation implementation

---

## Next Steps

1. ✅ Check SkatingVerse dataset structure
2. ✅ Create pseudo-labeling script
3. ✅ Start frame extraction (background job)
4. ✅ Implement distillation loss

**Status:** 📝 PLANNING
**Last updated:** 2026-04-16
