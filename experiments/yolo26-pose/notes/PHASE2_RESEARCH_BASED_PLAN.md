# Phase 2: Research-Based Plan (Overfitting Prevention)

**Date:** 2026-04-15
**Sources:** AAAI 2025, WACV 2025, Ultralytics docs, GeeksforGeeks

---

## ⚠️ CRITICAL: READ BEFORE PROCEEDING

**This plan is ONLY valid AFTER Phase 1 confirms a clear winner.**

**See @ITERATIVE_RESEARCH_STRATEGY.md for complete decision tree.**

### Phase 1 → Phase 2 Transition Criteria (ALL must be TRUE):

- [✅/❌] Clear winner from HP search (mAP > baseline + 5%)
- [✅/❌] Statistically significant (p < 0.05, not noise)
- [✅/❌] Reproducible on 2 different subsets
- [✅/❌] No severe overfitting (train-val gap < 0.1)
- [✅/❌] Stable convergence (< 5% variance across last 10 epochs)

### If ANY criteria is FALSE:

**❌ DO NOT PROCEED TO PHASE 2**

Instead:
1. See @ITERATIVE_RESEARCH_STRATEGY.md
2. Start new research cycle (Cycle 2, 3, or 4)
3. Repeat until clear winner emerges
4. Only THEN return to this plan

### Current Status:

**Phase 1 Progress:** See @HP_SEARCH.md
**Decision:** [PENDING - waiting for HP search validation results]

---

## Linked notes:
- Iterative strategy → @ITERATIVE_RESEARCH_STRATEGY.md
- HP search progress → @HP_SEARCH.md
- Research comparison → @COMPREHENSIVE_RESEARCH_COMPARISON.md
- Post-mortem → @POST-MORTEM.md

---

## Critical Research Findings

### 1. Pseudo-Labeling Confidence Threshold

**Finding:** 0.95 (not 0.8!)

**Sources:**
- GeeksforGeeks: "probability > 0.95" for preventing noisy labels
- AAAI 2025: "Revisiting Pseudo-Labeling" — adaptive thresholds outperform fixed
- WACV 2025: "When Confidence Fails" — threshold methods suffer from overconfidence

**Implication:** Use 0.95 confidence, NOT 0.8

### 2. Confirmation Bias Problem

**Finding:** Models reinforce their own mistakes (Arazo et al. 2020)

**Sources:**
- Multiple SSL papers cite this critical issue
- Adaptive thresholds help but don't eliminate the problem

**Implication:** Conservative weight for pseudo-labels (0.15-0.20, not 0.3)

### 3. Freeze Layers Strategy

**Finding:** freeze=10 optimal for domain-specific fine-tuning

**Sources:**
- Ultralytics docs: freeze first 10 layers
- GitHub issues: freeze=10 prevents catastrophic forgetting

**Implication:** Use freeze=10 for teacher training

### 4. Early Stopping

**Finding:** patience=5 prevents overfitting

**Sources:**
- Ultralytics docs: "patience=5 means training will stop if no improvement"
- Standard practice for YOLO fine-tuning

**Implication:** Enable early stopping with patience=5

---

## Updated Phase 2 Plan

### Phase 2A: Teacher Training (3-4 hours)

```yaml
# teacher_config.yaml
model: yolo26s-pose.pt
data: ap3d_coco_mix.yaml  # 226K AP3D + 40K COCO (15%)
imgsz: 1280
epochs: 100
device: [0,1]

# Overfitting prevention
freeze: 10  # ✅ Research-backed
patience: 5  # ✅ Early stopping
mosaic: 0.0  # ✅ Confirmed from HP search
mixup: 0.05  # ✅ Regularization
fliplr: 0.5  # ✅ Symmetric task

# Validation
val: True  # S2 held-out subject
save_period: 5
```

**Success criteria:**
- S2 val mAP within 10% of S1 train mAP
- No overfitting (train/val gap < 0.3)

### Phase 2B: Pseudo-Labeling (24-48 hours)

```bash
python ml/scripts/pseudo_label_skatingverse.py \
    --model /root/yolo_runs/teacher/weights/best.pt \
    --input /root/data/datasets/skatingverse/ \
    --output /root/data/datasets/skatingverse_pseudo/ \
    --confidence 0.95 \
    --frames-per-video 300 \
    --skip-frames 8 \
    --min-keypoints 7
```

**Critical changes from research:**
- ✅ **confidence: 0.95** (was 0.8) — **CRITICAL**
- ✅ **min-keypoints: 7** (stricter filtering)
- ✅ **Adaptive threshold** (optional): use model's baseline uncertainty

**Expected output:** ~200K-500K pseudo-labels (lower due to 0.95 threshold)

### Phase 2C: Student Training (12-18 hours)

```yaml
# student_config.yaml
model: yolo26m-pose.pt
data: balanced_mix.yaml
imgsz: 1280
epochs: 100
device: [0,1]
freeze: 10  # ✅ Research-backed
patience: 5  # ✅ Early stopping

# Data mix: CONSERVATIVE
train:
  - path: ap3d_s1_gt
    weight: 1.0  # GT data, full weight
  - path: coco_train2017_gt
    weight: 0.5  # Generic pose
  - path: skatingverse_pseudo
    weight: 0.15  # ✅ Conservative (was 0.3)
    confidence: 0.95  # ✅ Research-backed
    min_kpts: 7  # ✅ Stricter filtering
```

**New balance:**
- GT (AP3D + COCO): **70%** influence
- Pseudo: **30%** influence (was 75%)

---

## Validation Strategy

### During Training

```python
# After each epoch
assert val_map_s2 > train_map_s1 * 0.9  # <10% drop on held-out
assert abs(train_loss - val_loss) < 0.3  # No overfitting
```

### Early Stopping Triggers

- S2 val mAP no improvement for 5 epochs → STOP
- Train/val loss gap > 0.3 → OVERFIT, stop
- Val loss increasing for 3 epochs → OVERFIT, stop

### Post-Training Validation

```python
# Teacher validation
assert teacher_map_s2 > teacher_map_s1 * 0.9  # <10% drop
assert teacher_map_coco > baseline_map_coco * 0.95  # Minimal forgetting

# Student validation
assert student_map_coco >= teacher_map_coco  # Not worse on generic
assert student_map_s2 >= teacher_map_s2 * 0.95  # <5% drop on domain
```

---

## Fallback Strategy

If overfitting detected:

1. **Reduce freeze**: freeze=15 (more constrained)
2. **Reduce epochs**: 50 instead of 100
3. **Increase mixup**: 0.1 instead of 0.05
4. **Reduce pseudo weight**: 0.1 instead of 0.15
5. **Increase confidence**: 0.97 instead of 0.95

---

## Cost

**Vast.ai:** ~$0.70/hr
- HP search (3h): ~$2.10 ✅ COMPLETE
- Phase 2A - Teacher (3h): ~$2.10
- Phase 2B - Pseudo-labels (36h): ~$25.20
- Phase 2C - Student (15h): ~$10.50
- **Total:** ~$39.90

**Cost optimization:**
- Stop instance after each phase
- Use spot instances (cheaper, risk of preemption)
- Pseudo-labeling on CPU (cheaper)

---

## References

- GeeksforGeeks: "Pseudo Labelling | Semi-Supervised learning"
- AAAI 2025: "Revisiting Pseudo-Labeling for Semi-Supervised Learning"
- WACV 2025: "When Confidence Fails: Revisiting Pseudo-Label Selection"
- Ultralytics docs: "Model Training Tips", "Transfer Learning"
- Arazo et al. 2020: "Confirmation bias in pseudo-labeling"

---

## Next Steps

1. ✅ Complete HP search (in progress)
2. [ ] Select top 2 configs
3. [ ] Prepare Phase 2A configs with research-backed settings
4. [ ] Run teacher training with freeze=10, patience=5
5. [ ] Validate teacher on S2 (check for overfitting)
6. [ ] Run pseudo-labeling with confidence=0.95
7. [ ] Train student with conservative pseudo weight (0.15)
8. [ ] Final validation on COCO + S2
