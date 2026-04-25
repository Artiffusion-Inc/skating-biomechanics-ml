# KD Architecture Research — Agent Synthesis Report
**Date:** 2026-04-24
**Status:** ACTION REQUIRED — v34 must be killed and restarted

## Executive Summary

Current KD pipeline is fundamentally broken. Three parallel research agents identified critical issues:
1. **Dead gradient path** — KD loss only trains sigma head, zero gradient to keypoint predictions
2. **Architecture mismatch** — YOLO26-Pose uses coordinate regression + RLE, not heatmaps
3. **Suboptimal loss** — KL divergence on heatmaps is theoretically mismatched

**Gemini curator verdict: Kill v34, restart with Hybrid RLE-Coordinate KD + Feature Distillation.**

---

## Agent 1: SimCC vs Heatmap Mismatch — CRITICAL

### Findings
- **DWPose distills 1D SimCC classification vectors** (KL on per-axis probability distributions)
- **YOLO26-Pose outputs direct coordinate regression + RLE** (RealNVP normalizing flow)
- Our "student heatmap" is built from **GT keypoints + student sigma** — not student predictions
- KD loss provides gradient signal **ONLY to sigma head**, not to keypoint coordinates

### Three Independent Mismatches
| Mismatch | Severity | Detail |
|----------|----------|--------|
| Teacher heatmap (72x96) vs DWPose SimCC (1280-length 1D) | HIGH | Different information density |
| Student cannot produce heatmaps | CRITICAL | Pose26 head outputs coordinates + RLE |
| Student "heatmap" built from GT | CRITICAL | Zero gradient to actual predictions |

### Code Reference (distill_trainer.py:691-699)
```python
kpts_xy = primary_kpts[..., :2]        # GT keypoints, NOT student predictions!
sigma_mean = sigma_reshaped.mean(dim=1)  # student sigma
student_hm = keypoints_to_heatmap(kpts_xy.detach(), sigma_mean, ...)
```

### Options
- **A: Coordinate-level KD** (MSE on coordinates) — simplest, no arch changes
- **B: RLE-based KD** — leverage YOLO26 native RealNVP flow
- **C: Add SimCC head** — closest to DWPose, but adds inference overhead

---

## Agent 2: Backbone Unfreezing — APPROVED

### Findings
- Ultralytics does **NOT** support progressive unfreezing natively
- Custom `build_optimizer` + `on_train_epoch_start` callback required
- YOLO26s-Pose: 363 layers, 11.87M params, backbone (layers 0-10) = 63-67%
- DWPose uses 0.1x LR for backbone during KD
- Memory: unfreeze all + AMP = ~14GB on RTX 5090 (fits in 32GB)

### Recommended Schedule
```
Epochs 1-10:    Head only (freeze=10), lr=2e-3
Epochs 11-210:  Full unfreeze, backbone_lr=2e-4 (0.1x), head_lr=2e-3
                + 1-2 epoch LR warm-up for backbone at unfreeze
```

### Key Insight
Unfreezing backbone **re-enables feature distillation** (currently disabled due to frozen backbone). Feature KD + Logit KD is strictly better than logit-only.

---

## Agent 3: KL vs MSE + Temperature — MSE WINS

### Findings
- Temperature (Hinton 2015) designed for unnormalized classification logits, NOT heatmaps
- Heatmaps are already soft distributions — temperature provides no benefit
- KL on heatmaps requires softmax on already-normalized values — destroys absolute magnitude
- **MSE is the standard** for heatmap distillation (DistilPose CVPR 2023, HRNet, SimpleBaseline)
- IJCAI 2021 (arXiv:2105.08919): MSE outperforms KL in KD

### Recommendation
Replace KL with MSE on raw heatmaps. Beta=0.1. No temperature.
**BUT:** This becomes secondary — student has no heatmap head, focus on coordinate MSE.

---

## Gemini Curator Verdict

### Strategy: Hybrid RLE-Coordinate KD + Feature Distillation

```
L_KD = alpha * MSE(mu_student, mu_teacher) + (1-alpha) * KL(P_student, P_teacher)
```

| Component | Decision |
|-----------|----------|
| KD Strategy | Hybrid RLE-Coordinate: MSE on (x,y) + KL on RealNVP parameters |
| Loss | Replace heatmap KD with Feature Distillation (P3/P4/P5 feature maps) |
| Scheduler | 2 phases with smooth unfreezing (1-2 epoch LR warm-up for backbone) |
| Architecture | NO SimCC head — keep RLE for real-time inference speed |

### Key Points
- Feature Distillation on P3/P4/P5 feature maps (MSE between teacher/student) gives more benefit than heatmap matching
- Do NOT add SimCC head — inference speed is priority for real-time on-ice analysis
- Kill v34 immediately — current iteration only trains sigma, wasting GPU cycles

### Open Question
- Is alpha=0.5 sufficient for coordinate loss to overcome distribution distillation noise in early epochs?

---

## Next Steps
1. Kill v34 on Vast.ai
2. Offline: extract teacher coordinates from LMDB → HDF5 (argmax + confidence)
3. Verify teacher features HDF5 spatial alignment with YOLO26s (stride check)
4. Rewrite distill_trainer.py with Hybrid RLE-Coordinate KD + Feature Distillation
5. Launch v35

---

## V35 Architecture (Confirmed by Gemini)

### KD Strategy: Hybrid Coordinate + Feature
```
L_total = L_pose_standard + coord_alpha * L_coord + feat_alpha * L_feat
L_coord = w_conf * MSE(student_kpts, teacher_kpts)  # confidence-weighted
L_feat  = MSE(L2_norm(adapter(student_feat)), L2_norm(teacher_feat))  # P3/P4/P5
```

### Backbone: Progressive Unfreeze
- Epochs 1-10: Head only (freeze=10), lr=2e-3
- Epochs 11-12: Linear warm-up backbone 0→2e-4
- Epochs 13-210: Full train, backbone_lr=2e-4 (0.1x), head_lr=2e-3

### Feature Distillation
- Layers: 4/6/8 (MogaNet-B) → P3/P4/P5 (YOLO26s)
- Adapter: trainable 1x1 conv, identity-like init (nn.init.eye_)
- L2 normalization before MSE
- feat_alpha=0.01 (initial)

### Coordinate Distillation
- Offline: extract teacher coords from LMDB heatmaps (argmax + max confidence)
- Storage: HDF5 for fast random access
- coord_alpha=0.5
- No temperature (coordinate MSE)

### Other
- COCO 10%: retained for catastrophic forgetting
- Model: yolo26s-pose.yaml from scratch (pretrained COCO backbone)
- 270K images, 210 epochs, batch=128, imgsz=384
- Adapters in optimizer from epoch 1 (converge before backbone unfreeze at 11)

### Gemini Notes
- Monitor adapter convergence first 10 epochs
- If coord/feat loss scales differ significantly, adjust alpha
- Feature KD should be ~20-30% of total loss, not dominant
- Identity init: ensure correct handling when C_in != C_out

---

## Independent Verifier Findings (4 CRITICAL, 4 HIGH)

### CRITICAL Issues

**C1. Student predictions not accessible** — PoseLoss26 decodes kpts inside loss() and doesn't return them. Need hook or duplicate decode logic to get student (x,y) for coordinate KD.

**C2. No sub-pixel teacher precision** — Hard argmax on 72x96 heatmap = integer pixels. Student regression = sub-pixel. Fix: use soft argmax (spatial softmax) for teacher coordinate extraction.

**C3. Coordinate space mismatch** — MogaNet-B top-down (crop space 288x384) vs YOLO26-Pose bottom-up (full image 384x384). Teacher coords crop-relative, student full-image. **Fix: Inverse Affine Transform** — convert teacher coords to global image space using GT bbox (cx, cy, w, h).

**C4. Multi-person assignment** — Teacher processes one person per crop, student predicts multiple persons per image. No reliable 1-to-1 mapping. Fix: Object-Centric Alignment via IoU matching.

### HIGH Issues

**H1. Spatial resolution mismatch** — Teacher input 384x288 (portrait), student 384x384 (square). Feature maps have different aspect ratios.

**H2. WRONG layer indices** — All 3 layers (4,6,8) are in Stage 2 of MogaNet-B (160ch, 1/8 res). No multi-scale diversity. **Fix: Use layers 3 (64ch, 1/4), 9 (160ch, 1/8), 31 (320ch, 1/16)** for proper multi-scale coverage.

**H3. Identity init impossible** — nn.init.eye_ fails when C_in != C_out. Fix: truncated SVD init or Xavier uniform with small gain.

**H4. Float16 teacher features** — L2 norm on float16 amplifies quantization noise. Fix: store in float32 or bfloat16.

### MEDIUM Issues
- M1: Optimizer rebuild loses AdamW momentum states at epoch 11
- M2: coord_alpha=0.5 likely too high — start with 0.01-0.05
- M3: L2 norm before MSE loses scale information
- M4: HDF5 index in JSON sidecar, not in HDF5 attrs (loader reads wrong location)
- M5: Dual storage backends (LMDB + HDF5) — consolidate to HDF5
- M6: Feature adapters not in optimizer param groups — register as model submodules

---

## Gemini Verdict on Verifier Findings

### Coordinate KD IS possible between top-down and bottom-up, but needs:

1. **Coordinate Transformation Module** — Inverse Affine Transform (crop→global) using GT bbox params saved during training
2. **Object-Centric Alignment** — IoU matching between teacher crop and student detections
3. **Confidence Thresholding** — Only distill where teacher is confident
4. **Correct layer mapping** — Stage 2 Student → Stage 1 Teacher (spatial), Stage 4 Student → Stage 3 Teacher (semantic)

### Revised Architecture Requirements
- Save crop bbox params (cx, cy, w, h) for each teacher instance during data prep
- Inverse affine transform: `x_global = M_inv * x_crop` before coordinate KD
- Feature KD: regenerate teacher features at correct layers (3, 9, 31)
- Adapter init: truncated SVD (not identity) when C_in != C_out
- coord_alpha: start at 0.01-0.05 (NOT 0.5)
- Store teacher data in float32
