# Pose Paradigm Research: SimCC vs RLE & Beyond (2026-04-25)

## Context

v36b training running: YOLO26s-pose (RLE head) fine-tuned on MogaNet-B pseudo-labels (291K skating images). Epoch 8/200, mAP50(P)=0.703. This document captures the strategic analysis from 3-agent deep research (Opus-level, 20+ paradigm variants analyzed).

**Full research report:** `data/specs/2026-04-25-pose-paradigms-research.md`

---

## The Three Paradigms We Know

| Paradigm | How it works | Best model | COCO AP | Sub-pixel | Overhead |
|----------|-------------|-----------|---------|-----------|----------|
| **Heatmap** | Predict 2D Gaussian per kp, argmax decode | MogaNet-B, ViTPose | 77.3 (MogaNet) | No (pixel-level) | Heavy (H×W per kp) |
| **SimCC** | Classify x,y coords into bins, expected value decode | RTMPose-m | 75.8 | Yes (free) | Light (~0.5M MLP) |
| **RLE** | Direct regression + RealNVP flow for uncertainty | YOLO26s-pose | 63.0 | Yes (continuous) | Minimal (~0.1M) |

---

## RLE is Structurally Limited

The 8.5pp gap between RLE and SimCC is **structural**, not training-related. Evidence:

| Model size | RLE (YOLO26) | SimCC (RTMPose) | Gap |
|-----------|-------------|-----------------|-----|
| Small | 63.0 (s, 11.9M) | 71.5 (s, ~15M) | **8.5 pp** |
| Medium | 67.2 (m, 21.5M) | 75.8 (m, ~34M) | **8.6 pp** |

**Why:** RLE models *error distribution* (offsets from detection anchors). SimCC models *spatial classification* (cross-entropy over bins — richer gradient signal). RLE couples keypoint accuracy to detection quality and box scale.

**RLE is NOT bad** — it solves a different problem (speed-first mobile deployment). Just not optimal for our accuracy-first desktop pipeline.

---

## Paradigms Beyond the Big Three

### Free Improvements (no retraining)

| Technique | What | Gain | Effort |
|-----------|------|------|--------|
| **Dark Pose** (ICCV 2019) | Taylor expansion offset on heatmap argmax | +1.0-1.7 AP | Post-processing only |
| **UDP** (CVPR 2020) | Unbiased coordinate transform between image/crop/heatmap space | +1.4-1.7 AP | Change transform matrices |
| **Dark + UDP combined** | Both stack | +2-3 AP total | Both above |

**Apply to MogaNet-B teacher before re-extracting pseudo-labels.**

### New Paradigms (2025-2026)

| Paradigm | Model | COCO AP | Status |
|----------|-------|---------|--------|
| FDR (Feature Distribution Regression) | DETRPose-S (ICCV 2025) | 72.5 | No ONNX export |
| Keypoint-driven (no bbox) | ER-Pose (arXiv 2025) | TBD | Preprint, untested |
| Probabilistic OKS Heatmap | ProbPose (CVPR 2025) | ~75.5 | Integrated in mmpose |
| Flow Matching (3D) | FMPose3D (CVPR 2026) | 3D task | Relevant for CorrectiveLens |
| Prototypical Embedding | PoseBH (CVPR 2026) | — | Multi-skeleton training |

### Distillation Approaches

| Approach | Teacher→Student | Student AP | Status |
|----------|----------------|-----------|--------|
| **DWPose Distiller** | SimCC→SimCC (feature+logit KD) | 68 (from tiny) | Proven, production-ready |
| **DistilPose TDE** | Heatmap→Regression (token distillation) | 71.0 | CVPR 2023, bridges paradigm gap |

---

## Strategy Decision Tree

```
v36b completes (200 epochs)?
├── skating mAP50(P) ≥ 0.70
│   └── YOLO26s sufficient for student
│       └── Deploy with RLE, single-stage speed
│
└── skating mAP50(P) < 0.70
    ├── Option A: Try YOLO26m (67.2 AP baseline) with KD
    ├── Option B: Switch to RTMPose-s (SimCC, 71.5 AP baseline) + DWPose distillation ← RECOMMENDED
    └── Option C: Keep YOLO26s for real-time preview, RTMPose-m for accuracy-critical analysis
```

**Why Option B (RTMPose-s SimCC):**
- 71.5 AP baseline vs 63.0 (YOLO26s) — 8.5pp head start
- Compatible with rtmlib (already in project)
- DWPose distillation proven (SimCC→SimCC via feature+logit KD)
- ONNX export via rtmlib
- Expected skating AP with KD: 72-78

---

## v36b Current Status (epoch 8/200)

| Epoch | mAP50(B) | mAP50(P) | mAP50-95(P) | pose_loss | rle_loss |
|-------|----------|----------|-------------|-----------|----------|
| 4 | 0.763 | 0.747 | 0.622 | 7.378 | 0.226 |
| 5 | 0.776 | 0.730 | 0.674 | 7.212 | 0.232 |
| 6 | 0.803 | 0.694 | 0.669 | 7.074 | 0.237 |
| 7 | 0.810 | 0.747 | 0.704 | 6.884 | 0.259 |
| **8** | **0.842** | **0.734** | **0.703** | **6.788** | **0.281** |

- mAP50(P) = 0.703 — already above 0.70 threshold
- rle_loss positive and stable (no negative bug)
- cos_lr working, pose_loss decreasing steadily
- ~15 min/epoch on RTX 5090

---

## DETRPose Assessment

DETRPose (ICCV 2025) is the most interesting new model but **not recommended** for adoption:

- DETRPose-S: 72.5 AP, 11.9M params, NMS-free, real-time
- **No ONNX export path** — dealbreaker for our pipeline
- No Ultralytics/mmpose integration — academic code only
- No advantage over RTMPose-m at comparable params (72.5 vs 75.8 AP)
- Monitor for ecosystem maturation

**RT-DETR-Pose does NOT exist.** RT-DETR v2 (Baidu/Ultralytics) = detection only. DETRPose = separate academic work.

---

## Transformer vs CNN vs Mamba (2025-2026)

| Architecture | Accuracy Leader | Real-time Leader | Best For |
|-------------|----------------|-----------------|----------|
| ViT | PoseSynViT 84.3 AP | — | Accuracy-at-any-cost |
| CNN | RTMPose-m 75.8 AP | YOLO26s 2.7ms T4 | Production |
| DETR | DETRPose-L 75.5 AP | DETRPose-N real-time | NMS-free multi-person |
| Mamba/SSM | PoseMamba SOTA 3D | — | 3D lifting, temporal |

**Trend:** CNN remains production standard. Mamba relevant for 3D lifting (CorrectiveLens). ViT scaling shows diminishing returns after ~300M params.

---

## Sources

- `data/specs/2026-04-25-pose-paradigms-research.md` — full 3-agent synthesis (20+ paradigms)
- `data/specs/2026-04-25-ml-figure-skating-research.md` — broader ML research (5 agents)
- `data/plans/2026-04-25-v36-hyperparameter-review.md` — v36b hyperparameter rationale
