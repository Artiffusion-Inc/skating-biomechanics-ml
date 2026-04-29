---
title: "VIFSS: View-Invariant Figure Skating-Specific Pose Representation"
date: "2026-04-12"
status: active
citekey: "tanaka2025vifssviewinvari"
---

# VIFSS: View-Invariant Figure Skating-Specific Pose Representation

**Paper:** arXiv:2508.10281  
**Authors:** Ryota Tanaka, Tomohiro Suzuki, Keisuke Fujii (Nagoya University / RIKEN AIP)  
**Code:** https://github.com/ryota-skating/VIFSS (Apache-2.0)  

---

## Problem

Temporal Action Segmentation for figure skating jumps from broadcast video faces two challenges:
1. **Sparse annotations** — only ~9% of frames labeled
2. **View sensitivity** — 2D pose projections vary with camera angle; direct 3D regression bottlenecked by unseen rotations (quads)

**Solution:** Learn latent pose embeddings rather than raw 2D/3D coordinates via contrastive pre-training (view-invariant) + fine-tuning (domain-specific).

---

## Architecture: JointFormer Encoder

- **Input:** 2D pose [B, 17, 2] (H3.6M format)
- **Embedding:** SemGraphConv(2 → 128) with skeleton adjacency
- **Backbone:** 4-layer Transformer (d_model=128, 8 heads)
- **Intermediate supervision:** 3D pose + error prediction after each layer
- **Output:** 36-dim embedding split into z_pose [32] + z_view [4]
- **Params:** ~2M, inference ~1ms/frame on GPU

---

## Pre-Training: 3D Multi-View Contrastive

**Datasets:** Human3.6M, MPI-INF-3DHP, AIST++, FS-Jump3D — unified to H3.6M 17kp.

**Virtual camera projection:** Random azimuth [-180°, +180°], elevation [-30°, +30°], distance [5, 10]. Ground plane aligned via RANSAC.

**Loss:** `L_total = L_pose + 10 * L_view + L_regularization`
- **L_pose:** Barlow Twins (negative-sample-free) on pose embeddings
- **L_view:** MSE between cos-sim of view embeddings and actual camera directions
- **L_reg:** Variance + KL uniform anti-collapse terms

**Training:** batch 1024, Adam, lr=1e-3, 60 epochs.

---

## Fine-Tuning: Action Classification

**Architecture:** Pre-trained JointFormer (frozen) → 2-layer BiGRU (hidden=128) → temporal max pooling → FC → 28 classes (SkatingVerse)

**SkatingVerse:** 19,993 train clips, 8,586 test, 28 classes (23 jumps × 4 rotations + spins + none). **Not publicly downloadable** — challenge-era distribution only.

---

## Results

### Element-level TAS (23 jump labels)

| Feature | Acc | F1@50 |
|---------|-----|-------|
| 2D pose | 71.34 | 78.78 |
| 3D pose (MotionAGFormer) | 70.17 | 76.57 |
| **VIFSS** | **85.82** | **92.56** |
| scratch-FSS (no pre-train) | 82.72 | 89.65 |

### Set-level TAS (6 jump types)

| Feature | Acc | F1@50 |
|---------|-----|-------|
| 2D pose | 78.55 | 84.17 |
| 3D pose | 79.89 | 86.56 |
| **VIFSS** | **89.91** | **94.68** |

### Key Ablations

- **FS-Jump3D adds +1.9 to +4.0 F1@50** — more impactful for 3D coords than embeddings
- **Procedure-aware annotations help raw features most** (+7.75 to +13.78 F1@50); VIFSS gains only +0.75
- **Low-data regime:** With 1% fine-tuning data, pre-trained VIFSS achieves >60% F1@50; scratch training fails entirely

---

## Reproducibility Assessment

### Available (can replicate)
- Code: Apache-2.0, ~500 lines, clean
- Pre-training data: Human3.6M, MPI-INF-3DHP, AIST++, FS-Jump3D (all public)
- Architecture: fully documented
- Skeleton: H3.6M 17kp — **directly compatible with our pipeline** (halpe26_to_h36m)

### Blocked
- **SkatingVerse** — no public download; blocks fine-tuning (19,993 clips)
- **TAS annotations** — promised "upon publication", not released
- **Pre-trained weights** — must train from scratch (~2-4h on RTX 3050 Ti)

### Substitutes
- **Figure-Skating-Classification** (HuggingFace): 5,168 clips, 64 classes, COCO 17kp → map to H3.6M
- **MMFS:** 26,198 sequences, 256 categories

---

## Practical Assessment for Our Project

**Strengths:**
- Same skeleton format (H3.6M 17kp) — zero conversion
- View-invariant embeddings solve single-camera phone video from arbitrary angles
- Pre-training uses only 3D pose data (no video) — runs offline
- Apache-2.0, clean code
- Real-time inference (~1ms/frame)

**Weaknesses:**
- Pre-training requires 3D datasets (we have FS-Jump3D, need others)
- Fine-tuning requires SkatingVerse or substitute
- Per-frame encoder — temporal modeling deferred to BiGRU
- 17kp only (no HALPE26 foot keypoints)

**Compute:** Pre-training ~2-4h on RTX 3050 Ti. Fine-tuning ~1-2h. Inference real-time.

---

## Comparison

| Method | Input | View-Invariant | Skating-Specific | TAS F1@50 (Elem) |
|--------|-------|---------------|------------------|-------------------|
| MCFS | 2D pose | No | No | ~70 |
| Tanaka 2024 | 3D pose | Partial | No | ~77 |
| **VIFSS** | **2D → embedding** | **Yes** | **Yes** | **92.56** |

---

## References

- VIFSS: Tanaka et al., arXiv:2508.10281, 2025
- JointFormer: Lutz et al., ICPR 2022
- Barlow Twins: Zbontar et al., ICML 2021
- FS-Jump3D: Tanaka et al., ACM MMSports 2024
- FACT (TAS model): Lu & Elhamifar, CVPR 2024
- SkatingVerse: Gan et al., IET Computer Vision 2024

**Full 445-line analysis preserved in git history.**
