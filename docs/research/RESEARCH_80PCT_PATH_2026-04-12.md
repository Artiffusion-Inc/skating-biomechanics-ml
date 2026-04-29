---
title: "Research: Path to 80% Classification Accuracy"
date: "2026-04-12"
status: active
---
# Research: Path to 80% Classification Accuracy
**Date:** 2026-04-12
**Sources:** 4 research agents + Gemini Deep Research (24 papers, 33 citations) + prior research
**Status:** COMPLETE — unified plan ready

---

## Summary

Combined research from 4 parallel investigations identifies **two complementary paths** to 80% accuracy:

1. **Pretrained encoder (MotionBERT)** — leverage 3D lifting as pretext task, use learned representations
2. **Data-efficient methods (SkeletonX, Class-Balanced Loss)** — specifically designed for 5K samples / 64 classes

Both paths can be combined for maximum effect.

---

## Path 1: MotionBERT Pretrained Encoder

**Source:** github.com/Walter0807/MotionBERT (1.4K stars, Apache 2.0)

### Why It Works
- Self-supervised pretraining on 2D→3D lifting task (Human3.6M + AMASS + PoseTrack)
- Learns geometric, kinematic, physical knowledge about human motion
- Accepts 2D H3.6M 17kp directly — perfect match for our format
- One-shot: 67.4% on 120 classes (we have 65× more data per class)

### Expected Accuracy
| Configuration | Expected | Confidence |
|--------------|----------|------------|
| Current BiGRU + angles | 67.5% | Measured |
| Frozen MotionBERT + linear | 70-74% | Medium |
| MotionBERT fine-tuned + BiGRU head | 75-80% | Medium-High |
| + Joint angle fusion | 77-82% | Medium |
| + FS-Jump3D contrastive pretraining | 80-85% | Lower |

### Implementation Plan
1. Download MotionBERT-Lite (61MB) from HuggingFace
2. Extract features with frozen encoder: `get_representation(x)` → (N, T, 17, 512)
3. Feed into BiGRU head (proven architecture) + joint angles
4. Fine-tune with lr_backbone=0.00005, lr_bigru=0.0005

### Critical Files
- `lib/model/model_action.py` — ActionNet architecture
- `configs/action/MB_ft_NTU60_xsub.yaml` — fine-tuning config
- **Risk:** pretrained on daily activities, not skating. Domain gap possible.

---

## Path 2: Data-Efficient Classification

### Tier 1: Quick Wins (1-4 hours)

**Class-Balanced Loss + Label Smoothing** (+2-5pp)
- 5 lines of code: reweight by inverse effective number
- `balanced-loss` pip package available
- Stacks with ALL other methods

**Multi-stream BiGRU Ensemble** (+2-4pp)
- Average softmax outputs from: angles + velocities + raw + bone vectors
- No new research needed, use existing trained models

### Tier 2: Medium Effort (2-5 days)

**SkeletonX** (IEEE TMM 2025, github.com/zzysteve/SkeletonX) — **+8-14pp**
- Designed for 10-30 samples/class (we have 65)
- Cross-sample aggregation: same action, different performers
- Cross-performer pairs perfect for skating (same jump by different skaters)
- Requires GCN backbone

**Shap-Mix** (IJCAI 2024, github.com/JHang2020/Shap-Mix) — **+3-8pp**
- Shapley-value guided mixing for long-tailed skeleton recognition
- Preserves discriminative joints of tail classes during mixing
- Plug-in augmentation on top of any backbone

**ProtoGCN** (CVPR 2025 Highlight, github.com/firework8/ProtoGCN) — **+5-10pp**
- Learns prototypical motion patterns per class
- Designed for fine-grained distinction (similar to skating jumps)
- 144 stars, well-maintained

### Tier 3: Advanced (1-2 weeks)

**BRL** (Machine Intelligence Research 2025, github.com/firework8/BRL) — +4-7pp
- Balanced representation learning with detached learning schedule
- Separates representation learning from classifier for tail classes

**SkeletonMAE** (ICCV 2023, github.com/HongYan1123/SkeletonMAE) — +3-6pp
- Graph-based Masked Autoencoder for skeletons
- Requires external pretraining data (MMFS 26K sequences viable)

### Key Insight: GCN Can Work With Proper Balancing
GCN failed before (28.5%) due to overfitting without augmentation/balancing. With Class-Balanced Loss + SkeletonX/Shap-Mix, GCN can handle small data effectively.

---

## Datasets

### NEW: FineFS REOPENED
- 1,167 samples with GOE scores, 3D skeleton (17kp), timestamps
- Available via Baidu + Google Drive
- Previously thought dead — link is back

### NEW: VIFSS Enhanced FS-Jump3D
- github.com/ryota-skating/VIFSS
- Fine-grained jump procedure annotations (entry, jump, landing)
- 6 jump types × rotation levels = 23 element-level labels
- Best 3D skating pose dataset, used for VIFSS pre-training

### Accessible by Request
- **SkatingVerse** — large-scale, multi-task, used by VIFSS. Contact BUPT/TeleAI
- **YourSkatingCoach** — 454 jump videos, air time labels. Contact Academia Sinica Taiwan
- **FS1000** — 1,000+ videos. jingfeixia708@gmail.com

### Total Accessible Sequences
~9,500+ across FSC + MCFS + MMFS + FineFS + FS-Jump3D + FSD-10

---

## 3D Pose for Classification: NOT RECOMMENDED

**VIFSS paper evidence:** 3D coordinates from MotionAGFormer HURT performance on figure skating:
- 2D baseline: Acc 71.34%, F1@50 78.78%
- 3D (MotionAGFormer): Acc 70.17%, F1@50 76.57% (−2.21pp)
- Reason: 3D lifter fails on fast rotations, error (38mm) overwhelms discriminating features

**Correct approach:** Use 3D only for pre-training (contrastive learning), inference on 2D. This is what VIFSS and MotionBERT both do.

---

## Recommended Experiment Priority

| # | Experiment | Expected Gain | Effort | Dependencies |
|---|-----------|---------------|--------|-------------|
| E1 | Class-Balanced Loss | +2-5pp | 1h | None |
| E2 | Multi-stream Ensemble (3 BiGRU variants) | +2-4pp | 4h | None |
| E3 | MotionBERT frozen + linear classifier | +3-7pp | 4-6h | Download model |
| E4 | MotionBERT + BiGRU head + joint angles | +5-13pp | 6-8h | E3 |
| E5 | Hierarchical classification (type→element) | +3-5pp | 1-2d | Class taxonomy |
| E6 | SkeletonX cross-sample aggregation | +8-14pp | 3-5d | GCN backbone |
| E7 | MotionBERT + SkeletonX combination | +10-18pp | 5-7d | E4 + E6 |
| E8 | FS-Jump3D contrastive pre-training (VIFSS-style) | +3-5pp | 3-4d | E4 |

**Path to 80%:** E1 + E2 + E4 = 67.5 + 4 + 3 + 10 = ~75-80%
**If short:** Add E6 or E8 = ~78-85%

---

## Gemini Deep Research Additions (24 papers analyzed)

### New Methods Not In Our Research

1. **ActionMamba** (MDPI 2024) — Shallow GCN (1 layer) + Mamba selective scan for O(T) sequence modeling. Selective state filters out redundant gliding, weights explosive kinematic frames. Expected +6-8.5pp. Code: from Mamba Pyramid repo (causal-conv1d + mamba modules).

2. **SkateFormer** (ECCV 2024, github.com/KAIST-VICLab/SkateFormer) — Partition-specific attention (Skate-MSA): 4 relation types (neighboring joints, distant joints, neighboring frames, distant frames). Avoids O(T²) memory. Handles joint+bone+motion modalities.

3. **SpeLER** (ICCV 2025) — Spectral-balanced Label Propagation. Class imbalance shifts eigenvalue spectrum of affinity matrices. Energy-based reliability preserves tail class boundary samples. Expected +3-4.5pp.

4. **Skeleton-Cache** (NeurIPS 2025) — Non-parametric retrieval for tail classes. Stores embeddings in memory bank, uses cosine similarity at inference. Bypasses parametric classifier for classes with <20 samples.

5. **SUGAR** (arXiv 2025) — Skeleton + LLM text knowledge distillation. Forces skeleton embeddings to align with LLM-generated motion descriptions ("rapid counter-clockwise rotation with elevated arms").

6. **STAR++** (TCSVT 2026, github.com/cseeyangchen/STAR_pp) — Semantic alignment for multi-dataset fusion. Replaces integer labels with dense semantic text vectors. Enables merging FSC+MCFS+MMFS without label mismatch.

### Key Disagreement: 3D for Classification
- **Gemini:** Fine-tuning on AthletePose3D reduces MPJPE by 69%, achieves 83% edge error detection. 3D surpasses 2D.
- **Our research:** VIFSS paper directly shows 3D coordinates HURT classification (70.17% vs 71.34%). MotionAGFormer fails on fast rotations.
- **Resolution:** 3D helps for EDGE ERROR DETECTION (quality), not for ELEMENT CLASSIFICATION. Different tasks. Our VIFSS evidence is more directly relevant.

### Updated Experiment Priority (Gemini + Our Research Combined)

| # | Experiment | Source | Expected | Effort |
|---|-----------|--------|----------|--------|
| E1 | Class-Balanced Loss + Label Smoothing | Both | +2-5pp | 1h |
| E2 | Multi-stream BiGRU Ensemble (3 variants) | Ours | +2-4pp | 4h |
| E3 | Biomechanical features (angular velocity, flight mask, takeoff angle) | Gemini | +4-5pp | 2-3h |
| E4 | MotionBERT frozen + linear baseline | Ours | +3-7pp | 4-6h |
| E5 | MotionBERT + BiGRU + joint angles | Ours | +5-13pp | 6-8h |
| E6 | ActionMamba (shallow GCN + Mamba) | Gemini | +6-8.5pp | 2-3d |
| E7 | SkateFormer (partition attention) | Gemini | +5-8pp | 2-3d |
| E8 | SkeletonX cross-sample aggregation | Ours | +8-14pp | 3-5d |
| E9 | STAR++ semantic multi-dataset fusion | Gemini | +3-5pp | 5-7d |
| E10 | SpeLER + Skeleton-Cache for tail classes | Gemini | +3-5pp | 3-5d |

### Recommended Execution Order

**Phase 1: Quick wins (this week)**
- E1 (1h) + E2 (4h) + E3 (2-3h) → expected 67.5 + 8 = **~75.5%**

**Phase 2: Pretrained encoder (next week)**
- E4 + E5 → expected 67.5 + 10 = **~77.5%**

**Phase 3: Architecture upgrade (week 3)**
- E6 (ActionMamba) OR E7 (SkateFormer) → if Phase 2 insufficient, +6-8pp

**Phase 4: Advanced data strategies (week 4+)**
- E8 (SkeletonX) + E9 (STAR++) + E10 (SpeLER) → stack for final push to 80%+

**Most likely path to 80%:**
Phase 1 + Phase 2 = E1 + E2 + E3 + E5 = 67.5 + 2 + 3 + 4 + 10 = **~76.5%**
Add Phase 3 (ActionMamba) if needed: +6pp = **~82.5%**
