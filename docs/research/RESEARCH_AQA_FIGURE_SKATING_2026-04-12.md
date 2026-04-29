---
title: "Action Quality Assessment for Figure Skating: Deep Research"
date: "2026-04-12"
status: active
---

# Action Quality Assessment for Figure Skating: Deep Research

**Date:** 2026-04-12  
**Context:** Predict GOE (Grade of Execution) scores from skeleton poses. Current Ridge regression on frame statistics achieves MAE=0.78 GOE, correlation=0.206 on FineFS (6065 segments). Needs significant improvement.

---

## Executive Summary

**Critical gap:** No published paper does **per-element GOE prediction from skeletons on FineFS**. All existing work predicts **whole-program TES** from **I3D video features** (not skeleton). Our task is genuinely novel and under-explored.

**Best approaches found:**
1. **Mamba Pyramid (ACM MM 2025)** — SOTA whole-program scoring (rho_TES=0.80) but uses video+audio, not skeleton
2. **HP-MCoRe (IEEE TIP 2025)** — best skeleton-based AQA approach, uses hierarchical pose guidance + contrastive regression
3. **OOFSkate (MIT, 2026)** — proprietary but validates physics-based metrics (height, angular velocity) as quality proxies
4. **CoRe (ICCV 2021)** — foundational contrastive regression; handles judge subjectivity better than direct regression

---

## Key Papers

### 1. Mamba Pyramid (ACM MM 2025)
- Two-stream: TES (visual-only) + PCS (visual+audio)
- Multi-scale Mamba Pyramid with 6-level score regression
- **Results on FineFS:** rho_TES=0.80, rho_PCS=0.96 (whole-program)
- **Caveat:** Spearman correlation, not MAE. Video+audio input, not skeleton.
- Zero-shot transfer: Fis-V rho_TES=0.79, FS1000 rho_TES=0.85

### 2. HP-MCoRe (IEEE TIP 2025)
- **Most relevant for skeleton-based AQA**
- Multi-scale dynamic visual-skeleton encoder + procedure segmentation
- Skeleton features as **physics structural priors** to guide visual learning
- Multi-stage contrastive regression: group by quality → learn within/between differences
- Results on FineDiving: SRCC=0.9365 (SOTA)

### 3. OOFSkate (MIT Sports Lab, 2026)
- iOS app, partnered with U.S. Figure Skating, deployed at 2026 Olympics with NBC Sports
- Physics-based metrics: jump height, angular velocity, peak angular velocity, time-to-peak
- GOE estimated by comparing against elite athlete database
- **Validates our proxy-feature direction** — no blade edge detection
- Closed-source, no published accuracy numbers

### 4. "From Beats to Scores" (CVPRW 2025)
- Predecessor to Mamba Pyramid; MLLM with audio-guide segmentation
- **Key finding:** Audio-guide segmentation significantly helps TES and PCS
- Video-only TES=0.71, audio-only=0.65, audio-guide=0.76

### 5. CoRe — Group-aware Contrastive Regression (ICCV 2021)
- Groups samples by score range → learns relative differences within groups
- Handles judge subjectivity variance better than direct regression
- Foundational for HP-MCoRe and subsequent methods
- FineDiving: SRCC=0.9061

### 6. Skeleton-Based Figure Skating AQA (IEEE 2022)
- ST-GCN + OpenPose on MIT-Skate and FIS-V
- **Skeleton outperforms RGB** for figure skating AQA
- More robust to camera angle changes

---

## What Features Carry Quality Signal?

**Why current features (mean+std position) fail:**
1. Mean position dominated by camera angle
2. Std captures spread, not quality
3. No temporal information (WHEN matters more than average WHERE)
4. No inter-joint relationships

**Tier 1 (easy, high signal):**
- Joint angle time series (17 angles × T frames)
- Velocity magnitude per joint (smoothness proxy)
- Phase-relative timing (takeoff/peak/landing as ratio of duration)
- Angular velocity

**Tier 2 (moderate effort, strong signal):**
- CoM trajectory + velocity/acceleration profiles
- Jerk (derivative of acceleration)
- Bone length consistency
- Left-right symmetry

**Tier 3 (needs model, proven effective):**
- DTW distance to reference execution
- Learned ST-GCN embeddings
- Contrastive features (CoRe-style)

---

## FineFS Dataset

- **1,167 samples** (729 short program, 438 free skating)
- 570 male, 597 female athletes
- 25 fps, ~5,000 frames/video
- Per-element annotations: timing, category, scores
- 3D skeletons (17kp) in camera space
- Official train/test split provided

**How papers use FineFS:**

| Paper | Task | Input | Best Result |
|-------|------|-------|-------------|
| LUSD-Net | Whole-program TES/PCS | I3D video | TES=0.73, PCS=0.84 |
| Beats to Scores | Whole-program + text | I3D + VGGish | TES=0.76, PCS=0.88 |
| **Mamba Pyramid** | **Whole-program** | **I3D + VGGish** | **TES=0.80, PCS=0.96** |

**Critical:** No paper reports per-element GOE from skeletons. Our task is novel.

---

## Regression Approach: Recommendation

| Approach | Pros | Cons | Best Performance |
|----------|------|------|-----------------|
| Direct Regression | Simple | Sensitive to noise | Baseline only |
| Pairwise Ranking | Robust | Needs pairs, O(n²) | Good for small data |
| **Contrastive Regression (CoRe)** | **Best of both** | **More complex** | **SOTA across benchmarks** |

**Recommendation for our per-element GOE:**
1. Group elements by type (axel, lutz, etc.)
2. Within each type, CoRe-style grouping by GOE range
3. Add ordinal loss to respect GOE ordering

Rationale: Judge subjectivity variance is known high. Contrastive regression designed for this.

---

## Clear Next Steps

### Priority 1: Better Features (1-2 days, highest ROI)
Replace mean+std position with:
1. Joint angle time series + statistical moments/DCT coefficients
2. Phase-relative timing (takeoff/peak/landing ratios)
3. CoM height range, velocity smoothness, angular velocity peak
4. DTW distance to reference (we have infrastructure)

**Expected:** MAE 0.78 → <0.6, correlation 0.206 → >0.4

### Priority 2: Lightweight Temporal Model (2-3 days)
- ST-GCN on H3.6M skeleton graph + small TCN/Transformer (4-6 layers)
- MSE + contrastive loss, per-element-type models
- **Expected:** MAE <0.5, correlation >0.5

### Priority 3: Contrastive Regression Framework (3-4 days)
- Group FineFS elements by type + GOE bin
- CoRe-style group-aware contrastive loss
- Combine with ST-GCN encoder
- **Expected:** MAE <0.4, correlation >0.6

### Priority 4: Reference-Based DTW Quality Score (1-2 days)
- Top-10 GOE executions per element type as references
- DTW-align each segment, use mean distance as quality feature
- Most biomechanically principled — aligns with coach evaluation

---

## Summary of Key Findings

1. **No paper does per-element GOE from skeletons on FineFS.** Our task is novel.
2. **Skeleton-based AQA works better than video for sports** — validates our skeleton-only approach.
3. **Quality signal is in temporal dynamics, not spatial averages.** Joint angles, velocity profiles, phase timing, CoM smoothness carry signal.
4. **Contrastive regression (CoRe) dominates AQA** — handles judge subjectivity better than direct regression.
5. **Procedure segmentation matters** — HP-MCoRe shows sub-action processing significantly improves prediction.
6. **OOFSkate uses physics-based metrics** — validates our proxy-feature direction.
7. **DTW-based comparison against reference** is proven (QAQA 2025) and we already have infrastructure.

---

## References

1. Wang et al., "Mamba Pyramid", ACM MM 2025. arXiv:2508.16291
2. Wang et al., "From Beats to Scores", CVPRW 2025
3. Qi et al., "HP-MCoRe", IEEE TIP 2025
4. Yu et al., "CoRe", ICCV 2021
5. Zhou et al., "AQA Survey", arXiv:2412.11149, 2024
6. Ji et al., "FineFS", 2023
7. Fu et al., "QAQA", Sensors 2025
8. Chen et al., "YourSkatingCoach", arXiv:2410.20427, 2024
9. Xu et al., "Skeleton-Based AQA for Figure Skating", JVCIR 2022
10. MIT News, "AI for Olympic Skaters", 2026

**Full 494-line analysis preserved in git history.**
