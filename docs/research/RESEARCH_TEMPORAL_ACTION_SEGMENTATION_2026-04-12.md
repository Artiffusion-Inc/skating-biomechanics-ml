---
title: "Temporal Action Segmentation (TAS) for Figure Skating — Deep Research"
date: "2026-04-12"
status: active
---

# Temporal Action Segmentation (TAS) for Figure Skating — Deep Research

**Date:** 2026-04-12  
**Sources:** arXiv papers (2410.20427, 2408.16638, 2110.08568, 2303.17959, 2412.04353, 2006.09220), GitHub repos, survey TPAMI 2023  
**Status:** ACTIVE

---

## Summary

Temporal Action Segmentation assigns a semantic label to every frame in an untrimmed video. For figure skating: automatically finding where each jump, spin, and step sequence begins and ends.

**Key finding:** Skeleton-only TAS **outperforms RGB** for figure skating due to the motion-centric nature of the sport (78.8% vs 69.1% on MMFS-63).

---

## TAS Task: Key Challenges

- **Over-segmentation** — #1 problem: many short spurious segments
- **Class imbalance** — background dominates, elements are rare
- **Long-range dependencies** — 4-minute free skate = thousands of frames
- **Boundary ambiguity** — entry transitions blend into elements

---

## Core Methods

### MS-TCN++ (Baseline)
- Multi-stage refinement: probabilities only (no raw features in later stages)
- Dual Dilated Layers: both local + global context at every layer
- Smoothing loss penalizes probability jumps between frames
- **Pros:** Feature-agnostic, efficient, battle-tested (~1000 lines PyTorch)
- **Repo:** `sj-li/MS-TCN2` (181 stars, MIT)

### ASFormer (Transformer)
- Local attention mask + hierarchical reduction for long sequences
- Improves over MS-TCN++ by ~1-3% F1 across standard benchmarks
- **Repo:** `ChinaYi/ASFormer` (135 stars, MIT)

### YourSkatingCoach — BIOES Labeling
- Sequence labeling: B(egin), I(nside), O(utside), E(nd)
- GCN + Encoder-CRF: 96.3% accuracy, F1=0.671 on air time detection
- Limitations: single skater, jump-only clips, 454 videos

### Skeleton-Only Approaches

| Method | Input | Result | Notes |
|--------|-------|--------|-------|
| LSTM-CNN (mayupei) | COCO 17kp skeleton | **0.89 F1@50** | Full competition videos |
| STGA-Net | Skeleton graph attention | Good | Spatial + temporal attention |
| Skeleton Motion Words (ICCV 2025) | Skeleton | Unsupervised | No labels needed |

**Why skeleton-only wins for skating:**
1. Excludes background/audience noise
2. Much lower dimensionality (34-dim vs 2048-dim I3D)
3. Privacy-preserving, camera-agnostic
4. Our pipeline already produces H3.6M 17kp

---

## Figure Skating Datasets

| Dataset | Videos | Labels | Skeleton | Task | Access |
|---------|--------|--------|----------|------|--------|
| **MCFS** | 271 | Per-frame | BODY_25 (OpenPose) | TAS | Available |
| **MMFS** | 1176 | Clip-level | COCO 17kp (HRNet) | AR + AQA | Available |
| **FineFS** | 1167 | Start/end times | 17kp 3D | AQA + TAS | Baidu/GDrive |
| **YourSkatingCoach** | 454 | BIOES | COCO 17kp | Air time | In paper |
| **FS-Jump3D** | ~100 | Procedure-aware | 3D mocap | TAS | GitHub |

**Quality issues:** MCFS has 56% frames with ≥1 missing joint; start/end labels ~57 frames (~2s) off.

---

## Evaluation Metrics

| Metric | Measures | Use |
|--------|----------|-----|
| **F1@50** | Strict boundary quality | **Primary** — punishes over-segmentation |
| F1@25 | Moderate tolerance | Standard comparison |
| Edit score | Levenshtein distance | Catches over-segmentation |
| Frame-wise Acc | % correct frames | Baseline (dominated by long classes) |

---

## Recommended Strategy

### Phase 1: Coarse Segmentation (Jump/Spin/Background)
- **Backbone:** MS-TCN++ with H3.6M 17kp skeleton features
- **Data:** MCFS+MMFS 222 shared routines (per-frame labels from MCFS, skeletons from MMFS)
- **Expected:** F1@50 > 0.85 (based on mayupei's results)
- **Classes:** ~10 (background, jump, spin, step_sequence, choreo_sequence)

### Phase 2: Fine-Grained Classification
- Two-stage: coarse segmentation → per-segment classification
- **Data:** MMFS (11,671 clips, 256 categories) or FineFS
- **Model:** ST-GCN or CTRGCN on skeleton clips

### Phase 3: Sub-Phase BIOES Labeling
- Within each element: approach → take-off → flight → landing → exit
- **Data:** YourSkatingCoach (454 clips) or FS-Jump3D

### Data Sources by Phase

| Phase | Dataset | Status |
|-------|---------|--------|
| 1 (coarse) | MCFS + MMFS overlap | Already downloaded |
| 2 (classification) | MMFS | Already downloaded |
| 3 (sub-phase) | FineFS or YourSkatingCoach | Need FineFS download |

---

## Key Questions Answered

**Q: Can we do TAS from skeleton alone?**  
Yes. MMFS benchmark: ST-GCN 78.8% vs PAN RGB 69.1%. mayupei's LSTM-CNN achieves 0.89 F1@50 with only COCO 17kp.

**Q: Minimum data needed?**  
~50-100 annotated routines for coarse segmentation. 200+ for F1@50 > 0.85. FineFS 1167 routines supports 10-15 class TAS.

**Q: Does YourSkatingCoach handle full programs?**  
No — isolated jump clips only (1.3-10s). For full programs use MCFS+MMFS or FineFS.

---

## Open-Source Implementations

| Project | Stars | License | Best For |
|---------|-------|---------|----------|
| MS-TCN2 | 181 | MIT | General TAS baseline |
| ASFormer | 135 | MIT | Transformer TAS |
| mayupei/figure-skating | 2 | — | Skating-specific LSTM-CNN |
| awesome-tas | 246 | — | Curated paper list |

---

## References

1. Chen et al., "YourSkatingCoach", arXiv:2410.20427, 2024
2. Tanaka et al., "3D Pose-Based TAS for Figure Skating", ACM MM Workshop 2024
3. Li et al., "MS-TCN++", TPAMI 2020, arXiv:2006.09220
4. Yi et al., "ASFormer", BMVC 2021, arXiv:2110.08568
5. Liu et al., "Diffusion Action Segmentation", ICCV 2023
6. Gong et al., "ActFusion", NeurIPS 2024
7. Ding et al., "TAS Survey", TPAMI 2023, arXiv:2210.10352
8. Liu et al., "MMFS", ACM MM 2023
9. Ji et al., "FineFS", ACM MM 2023
10. Gokay et al., "Skeleton Motion Words", ICCV 2025
11. mayupei, "Skeleton-Based Figure Skating TAS", 2025

**Full 659-line analysis preserved in git history.**
