# YOLO26-Pose Fine-tuning Research Report

**Date:** 2026-04-15

**Linked notes:**
- Current progress → @HP_SEARCH.md
- Planning mistakes → @POST-MORTEM.md
- Iterative strategy → @ITERATIVE_RESEARCH_STRATEGY.md
- Research comparison → @COMPREHENSIVE_RESEARCH_COMPARISON.md
**Goal:** Beat RTMO baseline (19.1px mean error) with fine-tuned YOLO26-Pose
**Dataset:** AthletePose3D S1 train / S2 val (COCO 17kp GT)

---

## Research Questions

### Primary Question
**Can YOLO26-Pose fine-tuned on AthletePose3D achieve <15px mean error on figure skating pose estimation?**

### Secondary Questions
1. What hyperparameters minimize PKE (Pixel Keypoint Error) for skating-specific poses?
2. How does freeze depth affect generalization to held-out subject (S2)?
3. Which augmentation strategy works best for ice skating (vertical sport)?
4. Is 15% COCO-Pose mixing beneficial or harmful for domain-specific accuracy?

---

## Hypotheses

### H1: Learning Rate
**Hypothesis:** LR=0.0005 achieves best balance between convergence and overfitting

| LR | Expected Outcome | Rationale |
|----|------------------|-----------|
| 0.001 | Overfit, high variance | Too aggressive for small domain (226K frames, 1 subject) |
| **0.0005** | **Optimal** | Official recipe, gentle decay (lrf=0.88) |
| 0.0001 | Underfit, slow convergence | Too conservative, wastes compute |

**Success metric:** mAP50 on S2 val after 50 epochs

### H2: Freeze Depth
**Hypothesis:** freeze=10 prevents catastrophic forgetting while allowing domain adaptation

| Freeze | Expected Outcome | Rationale |
|--------|------------------|-----------|
| 0 | Overfit S1, fail S2 | Unconstrained fine-tuning memorizes single subject |
| 5 | Moderate overfit | Some backbone adaptation, but risky |
| **10** | **Best generalization** | Gandhi et al. 2025: freeze=10 + mixed data = +10% mAP |
| 20 | Underfit | Too constrained, no domain adaptation |

**Success metric:** PCK@0.05 on critical joints (ankles, knees)

### H3: Augmentation Strategy
**Hypothesis:** Image-level augmentation only (mosaic + mixup + fliplr), no SkeletonMix

| Augmentation | Expected Outcome | Rationale |
|--------------|------------------|-----------|
| **Mosaic 0.9** | **Beneficial** | Adds background diversity |
| **Mixup 0.05** | **Beneficial** | Regularization, prevents overfitting |
| **Fliplr 0.5** | **Beneficial** | Symmetric task, flip_idx handles keypoints |
| **Copy-paste 0.0** | **Harmful** | GCN exp: mirror = -4.1pp classification |
| **Rotation 0.0** | **N/A** | Vertical sport, rotation unnatural |

**Success metric:** Training stability (val loss curve smoothness)

### H4: Domain Mixing
**Hypothesis:** 15% COCO-Pose mix improves generalization without domain pollution

| Mix | Expected Outcome | Rationale |
|-----|------------------|-----------|
| 0% | Overfit S1 subject | Body proportion bias (Ni et al. 2025) |
| **15%** | **Optimal** | Gandhi et al.: +10% mAP, <0.1% forgetting |
| 30%+ | Domain pollution | Too much generic pose, loses skating specificity |

**Success metric:** mAP50-95 (stricter than mAP50)

### H5: Model Size
**Hypothesis:** yolo26s-pose achieves optimal speed/accuracy tradeoff for RTX 4090

| Model | Expected Outcome | Rationale |
|-------|------------------|-----------|
| yolo26n | Too weak for keypoints | 7MB, lacks capacity for 17kp detail |
| **yolo26s** | **Optimal** | 24MB, proven on COCO-Pose |
| yolo26m+ | Diminishing returns | 50MB+, slower training, marginal gain |

**Success metric:** Mean PKE < 15px (vs RTMO 19.1px)

---

## Baseline Comparison

| Model | Mean PKE (px) | mAP50 | PCK@0.05 (ankles/knees) | Speed (ms) |
|-------|---------------|-------|-------------------------|------------|
| RTMO (rtmlib) | **19.1** | ? | ? | ? |
| YOLO26s pretrained | ? | ? | ? | ? |
| **YOLO26s fine-tuned** | **<15 (target)** | **>baseline** | **>baseline** | **<baseline** |

### ⚠️ CRITICAL: Baseline Metrics Research (2026-04-15)

**RTMO on COCO (detection-based mAP):**
- RTMO-s (9.9M params): mAP50-95 = **0.677** on COCO val2017
- RTMO-m (22.6M params): mAP50-95 = **0.715**
- RTMO-l (44.8M params): mAP50-95 = **0.748**

**Our result (YOLO26n-pose batch64):**
- mAP50-95(P) = **0.406** on AthletePose3D val (2198 images)
- **NOT directly comparable** — different datasets!

**Why NOT comparable:**
1. **Dataset mismatch:** COCO (everyday poses) vs AthletePose3D (skating + athletics)
2. **Metric paradigm:** RTMO uses standard detection mAP, but domain is completely different
3. **Training data:** RTMO trained on massive datasets vs our 25K subset

**AthletePose3D Paper Baseline:**
- Model: MogaNet-B (570MB, heatmap-based)
- Method: Heatmap COCO mAP (different from detection mAP)
- Their metric: ~70% AP on COCO (heatmap-based)
- **Our metric: 0.406 detection-based**
- **Conclusion:** Heatmap mAP ≠ Detection mAP — NOT comparable!

**Key Insight:**
- **0.406 is reasonable baseline** for YOLO26n-pose detection-based on AthletePose3D
- **Comparison target:** +10% improvement (≥0.45) = good result
- **Failure signal:** All configs within ±5% (0.38-0.42) = no clear winner

**Sources:**
- RTMO paper: arxiv.org/html/2312.07526v2 (Table 1)
- AthletePose3D paper: https://arxiv.org/abs/2503.07499
- AthletePose3D GitHub: https://github.com/calvinyeungck/AthletePose3D

---

## Experimental Design

### Phase 1: Hyperparameter Search (3 hours, 10 configs)

**Current status:** See @HP_SEARCH.md for real-time progress

**Model:** yolo26n-pose (fast iteration)
**Data:** 25K subset (10% of full training)
**Resolution:** 640px (fast training)
**Validation:** Offline (save_period=10)

| Config | LR | Freeze | Mosaic | Fliplr | Mixup | Batch |
|--------|-----|--------|--------|--------|-------|-------|
| hp_lr001_640 | 0.001 | 5 | 0.9 | 0.5 | 0.05 | 16 |
| hp_lr0005_640 | **0.0005** | 5 | 0.9 | 0.5 | 0.05 | 16 |
| hp_lr0001_640 | 0.0001 | 5 | 0.9 | 0.5 | 0.05 | 16 |
| hp_lr0005_f10_640 | 0.0005 | **10** | 0.9 | 0.5 | 0.05 | 16 |
| hp_lr0005_f20_640 | 0.0005 | 20 | 0.9 | 0.5 | 0.05 | 16 |
| hp_lr0005_f0_640 | 0.0005 | 0 | 0.9 | 0.5 | 0.05 | 16 |
| hp_lr0005_mos0_640 | 0.0005 | 5 | 0.0 | 0.5 | 0.05 | 16 |
| hp_lr0005_fliplr0_640 | 0.0005 | 5 | 0.9 | 0.0 | 0.05 | 16 |
| hp_lr0005_mixup01_640 | 0.0005 | 5 | 0.9 | 0.5 | **0.1** | 16 |
| hp_lr0005_batch64_640 | 0.0005 | 5 | 0.9 | 0.5 | 0.05 | **64** |

### Phase 2: Full Training (6 hours, 2 configs)

**⚠️ CRITICAL PREREQUISITE:** Phase 2 ONLY starts after Phase 1 confirms clear winner.

**See @ITERATIVE_RESEARCH_STRATEGY.md for complete decision tree.**

**Go/No-Go Criteria for Phase 1 → Phase 2 transition:**
- [ ] Clear winner (mAP > baseline + 5%, statistically significant)
- [ ] Reproducible on 2 different subsets
- [ ] No severe overfitting (train-val gap < 0.1)
- [ ] Documented in @RESEARCH_REPORT.md

**If Phase 1 FAILS criteria:** Do NOT proceed. Start new research cycle (see @ITERATIVE_RESEARCH_STRATEGY.md Cycle 2+).

---

**Model:** yolo26s-pose (final model)
**Data:** 285K frames (226K AP3D + 59K COCO)
**Resolution:** 1280px (full detail)
**Validation:** Online (early stopping, patience=20)

| Config | LR | Freeze | Epochs | imgsz |
|--------|-----|--------|--------|-------|
| full_1280_best | **from HP** | **from HP** | 100 | 1280 |
| full_1280_alt | **2nd best** | **2nd best** | 100 | 1280 |

---

## Success Criteria

### 🎯 FINAL GOAL (Critical Requirement)
**Find teacher pose extraction model that:**
1. ✅ **Significantly outperforms ALL baselines:**
   - RTMO-s, RTMO-m (heatmap-based)
   - YOLO26n, YOLO26s, YOLO26m pretrained (COCO)
   - YOLOv8x-pose pretrained
   - MogaNet-B (AthletePose3D paper baseline)
2. ✅ **NO overfitting:** train-val gap < 10% mAP
3. ✅ **NO bias fitting:** generalizes to held-out subject S2
4. ✅ **Stable training:** convergence < 50 epochs, no NaN/crash

**STOP CONDITION:** Only proceed to Phase 2 (pseudo-labeling) when ALL above criteria are MET.

**If criteria NOT met after 3 research cycles:**
- Re-evaluate project approach
- Consider using pretrained directly
- Document failure analysis in @POST-MORTEM.md

### Minimum Viable Success
- [x] Mean PKE < 19.1px (match RTMO baseline) ⚠️ NOT COMPARABLE
- [ ] PCK@0.05 (ankles) > baseline
- [ ] Training completes without NaN/crash
- [ ] **CRITICAL:** Fine-tuned > pretrained baseline

**⚠️ CRITICAL DISCOVERY (2026-04-15):**
- **Pretrained YOLO26n:** mAP50-95(P) = **0.604**
- **Fine-tuned batch64:** mAP50-95(P) = **0.406**
- **Performance drop: -49%**

**Updated Success Criteria:**
- [ ] **NEW:** Fine-tuned model > pretrained baseline (≥0.604)
- [ ] If pretrained wins: **SKIP fine-tuning**, use pretrained directly
- [ ] Target for Phase 1: Find config achieving **>0.604**

**Implication:**
- HP search may be **invalid** if pretrained > all fine-tuned configs
- Need to compare ALL configs against pretrained baselines
- **Phase 2 may be unnecessary** if pretrained is already optimal

### Target Success
- [ ] Mean PKE < 15px (21% improvement)
- [ ] PCK@0.05 (ankles) > 80%
- [ ] ONNX export works
- [ ] Inference speed < RTMO

### Stretch Goals
- [ ] Mean PKE < 12px (37% improvement)
- [ ] PCK@0.05 (ankles) > 85%
- [ ] TensorRT export works
- [ ] Per-joint analysis complete

---

## Progress Tracking

### Completed
- [x] Server setup (2x RTX 4090)
- [x] Data downloaded (AP3D, COCO, SkatingVerse)
- [x] Models downloaded (yolo26n, yolo26s)
- [x] Research report created

### In Progress
- [ ] HP search configs created
- [ ] tmux sessions setup
- [ ] HP search experiments running

### Pending
- [ ] HP search results analysis
- [ ] Full training execution
- [ ] Per-joint PCK evaluation
- [ ] ONNX export
- [ ] Benchmark vs RTMO

---

## Notes

### Session 1 (2026-04-15)
- Initial research setup
- Created 10 HP search configs
- Goal: Find best LR, freeze, augmentation combo

### Open Questions
1. Should we include SkatingVerse pseudo-labels in round 1?
   - Research says "label quality minimal impact for feature transfer"
   - But adds 168K frames from diverse skaters
   - **Decision:** Defer to round 2 (focus on GT quality first)

2. Is imgsz=1280 necessary or can we use 640?
   - 1280 preserves wrist/ankle detail from 1920px source
   - But 2x slower training
   - **Decision:** HP search @ 640, full train @ 1280

3. Should we use S2 val for HP search or separate split?
   - S2 is held-out subject (correct validation)
   - Risk: overfitting to S2 if used for HP selection
   - **Decision:** Use S2 for HP search (it's the real target)

---

## References

- Gandhi et al. 2025: freeze=10 fine-tuning with mixed data yields +10% mAP
- Ni et al. 2025: Single-subject data creates body-proportion bias
- Jaus et al. 2025: "Label quality has minimal impact for feature transfer"
- Drolet-Roy et al. 2026: 12K diverse sports images = +17.3 AP
- Wang et al. 2020: Viewpoint diversity > correlation risk
- DeepLabCut: k-means selects visually diverse poses
