# Iterative Research Strategy: YOLO26-Pose Fine-tuning

**Date:** 2026-04-15
**Status:** Phase 1 in progress (HP search running)

**🎯 FINAL GOAL:** Find teacher model that significantly outperforms ALL baselines (RTMO, YOLO, MogaNet-B) without overfitting or bias fitting.

**STOP CONDITION:** Only proceed to Phase 2 when:
- [ ] mAP > ALL baselines by >5%
- [ ] No overfitting (train-val gap < 10%)
- [ ] Generalizes to S2 (held-out subject)
- [ ] Stable training (< 50 epochs)

**Linked notes:**
- Current HP search → @HP_SEARCH.md (comparison table)
- Hypotheses & success criteria → @RESEARCH_REPORT.md (updated)
- Lessons learned → @POST-MORTEM.md
- Phase 2 plan → @PHASE2_RESEARCH_BASED_PLAN.md
- Research comparison → @COMPREHENSIVE_RESEARCH_COMPARISON.md

---

## Core Philosophy

**Если гипотезы не подтверждаются → продолжаем research цикл, не переходим к Phase 2.**

❌ **НЕ ПРИНИМАЕМ:** "Достаточно хороших результатов"
✅ **ПРИНИМАЕМ ТОЛЬКО:** "Статистически значимое улучшение над baseline"

---

## Research Cycle (Repeat Until Success)

```
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH CYCLE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 📚 RESEARCH (writing-research skill)                    │
│     ├─ tvly search "YOLO26 pose hyperparameter"             │
│     ├─ gdr research "YOLO26 fine-tuning best practices"    │
│     ├─ gh search repos "yolo pose figure skating"           │
│     └─ Read papers (arXiv, CVPR, ICCV)                      │
│                                                             │
│  2. 💡 FORMULATE HYPOTHESIS                                 │
│     ├─ Parameter X improves metric Y                        │
│     ├─ Based on research source Z                          │
│     └─ Success criteria defined                            │
│                                                             │
│  3. 🧪 TEST ON SMALL SUBSET (25K frames, 640px)            │
│     ├─ Fast iteration (1-2 hours)                          │
│     ├─ 10-20 configs in parallel                           │
│     └─ Statistical significance test                        │
│                                                             │
│  4. 📊 ANALYZE RESULTS                                     │
│     ├─ Hypothesis confirmed? → DOCUMENT → Next cycle       │
│     ├─ Hypothesis rejected? → DOCUMENT → Research more     │
│     └─ Inconclusive? → More data needed → Rerun           │
│                                                             │
│  5. 📝 DOCUMENT IN MD FILES                                 │
│     ├─ @HP_SEARCH.md — real-time progress                  │
│     ├─ @RESEARCH_REPORT.md — hypotheses status             │
│     ├─ @POST-MORTEM.md — lessons learned                   │
│     └─ @ITERATIVE_RESEARCH_STRATEGY.md — cycle update      │
│                                                             │
│  6. 🔄 DECIDE                                              │
│     ├✅ Strong teacher candidate (mAP > baseline + 10%)    │
│     │   → Go to Phase 2 (Full Training)                    │
│     │                                                         │
│     └❌ No clear winner                                      │
│        → REPEAT cycle with new hypothesis                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Success Criteria for Phase 1 → Phase 2 Transition

### Minimum Requirements (ALL must be met)

| Criterion | Threshold | How to Measure |
|-----------|-----------|----------------|
| **Validation mAP** | > baseline + 5% | RTMO baseline: 19.1px PKE |
| **Statistical significance** | p < 0.05 | Compare top 3 configs |
| **Stability** | < 5% variance across last 10 epochs | No wild fluctuations |
| **Generalization** | Train mAP - Val mAP < 0.1 | No severe overfitting |
| **Reproducibility** | Same result on 2 different subsets | Not random luck |

### Red Flags (STOP, do NOT proceed to Phase 2)

- 🚨 All configs within 2% of each other → **No clear signal**
- 🚨 Best config has train-val gap > 0.15 → **Overfitting**
- 🚨 Results contradict research papers → **Data issue**
- 🚨 Different subsets give different winners → **Not robust**

---

## Phase 1 Current Status

### What We're Testing Now (HP Search)

**10 configs on 25K subset (10% of full data):**

| Config | Hypothesis | Expected Outcome |
|--------|-----------|------------------|
| lr001 / lr0005 / lr0001 | LR tuning | lr0005 optimal |
| f0 / f10 / f20 | Freeze depth | f10 prevents overfit |
| mos0 | Mosaic=0.0 | Domain-specific benefit |
| fliplr0 | No flip | Symmetric task penalty |
| mixup01 | More mixup | Regularization helps |
| batch64 | Large batch | Faster convergence |

**Status:** See @HP_SEARCH.md for real-time results

**⚠️ CRITICAL RULE (from @POST-MORTEM.md):**
```yaml
# EVERY training config MUST have:
val: true        # NO EXCEPTIONS!
save_period: 10  # Checkpoint every 10 epochs
patience: 20     # Early stopping
```

**Why:** HP search used `val: false` → couldn't monitor overfitting → wasted compute.

### Next Steps After HP Search Completes

1. **Run validation on all checkpoints** (see @HP_SEARCH.md "Next Actions")
2. **Analyze results:**
   - If clear winner (mAP diff > 5%, significant) → **PROCEED TO PHASE 2**
   - If no clear signal (diff < 2%) → **NEW RESEARCH CYCLE**

---

## If Phase 1 Fails: New Research Cycle Template

### Cycle 2: Augmentation Deep Dive (IF NEEDED)

**Trigger:** HP search shows no clear winner, or augmentation effects unclear

**⚠️ CRITICAL PRE-STEP (from @POST-MORTEM.md):**
```bash
# BEFORE starting ANY new fine-tuning experiments:
# 1. Validate ALL pretrained baselines
yolo val model=yolo26s-pose.pt data=ap3d device=0
yolo val model=yolo26m-pose.pt data=ap3d device=0
yolo val model=yolov8x-pose.pt data=ap3d device=0

# 2. Collect results → comparison table

# 3. ONLY THEN start fine-tuning (if pretrained < potential improvement)
```

**Research Questions:**
1. Is mosaic actually helpful for vertical sports?
2. Should we disable ALL geometric augmentations?
3. What mixup ratio optimizes generalization?

**Hypotheses:**
- **H2.1:** mosaic=0.5 (moderate) > mosaic=0.9 (default) for skating
- **H2.2:** mixup=0.0 (no mixup) > mixup=0.05 for pose estimation
- **H2.3:** degrees=5.0 (small rotation) > degrees=0.0 for invariance

**Test Plan:**
```yaml
# 10 new configs, 25K subset, 640px
base: lr0005 (best LR from Cycle 1)
variables:
  mosaic: [0.0, 0.3, 0.5, 0.7, 0.9]
  mixup: [0.0, 0.05, 0.1, 0.15]
  degrees: [0.0, 3.0, 5.0, 10.0]

# MANDATORY: Enable validation (see @POST-MORTEM.md)
val: true
save_period: 10
patience: 20
```

**Duration:** 3-4 hours (parallel on 2x GPU)

**Decision Point:**
- ✅ Clear winner (mAP > baseline + 5%) → PROCEED TO PHASE 2
- ❌ Still no signal → ESCALATE to Cycle 3

---

### Cycle 3: Architecture & Regularization (IF NEEDED)

**Trigger:** Cycles 1-2 show no hyperparameter helps

**⚠️ CRITICAL PRE-STEP (from @POST-MORTEM.md):**
```bash
# BEFORE starting ANY new fine-tuning experiments:
# 1. Validate ALL pretrained baselines (if not done already)
yolo val model=yolo26s-pose.pt data=ap3d device=0
yolo val model=yolo26m-pose.pt data=ap3d device=0
yolo val model=yolov8x-pose.pt data=ap3d device=0

# 2. Compare with previous cycle results

# 3. ONLY THEN start fine-tuning (if pretrained < potential improvement)
```

**Research Questions:**
1. Is yolo26n too small for 17kp pose?
2. Do we need stronger regularization (weight decay, dropout)?
3. Should we try different optimizers (AdamW vs SGD)?

**Hypotheses:**
- **H3.1:** yolo26s > yolo26n for pose capacity
- **H3.2:** weight_decay=0.001 > 0.0005 for regularization
- **H3.3:** AdamW optimizer > SGD for convergence

**Test Plan:**
```yaml
# 8 configs, 25K subset, 640px
variables:
  model: [yolo26n-pose, yolo26s-pose]
  weight_decay: [0.0001, 0.0005, 0.001]
  optimizer: [SGD, AdamW]

# MANDATORY: Enable validation (see @POST_MORTEM.md)
val: true
save_period: 10
patience: 20
```

**Duration:** 4-6 hours (yolo26s slower)

**Decision Point:**
- ✅ Clear winner → PROCEED TO PHASE 2
- ❌ Still stuck → CONSIDER DATASET ISSUE

---

### Cycle 4: Data & Dataset Issues (IF NEEDED)

**Trigger:** All hyperparameter/architecture experiments fail

**Root Cause Investigation:**
1. **Dataset quality:** Check for label errors, corruption
2. **Dataset size:** 25K too small? Try 50K subset
3. **Domain gap:** Skating + athletics mixed → try skating-only
4. **Baseline mismatch:** RTMO pre-trained too different?

**Actions:**
- Visualize predictions vs GT (find systematic errors)
- Compute per-joint PCK (which keypoints fail?)
- Train on 50K subset (is it data size?)
- Try skating-only subset (filter by camera 1+4)

**Duration:** 2-3 days (investigation + retraining)

**Decision Point:**
- ✅ Fix dataset issue → RETEST from Cycle 1
- ❌ Dataset OK but still failing → **CONSIDER ALTERNATIVE APPROACH**

---

## Phase 2: Full Training (ONLY after Phase 1 success)

### Preconditions

ALL of the following must be TRUE:

- [ ] Clear winner from Phase 1 (mAP > baseline + 5%)
- [ ] Winner reproducible on 2 different subsets
- [ ] No overfitting (train-val gap < 0.1)
- [ ] Documented in @RESEARCH_REPORT.md

### Phase 2A: Teacher Training (3-4 hours)

```yaml
# teacher_config.yaml
model: yolo26s-pose
data: ap3d_coco_mix.yaml  # 226K AP3D + 59K COCO = 285K frames
imgsz: 1280
epochs: 100
device: [0,1]  # DDP on 2x GPU

# Use BEST config from Phase 1
lr0: <from_phase_1>
freeze: <from_phase_1>
mosaic: <from_phase_1>
mixup: <from_phase_1>

# Enable validation this time!
val: true
patience: 15
save_period: 5
```

**Success criteria:**
- Val mAP > baseline (RTMO)
- No overfitting (train-val gap < 0.15)
- Stable convergence

**If teacher fails:** → GO BACK TO Phase 1 (new hypothesis)

### Phase 2B: Pseudo-Labeling SkatingVerse (24-48 hours)

```python
# Use TEACHER model to label SkatingVerse
teacher = YOLO("/root/yolo_runs/teacher/weights/best.pt")

# Generate pseudo-labels with HIGH confidence
for video in skatingverse_videos:
    poses = teacher.predict(video, conf=0.875)
    save_poses(poses, output_dir)

# Expected: ~200K-500K pseudo-labels
# Filter: min 7 keypoints/frame, confidence > 0.875
```

**Quality checks:**
- Manual inspection of 100 random samples
- Per-joint confidence distribution
- Temporal consistency (no wild jumps)

### Phase 2C: Student Training (12-18 hours)

```yaml
# student_config.yaml
model: yolo26m-pose  # Larger capacity
data: gt_plus_pseudo.yaml  # GT + pseudo mix
imgsz: 1280
epochs: 100
device: [0,1]

# Pseudo-label weight: 0.2 (1:4 GT:pseudo ratio)
pseudo_weight: 0.2
```

**Success criteria:**
- Val mAP > teacher by > 2%
- No catastrophic forgetting on COCO
- Real-time inference < 20ms/frame

---

## Decision Tree Summary

```
Phase 1 (HP Search)
│
├─ ✅ Clear winner (mAP > baseline + 5%)
│   └─> Phase 2A (Teacher full train)
│       │
│       ├─ ✅ Teacher succeeds
│       │   └─> Phase 2B (Pseudo-labeling)
│       │       └─> Phase 2C (Student)
│       │           └─> ✅ DEPLOYMENT READY
│       │
│       └─ ❌ Teacher fails
│           └─> BACK TO Phase 1 (new cycle)
│
└─ ❌ No clear winner (diff < 2%)
    └─> Cycle 2 (Augmentation deep dive)
        │
        ├─ ✅ Clear winner → Phase 2
        │
        └─ ❌ Still no signal → Cycle 3 (Architecture)
            │
            ├─ ✅ Clear winner → Phase 2
            │
            └─ ❌ Still stuck → Cycle 4 (Dataset investigation)
                │
                ├─ ✅ Fix found → Retest Phase 1
                │
                └─ ❌ Dataset OK → CONSIDER ALTERNATIVE APPROACH
```

---

## Timeline Estimates

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1 (HP Search) | 3-4h | 3-4h |
| **If Phase 1 fails:** |
| Cycle 2 (Augmentation) | 3-4h | 6-8h |
| Cycle 3 (Architecture) | 4-6h | 10-14h |
| Cycle 4 (Dataset) | 2-3d | 2.5-3d |
| **If Phase 1 succeeds:** |
| Phase 2A (Teacher) | 3-4h | 6-8h |
| Phase 2B (Pseudo-labeling) | 24-48h | 30-56h |
| Phase 2C (Student) | 12-18h | 42-74h |

**Best case:** 6-8 hours (Phase 1 succeeds → Teacher works)
**Worst case:** 3 days (multiple cycles + dataset investigation)

---

## Key Principles

1. **NO PREMATURE OPTIMIZATION** — Don't proceed to Phase 2 without clear evidence
2. **STATISTICAL SIGNIFICANCE** — 2% difference is noise, not signal
3. **DOCUMENT EVERYTHING** — All cycles logged in MD files
4. **FAIL FAST** — Each cycle is 3-6 hours, not days
5. **KNOW WHEN TO STOP** — After 3 failed cycles, reconsider approach

---

## References

- Research sources → @COMPREHENSIVE_RESEARCH_COMPARISON.md
- NLM research guidelines → @NLM_RESEARCH_GUIDELINES.md
- NotebookLM analysis → @NOTEBOOKLM_YOLO26_ANALYSIS.md

---

## Next Actions (Current)

**❌ PHASE 1 FAILED - DECISION MADE (2026-04-15 23:10 UTC)**

**Validation Results:**
- YOLO26s pretrained: 0.657 mAP50-95(P) ✅ **OPTIMAL**
- Best fine-tuned (f20): 0.517 (-21% vs pretrained) ❌
- batch64: 0.406 (-38% vs pretrained) ❌

**Decision:**
1. ❌ **DO NOT PROCEED TO PHASE 2** (pseudo-labeling)
2. ✅ **USE YOLO26s PRETRAINED DIRECTLY** as teacher model
3. 📝 Document findings in @POST-MORTEM.md ✅ DONE
4. 📝 Update @HP_SEARCH.md with results ✅ DONE

**Reasoning:**
- Fine-tuning on 25K subset causes catastrophic forgetting (-14% to -38%)
- Pretrained YOLO26s (0.657) outperforms all fine-tuned configs
- No clear winner from HP search - all configs worse than pretrained
- Training infrastructure issues (configs killed before completion)

**Alternative Approach:**
- Use YOLO26s pretrained directly as teacher for pseudo-labeling
- Skip fine-tuning step entirely
- Proceed to Phase 2B (pseudo-labeling SkatingVerse) with pretrained teacher
- Consider fine-tuning only if pretrained teacher proves insufficient

**Status:** Go/no-go decision made - **NO GO** for Phase 2 with fine-tuned teacher.
