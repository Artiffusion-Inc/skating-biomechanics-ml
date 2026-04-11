# Experiment Log: VIFSS Decision

**Date:** 2026-04-11
**Goal:** Determine whether VIFSS contrastive pre-training is needed, or if simpler approaches suffice.
**Success criteria:** Element classification accuracy >= 0.80 on test split.

---

## Hypotheses

| ID | Hypothesis | Status | Verdict |
|----|-----------|--------|---------|
| H0 | VIFSS contrastive pre-training is necessary (accuracy < 0.70 without it) | Testing | — |
| H1 | Raw normalized 2D poses + simple classifier gives >= 0.80 accuracy | REJECTED | 21.8% accuracy |
| H2 | Normalization alone provides view-invariance (cosine sim > 0.85 same-pose-different-angle) | Testing | — |
| H3 | VIFSS contrastive pre-training helps marginally (+3-5%) | Pending | — |

---

## Experiment 1: Normalization + Cosine Similarity Sanity Check

**Date:** 2026-04-11
**Dataset:** Figure-Skating-Classification (4161 train, 1007 test, 64 classes)
**Method:** Root-center + scale normalization, frame-wise cosine similarity, 500 within-class + 500 between-class pairs.

### Results

```
Within-class:   0.557 ± 0.239  [−0.401, 0.989]
Between-class: 0.466 ± 0.258  [−0.465, 0.869]
Gap (within − between): +0.090
```

### Threshold Analysis

| Threshold | Within-class >= t | Between-class >= t |
|----------|-------------------|---------------------|
| 0.0 | 96.4% | 92.0% |
| 0.2 | 91.0% | 86.2% |
| 0.4 | 77.0% | 67.8% |
| 0.6 | 58.2% | 37.8% |
| 0.8 | 5.0% | 2.4% |

### Conclusions

1. **Raw cosine similarity does NOT separate classes.** Gap of 0.09 is noise-level.
2. **Within-class variance is enormous** (range 1.39) — same-class pairs can be highly dissimilar.
3. **Negative similarities are common** — indicates normalization artifacts (different body proportions, execution speeds).
4. **Simple embedding (flatten + cosine) is insufficient** for cross-athlete comparison.
5. **A learned embedding is required** to extract discriminative, invariant features.

### Implications for VIFSS

- Variant A (technique_similarity as MetricResult) **cannot work** with raw pose similarity.
- VIFSS's learned encoder is likely necessary — the contrastive pre-training teaches the model to be invariant to view angle, body proportions, and execution speed.
- The key question shifts from "do we need VIFSS?" to "how much does VIFSS improve over a simple supervised baseline?"

---

## Experiment 2: Baseline 1D-CNN Element Classifier

**Date:** 2026-04-11
**Dataset:** Figure-Skating-Classification (4161 train, 1007 test, 64 classes)
**Model:** 3-layer 1D-CNN (Conv1d 34→64→128→128, AdaptiveAvgPool, FC 128→64→64)
**Training:** Adam lr=1e-3, CrossEntropyLoss, 30 epochs, CPU, batch=64
**Input:** Normalized 2D poses (COCO 17kp x,y), truncated/padded to 150 frames.

### Results

```
Params: 114,368
Epoch  1:  loss=3.640  acc=0.197  best=0.197
Epoch  5:  loss=2.482  acc=0.050  best=0.197
Epoch 10: loss=2.272  acc=0.052  best=0.197
Epoch 15: loss=2.105  acc=0.059  best=0.197
Epoch 20: loss=1.868  acc=0.061  best=0.197
Epoch 25: loss=1.702  acc=0.208  best=0.219
Epoch 30: loss=1.579  acc=0.099  best=0.219

Best accuracy: 21.8% (random baseline = 1/64 = 1.6%)
```

### Training Dynamics

- Loss decreases steadily (3.64 → 1.58) — model IS learning.
- Accuracy oscillates and doesn't converge — training instability.
- Best accuracy achieved early (epoch 1-2), then overfitting.

### Identified Problems

1. **150 frames truncation destroys content** — sequence lengths range 150-606 frames. Many elements occur in the latter half of the sequence.
2. **Padding dilutes signal** — zero-padded frames add noise.
3. **64 classes with ~4K samples** — ~65 samples per class (highly imbalanced, min=8, max=406).
4. **COCO 17kp lacks foot keypoints** — no blade edge information for jump discrimination.
5. **No temporal alignment** — same-class sequences may start at different phases of the element.

### Conclusions

1. **H1 REJECTED:** Simple supervised learning on truncated/padded sequences gives 21.8% — far from 0.80 target.
2. **Variable-length sequences are a critical issue** — need proper handling (attention, segment-level features).
3. **Class imbalance hurts** — 64 classes with min=8 samples is too sparse.
4. **Loss decreasing + accuracy not improving = overfitting** — model memorizes training set.
5. **Need either: (a) temporal model that handles variable length, (b) fewer classes, (c) VIFSS pre-trained features, or (d) all of the above.**

---

## Open Questions

1. **Does VIFSS pre-training + fine-tuning fix the overfitting?** Pre-trained features may generalize better.
2. **Is MMFS a better dataset for this?** 63 classes (vs 64), 3959 train samples, quality scores available.
3. **Would center-crop (middle 150 frames instead of first 150) help?** Elements may be centered in sequences.
4. **Would reducing to top-10 classes improve accuracy?** More data per class, cleaner signal.
5. **Does BiGRU handle variable length better than 1D-CNN + padding?**

## Next Experiments (Priority Order)

| ID | Experiment | Why | Est. Time |
|----|-----------|-----|----------|
| 2b | Center-crop + top-10 classes | Isolate truncation and class imbalance effects | 10 min |
| 2c | BiGRU with variable-length sequences | Proper temporal modeling | 20 min |
| 2d | MMFS dataset (63 classes, quality scores) | Better data source comparison | 20 min |
| 3 | Multi-camera view invariance (AthletePose3D) | Test H2 directly | 30 min |
