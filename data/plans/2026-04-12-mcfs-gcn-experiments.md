# MCFS + GCN Experiments — Science Protocol

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether GCN architectures and MCFS frame-level data improve element classification beyond BiGRU 67.3% baseline.

**Method:** Science skill protocol — each experiment follows: **Goal → Observe → Hypothesize → Experiment → Measure → Analyze → Iterate**

**Tech Stack:** PyTorch (CUDA), numpy, pickle

---

## Background Knowledge (from previous experiments)

| Finding | Evidence |
|---------|----------|
| BiGRU + packed sequences = best model so far | 67.3% on 64 classes (Exp 3) |
| Class imbalance = #1 bottleneck | Top-10: 81.5%, all-64: 67.3% |
| Augmentation doesn't help | Mirror harmful (direction-dependent), noise/mix: +0.6pp noise |
| Start-crop >> center-crop | 70.4% vs 31.2% on top-10 |
| FSC > MMFS for classification | Shorter sequences, cleaner labels |
| Raw cosine sim = useless | Gap 0.09, within-class variance enormous |
| 80% achievable with enough data/class | Top-10 BiGRU: 81.5% |

---

## Hypotheses

| ID | Hypothesis | Prediction | How to Falsify |
|----|-----------|------------|-----------------|
| **H6** | ST-GCN exploits skeleton graph structure → better than BiGRU | GCN > BiGRU by ≥3pp on FSC 64-class | GCN ≤ BiGRU + 1pp |
| **H7** | MCFS frame-level labels enable clean element extraction | ≥500 segments, ≥50% overlap with FSC classes | <100 segments or <10% overlap |
| **H8** | Adding MCFS segments to FSC training improves accuracy | +3pp over FSC-only baseline | No improvement or regression |
| **H9** | GCN on MCFS 130 classes captures fine-grained distinctions | >60% on 130 classes | <40% (worse than BiGRU on 64) |

---

## File Structure

```
data/experiments/
├── README.md                  ← Master report (update after each experiment)
├── exp_2b_2c_2d.py           ← Existing: CNN + BiGRU + MMFS
├── exp_augmentation.py        ← Existing: augmentation ablation
├── mcfs_prep.py              ← NEW: Task 1
└── exp_gcn_mcfs.py            ← NEW: Tasks 2-4
```

---

### Task 1: Observe — MCFS Data Exploration (H7)

**Goal:** Understand MCFS data quality, segment distribution, and FSC overlap before running any models.

**Predictions:**
- MCFS will yield 500-2000 element segments after filtering NONE frames
- 30-60% of MCFS labels will overlap with FSC's 64 classes
- Segment length distribution: 30-300 frames (1-10 sec at 30fps)
- Some MCFS labels have <5 samples → too rare to use

- [ ] **Step 1: Write MCFS preprocessing script**

Create `data/experiments/mcfs_prep.py`:

```python
"""
MCFS-130 preprocessing: OP25→COCO17 mapping, normalization, element segment extraction.

Science Protocol:
  Goal: Assess MCFS data quality and FSC overlap
  Observe: Run preprocessing, collect statistics
  Measure: Segment counts, label distribution, overlap with FSC

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    uv run python data/experiments/mcfs_prep.py
"""
import json
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

MCFS = Path("data/datasets/mcfs")
MAPPING_FILE = MCFS / "mapping.txt"

# OP25 → COCO17 index mapping (13 keypoints)
# OP25: Nose(0), Neck(1), RShoulder(2), RElbow(3), RWrist(4),
#       LShoulder(5), LElbow(6), LWrist(7), MidHip(8),
#       RHip(9), RKnee(10), RAnkle(11), LHip(12), LKnee(13), LAnkle(14),
#       REye(15), LEye(16), REar(17), LEar(18),
#       LBigToe(19), LSmallToe(20), LHeel(21), RBigToe(22), RSmallToe(23), RHeel(24)
OP25_TO_COCO17 = {
    0: 0, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10,
    9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
}
OP25_MIDHIP_SRC = [9, 12]  # RHip, LHip in OP25


def load_mapping():
    mapping = {}
    with open(MAPPING_FILE) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]
    return mapping


def op25_to_coco17(pose_op25):
    """(F, 25, 3) OP25 → (F, 17, 2) COCO17. Reconstructs MidHip from LHip+RHip."""
    F = pose_op25.shape[0]
    midhip = pose_op25[:, OP25_MIDHIP_SRC, :2].mean(axis=1, keepdims=True)
    out = np.zeros((F, 17, 2), dtype=np.float32)
    for op_idx, coco_idx in OP25_TO_COCO17.items():
        out[:, coco_idx, :] = pose_op25[:, op_idx, :2]
    out[:, 11, :] = midhip.squeeze(1)
    return out


def normalize(p):
    """Root-center + scale normalize. p: (F, 17, 2)."""
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


def extract_segments(pose, labels, min_frames=30):
    """Extract contiguous non-NONE segments. Returns [(pose, label), ...]."""
    segments = []
    current_label = None
    start = None
    for i, label in enumerate(labels):
        if label != "NONE":
            if current_label is None:
                start = i
                current_label = label
            elif label != current_label:
                seg = pose[start:i]
                if len(seg) >= min_frames:
                    segments.append((normalize(seg), current_label))
                start = i
                current_label = label
        else:
            if current_label is not None:
                seg = pose[start:i]
                if len(seg) >= min_frames:
                    segments.append((normalize(seg), current_label))
                current_label = None
                start = None
    if current_label is not None:
        seg = pose[start:]
        if len(seg) >= min_frames:
            segments.append((normalize(seg), current_label))
    return segments


def main():
    mapping = load_mapping()
    print(f"MCFS labels: {len(mapping)}")

    feat_dir = MCFS / "features"
    gt_dir = MCFS / "groundTruth"
    all_segments = []
    stats = Counter()
    nan_stats = Counter()

    for fn in sorted(feat_dir.iterdir()):
        if fn.suffix != ".npy":
            continue
        pose_op25 = np.load(fn)
        gt_path = gt_dir / fn.with_suffix(".txt").name
        with open(gt_path) as f:
            labels = [line.strip() for line in f]

        # NaN check per video
        nan_count = int(np.isnan(pose_op25).sum())
        nan_kp = int(np.any(np.isnan(pose_op25), axis=(0, 2)).sum())
        if nan_count > 0:
            nan_stats[fn.stem] = (nan_count, nan_kp)

        pose_coco17 = op25_to_coco17(pose_op25)
        segments = extract_segments(pose_coco17, labels, min_frames=30)
        for seg_pose, seg_label in segments:
            all_segments.append((seg_pose, seg_label))
            stats[seg_label] += 1

    # === OBSERVE ===
    print(f"\n{'='*60}")
    print(f"OBSERVATION: MCFS Data Quality")
    print(f"{'='*60}")
    print(f"Videos processed: 271")
    print(f"Total segments: {len(all_segments)}")
    print(f"Unique labels: {len(stats)}")
    lengths = [len(s[0]) for s in all_segments]
    print(f"Segment lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")
    print(f"Segments per label: min={min(stats.values())}, max={max(stats.values())}, "
          f"mean={np.mean(list(stats.values())):.1f}")

    # NaN stats
    print(f"\nVideos with NaN: {len(nan_stats)}/271")
    if nan_stats:
        total_nan = sum(v[0] for v in nan_stats.values())
        print(f"Total NaN values: {total_nan}")

    # Label distribution
    print(f"\nTop 20 labels:")
    for label, count in stats.most_common(20):
        print(f"  {label:40s}: {count:4d}")
    print(f"\nLabels with <5 segments ({sum(1 for v in stats.values() if v < 5)}):")
    for label, count in sorted(stats.items(), key=lambda x: x[1]):
        if count < 5:
            print(f"  {label}: {count}")

    # FSC overlap
    fsc_path = Path("data/datasets/figure-skating-classification/label_mapping.json")
    if fsc_path.exists():
        with open(fsc_path) as f:
            fsc_map = json.load(f)
        fsc_names = set(fsc_map.values())
        mcfs_names = set(stats.keys())
        overlap = fsc_names & mcfs_names
        print(f"\n{'='*60}")
        print(f"OBSERVATION: FSC/MCFS Overlap")
        print(f"{'='*60}")
        print(f"FSC classes: {len(fsc_names)}")
        print(f"MCFS labels: {len(mcfs_names)}")
        print(f"Overlap: {len(overlap)} / {len(fsc_names)} FSC classes")
        print(f"Shared labels: {sorted(overlap)}")
        shared_count = sum(stats[l] for l in overlap)
        print(f"Segments with shared labels: {shared_count} / {len(all_segments)} "
              f"({shared_count/len(all_segments):.1%})")

    # === ANALYZE ===
    print(f"\n{'='*60}")
    print(f"ANALYSIS: H7 Verdict")
    print(f"{'='*60}")
    h7_pass = len(all_segments) >= 500 and len(overlap) >= len(fsc_names) * 0.3
    print(f"H7 predictions:")
    print(f"  ≥500 segments: {'PASS' if len(all_segments) >= 500 else 'FAIL'} ({len(all_segments)})")
    print(f"  ≥30% FSC overlap: {'PASS' if len(overlap) >= len(fsc_names)*0.3 else 'FAIL'} "
          f"({len(overlap)}/{len(fsc_names)} = {len(overlap)/max(len(fsc_names),1):.0%})")
    print(f"  H7 verdict: {'SUPPORTED' if h7_pass else 'REJECTED'}")

    # Save
    out_path = MCFS / "segments.pkl"
    pickle.dump(all_segments, open(out_path, "wb"))
    print(f"\nSaved {len(all_segments)} segments to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and observe**

Run: `uv run python data/experiments/mcfs_prep.py`
Collect: segment counts, label distribution, FSC overlap, NaN statistics.

- [ ] **Step 3: Analyze H7**

Based on observations, determine if MCFS data is usable. If H7 REJECTED (too few segments, no overlap), stop and report. If SUPPORTED, proceed to Task 2.

- [ ] **Step 4: Commit**

```bash
git add data/experiments/mcfs_prep.py
git commit -m "feat(experiments): add MCFS preprocessing and segment extraction"
```

---

### Task 2: Experiment — GCN vs BiGRU on FSC (H6)

**Goal:** Fair comparison of ST-GCN vs BiGRU on identical FSC 64-class data.

**Hypothesis:** ST-GCN exploits skeleton graph structure → better than BiGRU.
**Prediction:** GCN ≥ BiGRU + 3pp (≥70.3%).
**Falsification:** GCN ≤ BiGRU + 1pp (≤68.3%).

**Controls:**
- Same train/val/test split as Exp 3 baseline (3765/394/1007, stratified)
- Same training config: AdamW lr=1e-3, CosineAnnealing, label smoothing 0.1, grad clip 1.0
- Same early stopping: patience=10 on validation loss
- Same random seed: 42
- BiGRU re-run for exact comparison (same script, same conditions)

**ST-GCN Architecture:**
```
Input: (B, T, 17, 2) → (B, C=2, T, V=17)
ST-Block 1: SpatialGraphConv(2→64) + BN + ReLU, TemporalConv(k=9,pad=4) + BN + ReLU, Residual
ST-Block 2: SpatialGraphConv(64→128) + BN + ReLU, TemporalConv(k=9,pad=4) + BN + ReLU, Residual
ST-Block 3: SpatialGraphConv(128→256) + BN + ReLU, TemporalConv(k=9,pad=4) + BN + ReLU, Residual
Global AvgPool over T → FC(256→128→Dropout→num_classes)
```

**SpatialGraphConv:** Multiply input by normalized adjacency matrix A (17×17). A derived from COCO 17kp skeleton edges: `A[i,j] = 1 if (i,j) is an edge or i==j`, then row-normalize.

**Variable-length handling:** Pad to batch max, use attention mask for GlobalAvgPool (mask padded frames).

- [ ] **Step 1: Write experiment script**

Create `data/experiments/exp_gcn_mcfs.py` with:
- `STGCN` model class (spatial graph conv + temporal conv blocks)
- `build_adjacency()` function from COCO 17kp edges
- `train_eval()` identical to BiGRU baseline (AdamW, CosineAnnealing, early stopping)
- Exp 4a: load FSC, train BiGRU (control) + ST-GCN (treatment), compare

- [ ] **Step 2: Run experiment**

Run: `uv run python data/experiments/exp_gcn_mcfs.py`
Measure: test accuracy, epochs to convergence, train-val gap, wall time.

- [ ] **Step 3: Analyze H6**

Compare GCN vs BiGRU. If GCN ≤ BiGRU + 1pp: H6 REJECTED (graph structure doesn't help). If GCN ≥ BiGRU + 3pp: H6 SUPPORTED. Record which model overfits more.

- [ ] **Step 4: Update README**

Add Experiment 4 section to `data/experiments/README.md` with full training curves and analysis.

- [ ] **Step 5: Commit**

```bash
git add data/experiments/exp_gcn_mcfs.py data/experiments/README.md
git commit -m "feat(experiments): GCN vs BiGRU comparison on FSC 64-class"
```

---

### Task 3: Experiment — MCFS + FSC Combined Training (H8)

**Goal:** Test if MCFS segments improve FSC classification when added as extra training data.

**Hypothesis:** MCFS provides additional real data for shared classes → reduces class imbalance → improves accuracy.
**Prediction:** FSC+MCFS ≥ FSC-only + 3pp (≥70.3%).
**Falsification:** FSC+MCFS ≤ FSC-only + 1pp or regression.

**Method:**
1. Load MCFS segments from `segments.pkl` (Task 1 output)
2. Filter to labels that overlap with FSC's 64 classes
3. Map MCFS label names → FSC label indices
4. Add MCFS segments to FSC training set (no validation/test contamination)
5. Train best model from Task 2 (BiGRU or GCN, whichever won) on combined data
6. Evaluate on FSC test set only (no MCFS data in test)

**Controls:**
- Same model architecture as Task 2 winner
- Same training config
- Test set = FSC test only (apples-to-apples with baseline)
- Only MCFS segments with matching FSC labels are used

- [ ] **Step 1: Add combined training to experiment script**

Add to `data/experiments/exp_gcn_mcfs.py`:
- Load `segments.pkl`, filter by FSC label overlap
- Concatenate with FSC training data
- Train and evaluate

- [ ] **Step 2: Run experiment**

Run: `uv run python data/experiments/exp_gcn_mcfs.py` (or add flag `--experiment 5b`)
Measure: test accuracy vs FSC-only baseline.

- [ ] **Step 3: Analyze H8**

If FSC+MCFS > FSC-only by ≥3pp: H8 SUPPORTED. If no improvement: H8 REJECTED. Check per-class accuracy changes — did rare classes improve?

- [ ] **Step 4: Update README and commit**

---

### Task 4: Experiment — GCN on MCFS 130 Classes (H9)

**Goal:** Test if MCFS's 130 fine-grained classes with GCN architecture capture distinctions that FSC's 64 classes miss.

**Hypothesis:** More classes + graph structure = richer feature learning → better overall accuracy.
**Prediction:** >60% on MCFS 130 classes (vs BiGRU 67.3% on FSC 64 classes — different task but indicative).
**Falsification:** <40% on MCFS 130 classes.

**Method:**
1. Load all MCFS segments from `segments.pkl`
2. Filter to labels with ≥5 samples (remove ultra-rare classes)
3. Train BiGRU and GCN on MCFS data with stratified train/val/test split
4. Compare BiGRU vs GCN on MCFS directly

**Controls:**
- Same training config as FSC experiments
- Minimum 5 samples per class (from observations in Task 1)
- 80/10/10 stratified split

- [ ] **Step 1: Add MCFS-only experiments to script**

- [ ] **Step 2: Run experiment**

- [ ] **Step 3: Analyze H9**

- [ ] **Step 4: Final README update and commit**

---

## Iteration Criteria

After all 4 tasks:

| Outcome | Next Step |
|---------|-----------|
| H6 SUPPORTED (GCN > BiGRU) | Use GCN as backbone for all future experiments |
| H6 REJECTED (GCN ≤ BiGRU) | Stick with BiGRU, GCN graph structure doesn't help on this data |
| H7 REJECTED (MCFS unusable) | Focus on getting more FSC-like data, skip MCFS |
| H8 SUPPORTED (combined helps) | Merge MCFS into permanent training pipeline |
| H9 SUPPORTED (MCFS 130 > 60%) | Use MCFS 130-class as primary classification target |
| All rejected | Pivot to VIFSS encoder (pre-trained features may be the only path) |

---

## Verification

1. `data/datasets/mcfs/segments.pkl` exists with extracted element segments
2. `data/experiments/exp_gcn_mcfs.py` runs end-to-end on GPU
3. `data/experiments/README.md` updated with all new experiment results
4. All hypotheses (H6-H9) have verdicts with evidence
5. Master results table updated
