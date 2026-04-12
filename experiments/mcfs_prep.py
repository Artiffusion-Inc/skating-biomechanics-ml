"""
MCFS-130 preprocessing: OP25→COCO17 mapping, normalization, element segment extraction.

Science Protocol:
  Goal: Assess MCFS data quality and FSC overlap (H7)
  Observe: Run preprocessing, collect statistics
  Measure: Segment counts, label distribution, overlap with FSC
  Analyze: H7 verdict

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    uv run python experiments/mcfs_prep.py
"""

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

MCFS = Path("data/datasets/mcfs")
MAPPING_FILE = MCFS / "mapping.txt"

# OP25 → COCO17 index mapping (13 keypoints)
OP25_TO_COCO17 = {
    0: 0,
    2: 5,
    3: 6,
    4: 7,
    5: 8,
    6: 9,
    7: 10,
    9: 11,
    10: 12,
    11: 13,
    12: 14,
    13: 15,
    14: 16,
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
        elif current_label is not None:
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
    print(f"\n{'=' * 60}")
    print("OBSERVATION: MCFS Data Quality")
    print(f"{'=' * 60}")
    print("Videos processed: 271")
    print(f"Total segments: {len(all_segments)}")
    print(f"Unique labels: {len(stats)}")
    lengths = [len(s[0]) for s in all_segments]
    print(
        f"Segment lengths: min={min(lengths)}, max={max(lengths)}, "
        f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}"
    )
    print(
        f"Segments per label: min={min(stats.values())}, max={max(stats.values())}, "
        f"mean={np.mean(list(stats.values())):.1f}"
    )

    print(f"\nVideos with NaN: {len(nan_stats)}/271")
    if nan_stats:
        total_nan = sum(v[0] for v in nan_stats.values())
        print(f"Total NaN values: {total_nan}")

    print("\nTop 20 labels:")
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
        # FSC uses numeric IDs, MCFS uses element names — no direct overlap.
        # Check if we can map via MMFS skeleton labels (FSC source).
        mmfs_labels_path = Path("data/datasets/mmfs/MMFS/skeleton/train_label.pkl")
        if mmfs_labels_path.exists():
            import pickle as _pkl

            mmfs_train_labels = _pkl.load(open(mmfs_labels_path, "rb"))
            mmfs_label_counts = Counter(mmfs_train_labels)
            print(f"\n{'=' * 60}")
            print("OBSERVATION: Label System Comparison")
            print(f"{'=' * 60}")
            print(
                f"FSC (via MMFS): {len(mmfs_label_counts)} numeric class IDs (0-{max(mmfs_label_counts)})"
            )
            print(f"MCFS: {len(stats)} named element classes")
            print("NOTE: FSC and MCFS use incompatible label schemes.")
            print("      Combined training requires manual name→ID mapping.")
        else:
            print("\nNOTE: MMFS skeleton labels not found.")

    # === ANALYZE ===
    print(f"\n{'=' * 60}")
    print("ANALYSIS: H7 Verdict")
    print(f"{'=' * 60}")
    h7_seg = len(all_segments) >= 500
    print("H7 predictions:")
    print(f"  >=500 segments: {'PASS' if h7_seg else 'FAIL'} ({len(all_segments)})")
    print(f"  H7 verdict: {'SUPPORTED' if h7_seg else 'REJECTED'}")

    # Save
    out_path = MCFS / "segments.pkl"
    pickle.dump(all_segments, open(out_path, "wb"))
    print(f"\nSaved {len(all_segments)} segments to {out_path}")


if __name__ == "__main__":
    main()
