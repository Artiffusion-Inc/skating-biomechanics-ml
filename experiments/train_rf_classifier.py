"""Experiment: RF segment classifier on MCFS top-30 labels.

Hypothesis: RF on 5 biomechanical features achieves >0.60 top-1 accuracy on MCFS fine labels
Status: PENDING

Usage:
    uv run python experiments/train_rf_classifier.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from ml.src.tas.classifier import SegmentClassifier, extract_segment_features
from ml.src.tas.dataset import MCFSCoarseDataset

BASE = Path("data/datasets/mcfs")
CHECKPOINT_DIR = Path("experiments/checkpoints/rf_classifier")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ds = MCFSCoarseDataset(BASE / "features", BASE / "groundTruth")

    # Collect all segments with features
    all_segments = []
    label_counts: Counter[str] = Counter()

    for i in range(len(ds)):
        poses, labels, length = ds[i]
        # Extract contiguous segments (skip None=0)
        start = None
        current_label = None
        fine_labels = ds.get_fine_labels(i)
        for t in range(length):
            fine_label = fine_labels[t]
            if labels[t] != 0:
                if start is None:
                    start = t
                    current_label = fine_label
                elif fine_label != current_label:
                    # End previous segment
                    seg_poses = poses[start:t]
                    features = extract_segment_features(seg_poses)
                    all_segments.append({"features": features, "label": current_label})
                    label_counts[current_label] += 1
                    start = t
                    current_label = fine_label
        # Handle last segment
        if start is not None and current_label is not None:
            seg_poses = poses[start:length]
            features = extract_segment_features(seg_poses)
            all_segments.append({"features": features, "label": current_label})
            label_counts[current_label] += 1

    # Filter to top 30 labels
    top30 = [label for label, _ in label_counts.most_common(30)]
    filtered = [s for s in all_segments if s["label"] in top30]
    print(f"Total segments: {len(all_segments)}, Top-30 segments: {len(filtered)}")

    if len(filtered) == 0:
        print("No segments found. Exiting.")
        return

    # Train/test split
    train, test = train_test_split(
        filtered, test_size=0.2, random_state=42, stratify=[s["label"] for s in filtered]
    )

    clf = SegmentClassifier(n_estimators=200)
    clf.fit(train)

    # Evaluate
    correct = 0
    for seg in test:
        pred, _conf = clf.predict(seg["features"])
        if pred == seg["label"]:
            correct += 1
    acc = correct / len(test)
    print(f"Top-30 accuracy: {acc:.4f}")

    # Save
    import joblib
    joblib.dump(clf, CHECKPOINT_DIR / "rf_top30.joblib")
    with open(CHECKPOINT_DIR / "results.json", "w") as f:
        json.dump({"accuracy": acc, "num_classes": 30, "num_train": len(train), "num_test": len(test)}, f)


if __name__ == "__main__":
    main()
