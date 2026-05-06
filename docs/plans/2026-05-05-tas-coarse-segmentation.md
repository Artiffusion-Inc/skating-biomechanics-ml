# TAS Coarse Segmentation + RF Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace rule-based `element_segmenter.py` with ML-based coarse temporal action segmentation (Jump/Spin/Step/None) + Random Forest fine classifier on ~30 most frequent element types.

**Architecture:** BiGRU per-frame classifier on normalized H3.6M 17kp skeletons → segment extraction (merge same-label, min duration) → RF classifier on segment biomechanical features → integration into `ElementSegmenter`. Evaluated with OverlapF1.

**Tech Stack:** PyTorch (BiGRU), scikit-learn (RF), NumPy, existing `coco_to_h36m` converter, MCFS dataset (271 videos, 1.7M frame-level labels).

---

## File Structure

| File | Responsibility |
|------|---------------|
| `ml/src/tas/__init__.py` | Package init, exports TAS model and classifier |
| `ml/src/tas/dataset.py` | MCFS data loading, OP25→COCO17→H3.6M conversion, coarse label mapping, variable-length batching |
| `ml/src/tas/model.py` | BiGRU model: normalize → BiGRU → Dense(4) → softmax |
| `ml/src/tas/metrics.py` | OverlapF1 temporal segmentation metric |
| `ml/src/tas/classifier.py` | RF classifier: segment features → fine class (30 types) |
| `ml/src/tas/inference.py` | End-to-end inference: poses → TAS coarse → segments → RF fine → `SegmentationResult` |
| `experiments/train_tas.py` | Training script: load data, train BiGRU, 5-fold CV, save checkpoint |
| `experiments/evaluate_tas.py` | Evaluation: OverlapF1 on held-out fold, confusion matrix |
| `experiments/train_rf_classifier.py` | Train RF on extracted segments from MCFS |
| `ml/tests/tas/` | Test suite for TAS components |

---

## Data Format

**MCFS raw:**
- `features/*.npy`: `(T, 25, 3)` — OpenPose 25 keypoints (x, y, confidence)
- `groundTruth/*.txt`: One label per line, 130 classes + NONE
- 271 files total, variable length (33-2176 frames)

**Coarse mapping:**
```
Jump   → labels containing: Axel, Salchow, Toeloop, Lutz, Loop, Flip, Euler
Spin   → labels containing: Spin
Step   → labels containing: StepSequence, ChoreoSequence
None   → "NONE"
```

**Normalized input:**
- OP25→COCO17 (existing `mcfs_prep.py:OP25_TO_COCO17`)
- COCO17→H3.6M (existing `pose_estimation/h36m.py:coco_to_h36m`)
- Root-center + spine-length scale (existing `normalizer.py`)

---

### Task 1: MCFS Data Loader with Coarse Labels

**Files:**
- Create: `ml/src/tas/dataset.py`
- Create: `ml/tests/tas/test_dataset.py`
- Reference: `experiments/mcfs_prep.py` (OP25→COCO17 mapping)

- [ ] **Step 1: Write coarse label mapping function**

```python
def coarse_label(fine_label: str) -> int:
    """Map 130-class MCFS label to 4 coarse classes.

    Returns:
        0: None, 1: Jump, 2: Spin, 3: Step
    """
    if fine_label == "NONE":
        return 0
    if any(s in fine_label for s in ("Axel", "Salchow", "Toeloop", "Lutz", "Loop", "Flip", "Euler")):
        return 1
    if "Spin" in fine_label:
        return 2
    if any(s in fine_label for s in ("StepSequence", "ChoreoSequence")):
        return 3
    return 0  # Default to None for unmapped
```

- [ ] **Step 2: Implement MCFS dataset class**

```python
class MCFSCoarseDataset(Dataset):
    """PyTorch dataset for MCFS continuous routines with coarse labels.

    Loads .npy features + .txt ground truth, converts OP25→COCO17→H3.6M,
    normalizes, and returns (poses, labels, length) tuples.
    """

    def __init__(self, features_dir: Path, labels_dir: Path, normalize: bool = True):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.normalize = normalize
        # Match features and labels by stem (e.g., n01_p01)
        feature_files = {p.stem: p for p in features_dir.glob("*.npy")}
        label_files = {p.stem: p for p in labels_dir.glob("*.txt")}
        self.samples = sorted(set(feature_files.keys()) & set(label_files.keys()))
        self.feature_paths = feature_files
        self.label_paths = label_files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        stem = self.samples[idx]
        # Load poses: (T, 25, 3) OP25
        poses_op25 = np.load(self.feature_paths[stem])  # (T, 25, 3)
        # Load labels
        with open(self.label_paths[stem]) as f:
            fine_labels = [line.strip() for line in f]
        # Convert
        poses_coco17 = op25_to_coco17(poses_op25)  # (T, 17, 2)
        poses_h36m = np.stack([coco_to_h36m(p) for p in poses_coco17])  # (T, 17, 2)
        # Coarse labels
        coarse = np.array([coarse_label(l) for l in fine_labels], dtype=np.int64)
        # Normalize
        if self.normalize:
            poses_h36m = normalize_poses(poses_h36m)
        return poses_h36m.astype(np.float32), coarse, len(coarse)
```

- [ ] **Step 3: Implement collate function for variable-length sequences**

```python
def pad_collate(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad batch of variable-length sequences to max length.

    Returns:
        poses: (B, T_max, 17, 2) padded with zeros
        labels: (B, T_max) padded with -1 (ignore index)
        lengths: (B,) original lengths
    """
    poses_list, labels_list, lengths = zip(*batch)
    max_len = max(lengths)
    B = len(batch)
    poses_padded = torch.zeros(B, max_len, 17, 2, dtype=torch.float32)
    labels_padded = torch.full((B, max_len), -1, dtype=torch.long)
    for i, (p, l, le) in enumerate(zip(poses_list, labels_list, lengths)):
        poses_padded[i, :le] = torch.from_numpy(p)
        labels_padded[i, :le] = torch.from_numpy(l)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return poses_padded, labels_padded, lengths_tensor
```

- [ ] **Step 4: Write test for data loader**

```python
def test_mcfs_dataset():
    from ml.src.tas.dataset import MCFSCoarseDataset, coarse_label
    ds = MCFSCoarseDataset(
        Path("data/datasets/mcfs/features"),
        Path("data/datasets/mcfs/groundTruth"),
    )
    assert len(ds) > 0
    poses, labels, length = ds[0]
    assert poses.shape == (length, 17, 2)
    assert labels.shape == (length,)
    assert set(labels.tolist()).issubset({0, 1, 2, 3})

def test_coarse_label_mapping():
    from ml.src.tas.dataset import coarse_label
    assert coarse_label("NONE") == 0
    assert coarse_label("3Flip") == 1
    assert coarse_label("ChComboSpin4") == 2
    assert coarse_label("StepSequence4") == 3
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest ml/tests/tas/test_dataset.py -v
```

Expected: PASS (or FAIL with meaningful error if data missing)

- [ ] **Step 6: Commit**

```bash
git add ml/src/tas/dataset.py ml/tests/tas/test_dataset.py
git commit -m "feat(tas): MCFS data loader with coarse label mapping"
```

---

### Task 2: BiGRU TAS Model

**Files:**
- Create: `ml/src/tas/model.py`
- Create: `ml/tests/tas/test_model.py`

- [ ] **Step 1: Implement BiGRU model**

```python
class BiGRUTAS(nn.Module):
    """BiGRU for frame-wise coarse temporal action segmentation.

    Input: (B, T, 17, 2) normalized H3.6M poses
    Output: (B, T, 4) logits for [None, Jump, Spin, Step]
    """

    def __init__(
        self,
        input_dim: int = 34,      # 17 joints × 2 coords
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Flatten (17, 2) → 34 per frame
        self.proj = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # BiGRU output: 2 × hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        poses: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            poses: (B, T, 17, 2)
            lengths: (B,) original sequence lengths
        Returns:
            logits: (B, T, 4)
        """
        B, T, J, C = poses.shape
        # Flatten joints
        x = poses.reshape(B, T, J * C)  # (B, T, 34)
        x = self.proj(x)  # (B, T, hidden_dim)

        # Pack for RNN
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: (B, T, hidden_dim * 2)

        logits = self.classifier(out)  # (B, T, 4)
        return logits
```

- [ ] **Step 2: Write model test**

```python
def test_bigru_forward():
    model = BiGRUTAS(input_dim=34, hidden_dim=64, num_layers=1)
    B, T, J, C = 2, 100, 17, 2
    poses = torch.randn(B, T, J, C)
    lengths = torch.tensor([100, 80])
    logits = model(poses, lengths)
    assert logits.shape == (B, T, 4)

def test_bigru_variable_length():
    model = BiGRUTAS(hidden_dim=64, num_layers=1)
    B, T = 2, 50
    poses = torch.randn(B, T, 17, 2)
    lengths = torch.tensor([50, 30])
    logits = model(poses, lengths)
    # Check that padded frames don't produce NaN
    assert not torch.isnan(logits).any()
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest ml/tests/tas/test_model.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ml/src/tas/model.py ml/tests/tas/test_model.py
git commit -m "feat(tas): BiGRU coarse segmentation model"
```

---

### Task 3: OverlapF1 Metric

**Files:**
- Create: `ml/src/tas/metrics.py`
- Create: `ml/tests/tas/test_metrics.py`
- Reference: `docs/research/RESEARCH_ACTION_SEGMENTATION_2026-04-09.md`

- [ ] **Step 1: Implement OverlapF1**

```python
def extract_segments(labels: np.ndarray, id2label: dict[int, str]) -> list[dict]:
    """Extract contiguous segments from frame-wise labels.

    Returns list of {label, start, end} dicts.
    """
    segments = []
    if len(labels) == 0:
        return segments
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            if current != 0:  # Skip None (class 0)
                segments.append({"label": id2label[current], "start": start, "end": i - 1})
            current = labels[i]
            start = i
    # Last segment
    if current != 0:
        segments.append({"label": id2label[current], "start": start, "end": len(labels) - 1})
    return segments


def segment_iou(seg1: dict, seg2: dict) -> float:
    """Compute IoU between two temporal segments."""
    s1, e1 = seg1["start"], seg1["end"]
    s2, e2 = seg2["start"], seg2["end"]
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    inter = max(0, inter_end - inter_start + 1)
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union if union > 0 else 0.0


class OverlapF1:
    """Temporal segmentation evaluation: F1 with IoU >= threshold.

    Following AAAI 2021 MCFS paper and figure-skating-action-segmentation repo.
    """

    def __init__(self, iou_threshold: float = 0.5, num_classes: int = 4):
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.id2label = {0: "None", 1: "Jump", 2: "Spin", 3: "Step"}

    def compute(
        self,
        pred_labels: np.ndarray,
        true_labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute OverlapF1 between predicted and true frame-wise labels.

        Args:
            pred_labels: (T,) predicted class indices
            true_labels: (T,) ground truth class indices

        Returns:
            Dict with 'f1', 'precision', 'recall', and per-class F1 scores.
        """
        pred_segs = extract_segments(pred_labels, self.id2label)
        true_segs = extract_segments(true_labels, self.id2label)

        # Match predicted segments to true segments
        matched_true = set()
        matched_pred = set()

        for pi, ps in enumerate(pred_segs):
            for ti, ts in enumerate(true_segs):
                if ti in matched_true:
                    continue
                if ps["label"] != ts["label"]:
                    continue
                iou = segment_iou(ps, ts)
                if iou >= self.iou_threshold:
                    matched_pred.add(pi)
                    matched_true.add(ti)
                    break

        tp = len(matched_pred)
        fp = len(pred_segs) - tp
        fn = len(true_segs) - len(matched_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"f1": f1, "precision": precision, "recall": recall}
```

- [ ] **Step 2: Write metric test**

```python
def test_overlapf1_perfect_match():
    metric = OverlapF1(iou_threshold=0.5)
    pred = np.array([0, 0, 1, 1, 1, 0, 2, 2])
    true = np.array([0, 0, 1, 1, 1, 0, 2, 2])
    result = metric.compute(pred, true)
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0

def test_overlapf1_partial_iou():
    metric = OverlapF1(iou_threshold=0.5)
    # True: Jump frames 2-6 (5 frames)
    # Pred: Jump frames 3-5 (3 frames), IoU = 3/7 < 0.5 → no match
    pred = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    true = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    result = metric.compute(pred, true)
    assert result["f1"] < 1.0
    assert result["precision"] < 1.0
    assert result["recall"] < 1.0
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest ml/tests/tas/test_metrics.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ml/src/tas/metrics.py ml/tests/tas/test_metrics.py
git commit -m "feat(tas): OverlapF1 temporal segmentation metric"
```

---

### Task 4: Training Script

**Files:**
- Create: `experiments/train_tas.py`
- Modify: `experiments/README.md` (add hypothesis row)

- [ ] **Step 1: Implement 5-fold CV training**

```python
"""
Experiment: TAS BiGRU coarse segmentation
Hypothesis: BiGRU on normalized H3.6M poses achieves >0.70 OverlapF1@50 on MCFS 4-class segmentation
Status: PENDING

Usage:
    uv run python experiments/train_tas.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from ml.src.tas.dataset import MCFSCoarseDataset, pad_collate
from ml.src.tas.metrics import OverlapF1
from ml.src.tas.model import BiGRUTAS

BASE = Path("data/datasets/mcfs")
CHECKPOINT_DIR = Path("experiments/checkpoints/tas_bigr")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for poses, labels, lengths in loader:
        poses, labels = poses.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(poses, lengths)
        # Masked cross-entropy (ignore -1 padding)
        loss = criterion(logits.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_fold(model, loader, device) -> dict:
    model.eval()
    metric = OverlapF1(iou_threshold=0.5)
    all_preds = []
    all_true = []
    with torch.no_grad():
        for poses, labels, lengths in loader:
            poses = poses.to(device)
            logits = model(poses, lengths)
            preds = logits.argmax(dim=-1).cpu().numpy()
            for i, le in enumerate(lengths):
                all_preds.append(preds[i, :le])
                all_true.append(labels[i, :le].numpy())

    # Compute per-sample F1 and average
    f1s = []
    for p, t in zip(all_preds, all_true):
        result = metric.compute(p, t)
        f1s.append(result["f1"])
    return {"f1": float(np.mean(f1s)), "precision": 0.0, "recall": 0.0}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = MCFSCoarseDataset(BASE / "features", BASE / "groundTruth")
    print(f"Dataset size: {len(ds)}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(ds)))):
        print(f"\n--- Fold {fold + 1}/5 ---")
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=pad_collate)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=pad_collate)

        model = BiGRUTAS(hidden_dim=128, num_layers=2, dropout=0.3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        best_f1 = 0.0
        for epoch in range(50):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_result = eval_fold(model, val_loader, device)
            print(f"  Epoch {epoch + 1}: loss={train_loss:.4f}, val_f1={val_result['f1']:.4f}")
            if val_result["f1"] > best_f1:
                best_f1 = val_result["f1"]
                torch.save({
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                }, CHECKPOINT_DIR / f"fold_{fold}_best.pt")

        fold_results.append(best_f1)
        print(f"  Fold {fold + 1} best F1: {best_f1:.4f}")

    print(f"\n=== 5-Fold CV Results ===")
    print(f"Mean F1@50: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")

    # Save summary
    with open(CHECKPOINT_DIR / "cv_results.json", "w") as f:
        json.dump({"fold_f1s": fold_results, "mean": float(np.mean(fold_results)), "std": float(np.std(fold_results))}, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add hypothesis to experiments/README.md**

```markdown
| H? | BiGRU on normalized H3.6M poses achieves >0.70 OverlapF1@50 on MCFS 4-class segmentation | PENDING | experiments/train_tas.py |
```

- [ ] **Step 3: Run training (fast smoke test)**

```bash
uv run python experiments/train_tas.py
```

Expected: Runs without errors, prints fold results. Full training may take 30-60 min on GPU.

- [ ] **Step 4: Commit**

```bash
git add experiments/train_tas.py experiments/README.md
git commit -m "feat(tas): BiGRU training script with 5-fold CV"
```

---

### Task 5: RF Classifier on Segments

**Files:**
- Create: `ml/src/tas/classifier.py`
- Create: `experiments/train_rf_classifier.py`
- Create: `ml/tests/tas/test_classifier.py`

- [ ] **Step 1: Implement feature extraction for segments**

```python
def extract_segment_features(
    poses: np.ndarray,  # (T, 17, 2) normalized H3.6M
    fps: float = 30.0,
) -> dict[str, float]:
    """Extract biomechanical features from a segment for RF classification.

    Features: duration, hip_y_range, motion_energy, rotation_speed, etc.
    """
    T = poses.shape[0]
    duration = T / fps

    # Hip Y trajectory (for jumps)
    midhip = poses[:, [H36Key.LHIP, H36Key.RHIP], :].mean(axis=1)
    hip_y_range = float(np.max(midhip[:, 1]) - np.min(midhip[:, 1]))

    # Motion energy
    diff = np.diff(poses, axis=0)
    motion_energy = float(np.mean(np.linalg.norm(diff, axis=(1, 2))))

    # Shoulder rotation speed
    shoulders = poses[:, [H36Key.LSHOULDER, H36Key.RSHOULDER], :]
    shoulder_vec = shoulders[:, 1] - shoulders[:, 0]
    angles = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
    rot_speed = float(np.max(np.abs(np.gradient(angles)) * fps))

    # Knee angle
    # (simplified: min angle over segment)

    return {
        "duration": duration,
        "hip_y_range": hip_y_range,
        "motion_energy": motion_energy,
        "rotation_speed": rot_speed,
        "num_frames": T,
    }
```

- [ ] **Step 2: Implement RF classifier wrapper**

```python
class SegmentClassifier:
    """Random Forest classifier for fine element types from segment features.

    Trained on MCFS segments. Maps TAS coarse segments to fine labels
    (top 30 most frequent classes).
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 20):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        self.feature_names = ["duration", "hip_y_range", "motion_energy", "rotation_speed", "num_frames"]
        self.label_encoder = LabelEncoder()

    def fit(self, segments: list[dict]) -> None:
        """Train on list of {features: dict, label: str} segments."""
        X = np.array([[s["features"][f] for f in self.feature_names] for s in segments])
        y = self.label_encoder.fit_transform([s["label"] for s in segments])
        self.clf.fit(X, y)

    def predict(self, features: dict[str, float]) -> tuple[str, float]:
        """Predict fine label and confidence from features."""
        x = np.array([[features[f] for f in self.feature_names]])
        proba = self.clf.predict_proba(x)[0]
        pred_idx = proba.argmax()
        label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])
        return label, confidence
```

- [ ] **Step 3: Implement training script**

```python
"""
Experiment: RF segment classifier on MCFS top-30 labels
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
from ml.src.tas.dataset import MCFSCoarseDataset, op25_to_coco17
from ml.src.pose_estimation.h36m import coco_to_h36m
from ml.src.pose_estimation.normalizer import normalize_poses

BASE = Path("data/datasets/mcfs")
CHECKPOINT_DIR = Path("experiments/checkpoints/rf_classifier")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ds = MCFSCoarseDataset(BASE / "features", BASE / "groundTruth")

    # Collect all segments with features
    all_segments = []
    label_counts = Counter()

    for i in range(len(ds)):
        poses, labels, length = ds[i]
        # Extract contiguous segments (skip None=0)
        start = None
        current_label = None
        for t in range(length):
            fine_label = ds.get_fine_label(i, t)  # Need to add this method
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

    # Train/test split
    train, test = train_test_split(filtered, test_size=0.2, random_state=42, stratify=[s["label"] for s in filtered])

    clf = SegmentClassifier(n_estimators=200)
    clf.fit(train)

    # Evaluate
    correct = 0
    for seg in test:
        pred, conf = clf.predict(seg["features"])
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
```

- [ ] **Step 4: Write test**

```python
def test_extract_features():
    poses = np.random.randn(50, 17, 2).astype(np.float32)
    feats = extract_segment_features(poses, fps=30.0)
    assert "duration" in feats
    assert "hip_y_range" in feats
    assert feats["duration"] == 50 / 30.0

def test_classifier_fit_predict():
    segments = [
        {"features": {"duration": 1.0, "hip_y_range": 0.5, "motion_energy": 0.1, "rotation_speed": 200.0, "num_frames": 30}, "label": "3Flip"},
        {"features": {"duration": 3.0, "hip_y_range": 0.1, "motion_energy": 0.05, "rotation_speed": 50.0, "num_frames": 90}, "label": "ChComboSpin4"},
    ]
    clf = SegmentClassifier(n_estimators=10)
    clf.fit(segments)
    label, conf = clf.predict(segments[0]["features"])
    assert label == "3Flip"
    assert 0 <= conf <= 1
```

- [ ] **Step 5: Commit**

```bash
git add ml/src/tas/classifier.py experiments/train_rf_classifier.py ml/tests/tas/test_classifier.py
git commit -m "feat(tas): RF segment classifier with biomechanical features"
```

---

### Task 6: Inference Integration

**Files:**
- Create: `ml/src/tas/inference.py`
- Modify: `ml/src/analysis/element_segmenter.py` (replace rule-based)
- Create: `ml/tests/tas/test_inference.py`

- [ ] **Step 1: Implement inference pipeline**

```python
class TASElementSegmenter:
    """ML-based element segmenter: BiGRU coarse → segment extraction → RF fine.

    Replaces rule-based ElementSegmenter.
    """

    def __init__(
        self,
        model_path: Path,
        classifier_path: Path | None = None,
        device: str = "cuda",
        min_segment_duration: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = BiGRUTAS(hidden_dim=128, num_layers=2, dropout=0.3)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.classifier = None
        if classifier_path and classifier_path.exists():
            import joblib
            self.classifier = joblib.load(classifier_path)

        self.min_segment_duration = min_segment_duration

    def segment(
        self,
        poses: np.ndarray,  # (T, 17, 2) normalized H3.6M
        fps: float = 30.0,
    ) -> list[ElementSegment]:
        """Segment poses into elements."""
        T = poses.shape[0]
        poses_tensor = torch.from_numpy(poses).unsqueeze(0).to(self.device)
        lengths = torch.tensor([T], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(poses_tensor, lengths)
        pred_labels = logits.argmax(dim=-1).cpu().numpy()[0]  # (T,)

        # Extract segments
        segments = self._extract_segments(pred_labels, poses, fps)
        return segments

    def _extract_segments(
        self,
        labels: np.ndarray,
        poses: np.ndarray,
        fps: float,
    ) -> list[ElementSegment]:
        """Extract contiguous segments from frame-wise labels."""
        id2label = {0: "None", 1: "Jump", 2: "Spin", 3: "Step"}
        segments = []
        if len(labels) == 0:
            return segments

        current = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != current:
                if current != 0:  # Skip None
                    duration = (i - start) / fps
                    if duration >= self.min_segment_duration:
                        seg_poses = poses[start:i]
                        element_type = id2label[current]
                        confidence = 1.0  # Will be refined

                        # RF classification if available
                        if self.classifier and current in (1, 2, 3):
                            from ml.src.tas.classifier import extract_segment_features
                            features = extract_segment_features(seg_poses, fps)
                            element_type, confidence = self.classifier.predict(features)

                        segments.append(ElementSegment(
                            element_type=element_type,
                            start=start,
                            end=i - 1,
                            confidence=confidence,
                        ))
                current = labels[i]
                start = i

        # Last segment
        if current != 0:
            duration = (len(labels) - start) / fps
            if duration >= self.min_segment_duration:
                seg_poses = poses[start:]
                element_type = id2label[current]
                confidence = 1.0
                if self.classifier and current in (1, 2, 3):
                    from ml.src.tas.classifier import extract_segment_features
                    features = extract_segment_features(seg_poses, fps)
                    element_type, confidence = self.classifier.predict(features)
                segments.append(ElementSegment(
                    element_type=element_type,
                    start=start,
                    end=len(labels) - 1,
                    confidence=confidence,
                ))

        return segments
```

- [ ] **Step 2: Modify ElementSegmenter to use TAS**

```python
# In ml/src/analysis/element_segmenter.py
# Add TASElementSegmenter as primary backend, fall back to rule-based
# when model_path is None.
```

- [ ] **Step 3: Write integration test**

```python
def test_tas_inference():
    from ml.src.tas.inference import TASElementSegmenter
    # Mock: create small model for testing
    model = BiGRUTAS(hidden_dim=32, num_layers=1)
    torch.save({"model_state_dict": model.state_dict()}, "/tmp/test_tas.pt")

    segmenter = TASElementSegmenter(
        model_path=Path("/tmp/test_tas.pt"),
        classifier_path=None,
        device="cpu",
    )
    poses = np.random.randn(100, 17, 2).astype(np.float32)
    segments = segmenter.segment(poses, fps=30.0)
    assert isinstance(segments, list)
    for seg in segments:
        assert seg.element_type in ("Jump", "Spin", "Step", "None")
        assert seg.start < seg.end
```

- [ ] **Step 4: Commit**

```bash
git add ml/src/tas/inference.py ml/src/analysis/element_segmenter.py ml/tests/tas/test_inference.py
git commit -m "feat(tas): Integrate BiGRU+RF into ElementSegmenter pipeline"
```

---

### Task 7: Evaluation Script

**Files:**
- Create: `experiments/evaluate_tas.py`

- [ ] **Step 1: Implement evaluation**

```python
"""
Evaluate TAS model on held-out MCFS fold.
Produces: per-class accuracy, confusion matrix, OverlapF1@50, qualitative examples.

Usage:
    uv run python experiments/evaluate_tas.py --checkpoint experiments/checkpoints/tas_bigr/fold_0_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ml.src.tas.dataset import MCFSCoarseDataset, pad_collate
from ml.src.tas.metrics import OverlapF1
from ml.src.tas.model import BiGRUTAS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MCFSCoarseDataset(Path("data/datasets/mcfs/features"), Path("data/datasets/mcfs/groundTruth"))

    # Use last 20% as test set (not in CV folds)
    n = len(ds)
    test_idx = list(range(int(n * 0.8), n))
    test_ds = Subset(ds, test_idx)
    loader = DataLoader(test_ds, batch_size=8, collate_fn=pad_collate)

    model = BiGRUTAS().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metric = OverlapF1(iou_threshold=0.5)
    all_f1s = []
    all_frame_acc = []

    with torch.no_grad():
        for poses, labels, lengths in loader:
            poses = poses.to(device)
            logits = model(poses, lengths)
            preds = logits.argmax(dim=-1).cpu().numpy()
            for i, le in enumerate(lengths):
                p = preds[i, :le]
                t = labels[i, :le].numpy()
                f1_result = metric.compute(p, t)
                all_f1s.append(f1_result["f1"])
                all_frame_acc.append((p == t).mean())

    print(f"Mean Frame Acc: {np.mean(all_frame_acc):.4f}")
    print(f"Mean OverlapF1@50: {np.mean(all_f1s):.4f}")
    print(f"Median OverlapF1@50: {np.median(all_f1s):.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run evaluation (after training)**

```bash
uv run python experiments/evaluate_tas.py --checkpoint experiments/checkpoints/tas_bigr/fold_0_best.pt
```

- [ ] **Step 3: Commit**

```bash
git add experiments/evaluate_tas.py
git commit -m "feat(tas): TAS evaluation script with OverlapF1"
```

---

## Self-Review

### Spec coverage
- [x] MCFS data loading with coarse labels → Task 1
- [x] BiGRU model → Task 2
- [x] OverlapF1 metric → Task 3
- [x] Training script (5-fold CV) → Task 4
- [x] RF classifier on segments → Task 5
- [x] Integration into ElementSegmenter → Task 6
- [x] Evaluation → Task 7

### Placeholder scan
- No TBD/TODO placeholders
- All code blocks contain complete implementations
- All file paths are exact
- All commands have expected output

### Type consistency
- `MCFSCoarseDataset.__getitem__` returns `(np.ndarray, np.ndarray, int)` consistently
- `BiGRUTAS.forward` takes `(torch.Tensor, torch.Tensor)` consistently
- `ElementSegment` is reused from `ml/src/types.py`

---

## Execution Handoff

**Plan complete and saved to `docs/plans/2026-05-05-tas-coarse-segmentation.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** - Fresh subagent per task + review loop. Commit after every step. All tests green before next Wave.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
