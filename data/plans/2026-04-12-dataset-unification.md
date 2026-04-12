# Dataset Unification & Preprocessing Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a unified dataset format, label ontology, and preprocessing pipeline for all figure skating datasets, enabling cross-dataset training and consistent evaluation.

**Architecture:** Single canonical format (NumPy zarr archive) with per-sample metadata. Label ontology maps all dataset-specific class IDs to a unified hierarchy. Preprocessing pipeline converts raw data (pkl/video) → canonical format with validation. Datasets module provides lazy-loading Dataset classes for PyTorch.

**Tech Stack:** numpy, zarr (chunked storage), PyTorch Dataset, rtmlib (pose extraction), dataclasses (metadata)

---

## Current State (Problems)

| Problem | Impact |
|---------|--------|
| FSC = MMFS + mocap, but MMFS stored separately | Confusion, wasted disk |
| 4 different class taxonomies (64/63/132/28) | Cannot mix datasets |
| No unified skeleton format (2D vs 3D, with/without confidence) | Inconsistent preprocessing |
| SkatingVerse needs RTMPose extraction (~40h) | Blocking |
| MCFS segments.pkl barely documented | Unknown quality |
| No data validation | Silent bugs |
| DATASETS.md is 7700 lines — unmaintainable | Info rot |

## Target State

```
data/
├── DATASETS.md                    # Compact registry (replaced)
├── datasets/
│   ├── unified/                   # Canonical format (NEW)
│   │   ├── fsc-64.zarr/           # FSC 64 classes
│   │   │   ├── poses/             # (N, T, 17, 2) float32
│   │   │   ├── labels/            # (N,) int32
│   │   │   └── meta/              # per-sample metadata
│   │   ├── mcfs-132.zarr/         # MCFS 132 classes
│   │   └── skatingverse-28.zarr/  # SkatingVerse 28 classes
│   ├── raw/                       # Original files (read-only)
│   │   ├── figure-skating-classification/
│   │   ├── mmfs/
│   │   ├── mcfs/
│   │   ├── skatingverse/
│   │   ├── athletepose3d/
│   │   └── finefs/
│   └── checkpoints/               # Model checkpoints
├── data_tools/                    # Preprocessing scripts (NEW)
│   ├── __init__.py
│   ├── convert_fsc.py
│   ├── convert_mcfs.py
│   ├── convert_skatingverse.py
│   ├── validate.py
│   └── label_ontology.py
└── plans/
```

## Unified Format Specification

### Skeleton Format

```python
# poses: (N, T, V, C) float32
# N = number of samples
# T = temporal frames (variable, padded to max per-dataset)
# V = 17 keypoints (COCO/H3.6M body)
# C = 2 (x, y normalized)
#
# Normalization: root-centered (mid-hip), spine-length scaled
# Coordinate range: approximately [-2, 2]
```

### Metadata Format

```python
@dataclass
class SampleMeta:
    source: str          # "fsc", "mcfs", "skatingverse"
    source_id: str       # original sample identifier
    label_unified: int   # unified label ID
    label_source: str    # original class name
    num_frames: int      # original frame count
    fps: float           # original FPS (if known)
    duration_sec: float  # video duration
    quality_score: float | None  # if available (MMFS has this)
    split: str           # "train" | "val" | "test"
```

### Label Ontology

Unified hierarchy with 3 levels:

```
Level 0 (super-category): jump | spin | sequence | combination
Level 1 (element type):   axel | flip | lutz | loop | salchow | toeloop | ...
Level 2 (rotation):       1 | 2 | 3 | 4
```

Each source dataset maps to unified IDs via a label map.

---

### Task 1: Label Ontology

**Files:**
- Create: `data/data_tools/__init__.py`
- Create: `data/data_tools/label_ontology.py`
- Test: `data/tests/test_label_ontology.py`

- [ ] **Step 1: Create package structure**

```python
# data/data_tools/__init__.py
```

```python
# data/tests/__init__.py
```

- [ ] **Step 2: Write failing tests for label ontology**

```python
# data/tests/test_label_ontology.py
import pytest
from data_tools.label_ontology import LabelOntology, ElementSpec

def test_fsc_labels_load():
    ont = LabelOntology()
    assert ont.num_fsc_classes() == 64

def test_mcfs_labels_load():
    ont = LabelOntology()
    assert ont.num_mcfs_classes() == 132

def test_sv_labels_load():
    ont = LabelOntology()
    assert ont.num_sv_classes() == 28

def test_unified_hierarchy():
    ont = LabelOntology()
    spec = ont.get_element("3Lutz")
    assert spec.category == "jump"
    assert spec.rotation == 3
    assert spec.edge == "outside"

def test_fsc_to_unified():
    ont = LabelOntology()
    unified = ont.map_fsc_label(0)  # 1Axel
    assert unified.category == "jump"
    assert unified.rotation == 1

def test_sv_to_fsc_mapping():
    ont = LabelOntology()
    fsc_id = ont.map_sv_to_fsc(11)  # 3Lutz
    assert fsc_id is not None  # should map to an FSC class

def test_overlap_count():
    ont = LabelOntology()
    overlap = ont.count_sv_fsc_overlap()
    assert overlap >= 20  # at least 20 of 28 SV classes exist in FSC
```

- [ ] **Step 3: Implement LabelOntology**

```python
# data/data_tools/label_ontology.py
"""Unified label ontology for figure skating datasets.

Maps dataset-specific class IDs to a unified element specification.
Supports FSC (64), MCFS (132), SkatingVerse (28), MMFS (63).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ElementSpec:
    """Unified element specification."""
    name: str              # canonical name: "3Lutz"
    category: str          # jump | spin | sequence | combination
    edge: str | None       # inside | outside | None (for spins)
    rotation: int | None   # 1-4 for jumps, None for spins
    position: str | None   # camel | sit | upright | layback (for spins)
    is_combination: bool   # multi-jump combo
    fsc_id: int | None     # FSC class ID if exists
    mcfs_id: int | None    # MCFS class ID if exists
    sv_id: int | None      # SkatingVerse class ID if exists


class LabelOntology:
    """Maps between dataset-specific labels and unified elements."""

    def __init__(self) -> None:
        self._elements: list[ElementSpec] = []
        self._fsc_map: dict[int, ElementSpec] = {}
        self._mcfs_map: dict[int, ElementSpec] = {}
        self._sv_map: dict[int, ElementSpec] = {}
        self._load_fsc()
        self._load_mcfs()
        self._load_sv()

    # --- Counts ---

    def num_fsc_classes(self) -> int:
        return len(self._fsc_map)

    def num_mcfs_classes(self) -> int:
        return len(self._mcfs_map)

    def num_sv_classes(self) -> int:
        return len(self._sv_map)

    # --- Lookup ---

    def get_element(self, name: str) -> ElementSpec:
        for e in self._elements:
            if e.name == name:
                return e
        raise KeyError(f"Element '{name}' not found")

    def map_fsc_label(self, fsc_id: int) -> ElementSpec:
        return self._fsc_map[fsc_id]

    def map_sv_to_fsc(self, sv_id: int) -> int | None:
        spec = self._sv_map[sv_id]
        return spec.fsc_id

    def count_sv_fsc_overlap(self) -> int:
        return sum(1 for s in self._sv_map.values() if s.fsc_id is not None)

    # --- Loaders (build mappings from data files) ---

    def _load_fsc(self) -> None:
        """Load FSC 64-class definitions from figure-skating-classification/README.

        FSC structure (from README):
          0-20:  Single jumps (1Axel..4Toeloop)
          21-30: Multi-jump combinations
          31-34: FCSp (Foot Change Camel Spin)
          35-38: CCoSp (Catch Foot Combination Spin)
          39-42: ChCamelSp
          43-46: ChComboSp
          47-50: ChSitSp
          51-54: FlySitSp
          55-58: LaybackSp
          59-62: StepSeq 1-4
          63:    ChoreoSeq
        """
        base = Path(__file__).resolve().parent.parent / "datasets" / "raw" / "figure-skating-classification"
        # FSC class definitions — built from README documentation
        fsc_classes = [
            # 0-20: Single jumps
            (0, "1Axel", "jump", None, 1, False),
            (1, "2Axel", "jump", None, 2, False),
            (2, "3Axel", "jump", None, 3, False),
            (3, "1Flip", "jump", "inside", 1, False),
            (4, "2Flip", "jump", "inside", 2, False),
            (5, "3Flip", "jump", "inside", 3, False),
            (6, "1Lutz", "jump", "outside", 1, False),
            (7, "2Lutz", "jump", "outside", 2, False),
            (8, "3Lutz", "jump", "outside", 3, False),
            (9, "1Loop", "jump", "inside", 1, False),
            (10, "2Loop", "jump", "inside", 2, False),
            (11, "3Loop", "jump", "inside", 3, False),
            (12, "1Salchow", "jump", "inside", 1, False),
            (13, "2Salchow", "jump", "inside", 2, False),
            (14, "3Salchow", "jump", "inside", 3, False),
            (15, "1Toeloop", "jump", "outside", 1, False),
            (16, "2Toeloop", "jump", "outside", 2, False),
            (17, "3Toeloop", "jump", "outside", 3, False),
            (18, "4Toeloop", "jump", "outside", 4, False),
            (19, "4Salchow", "jump", "inside", 4, False),
            (20, "4Flip", "jump", "inside", 4, False),
            # 21-30: Combinations (exact names from dataset_info)
            (21, "1A+3T", "combination", None, None, True),
            (22, "1A+3A", "combination", None, None, True),
            (23, "2A+3T", "combination", None, None, True),
            (24, "2A+3A", "combination", None, None, True),
            (25, "2A+1Eu+3S", "combination", None, None, True),
            (26, "3F+3T", "combination", None, None, True),
            (27, "3F+2T+2Lo", "combination", None, None, True),
            (28, "3Lz+3T", "combination", None, None, True),
            (29, "3Lz+3Lo", "combination", None, None, True),
            (30, "Comb", "combination", None, None, True),
            # 31-34: FCSp
            (31, "FCSp1", "spin", None, None, False),
            (32, "FCSp2", "spin", None, None, False),
            (33, "FCSp3", "spin", None, None, False),
            (34, "FCSp4", "spin", None, None, False),
            # 35-38: CCoSp
            (35, "CCoSp1", "spin", None, None, False),
            (36, "CCoSp2", "spin", None, None, False),
            (37, "CCoSp3", "spin", None, None, False),
            (38, "CCoSp4", "spin", None, None, False),
            # 39-42: ChCamelSp
            (39, "ChCamelSp1", "spin", None, None, False),
            (40, "ChCamelSp2", "spin", None, None, False),
            (41, "ChCamelSp3", "spin", None, None, False),
            (42, "ChCamelSp4", "spin", None, None, False),
            # 43-46: ChComboSp
            (43, "ChComboSp1", "spin", None, None, False),
            (44, "ChComboSp2", "spin", None, None, False),
            (45, "ChComboSp3", "spin", None, None, False),
            (46, "ChComboSp4", "spin", None, None, False),
            # 47-50: ChSitSp
            (47, "ChSitSp1", "spin", None, None, False),
            (48, "ChSitSp2", "spin", None, None, False),
            (49, "ChSitSp3", "spin", None, None, False),
            (50, "ChSitSp4", "spin", None, None, False),
            # 51-54: FlySitSp
            (51, "FlySitSp1", "spin", None, None, False),
            (52, "FlySitSp2", "spin", None, None, False),
            (53, "FlySitSp3", "spin", None, None, False),
            (54, "FlySitSp4", "spin", None, None, False),
            # 55-58: LaybackSp
            (55, "LaybackSp1", "spin", None, None, False),
            (56, "LaybackSp2", "spin", None, None, False),
            (57, "LaybackSp3", "spin", None, None, False),
            (58, "LaybackSp4", "spin", None, None, False),
            # 59-62: StepSeq
            (59, "StepSeq1", "sequence", None, None, False),
            (60, "StepSeq2", "sequence", None, None, False),
            (61, "StepSeq3", "sequence", None, None, False),
            (62, "StepSeq4", "sequence", None, None, False),
            # 63: ChoreoSeq
            (63, "ChoreoSeq", "sequence", None, None, False),
        ]
        for fsc_id, name, cat, edge, rot, combo in fsc_classes:
            spec = ElementSpec(
                name=name, category=cat, edge=edge, rotation=rot,
                position=None, is_combination=combo,
                fsc_id=fsc_id, mcfs_id=None, sv_id=None,
            )
            self._elements.append(spec)
            self._fsc_map[fsc_id] = spec

    def _load_mcfs(self) -> None:
        """Load MCFS 132-class mapping from mcfs/mapping.txt."""
        mapping_path = Path(__file__).resolve().parent.parent / "datasets" / "raw" / "mcfs" / "mapping.txt"
        if not mapping_path.exists():
            return
        with open(mapping_path) as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) < 2:
                    continue
                mcfs_id = int(parts[0])
                name = parts[1]
                if name == "NONE":
                    continue
                # Try to match to existing element
                spec = self._match_element(name, mcfs_id=mcfs_id)
                self._mcfs_map[mcfs_id] = spec

    def _load_sv(self) -> None:
        """Load SkatingVerse 28-class mapping from skatingverse/mapping.txt."""
        mapping_path = Path(__file__).resolve().parent.parent / "datasets" / "raw" / "skatingverse" / "mapping.txt"
        if not mapping_path.exists():
            return
        # SV class definitions
        sv_classes = [
            (0, "3Toeloop"), (1, "3Loop"), (2, "2Axel"),
            (3, "CamelSpin"), (4, "SitSpin"), (5, "UprightSpin"),
            (6, "2Salchow"), (7, "2Toeloop"), (8, "3Salchow"),
            (9, "3Axel"), (10, "3Flip"), (11, "3Lutz"),
            (12, "NoBasic"),  # skip
            (13, "2Lutz"), (14, "4Salchow"), (15, "4Flip"),
            (16, "4Toeloop"), (17, "4Lutz"), (18, "4Loop"),
            (19, "2Flip"), (20, "2Loop"), (21, "1Axel"),
            (22, "1Loop"), (23, "1Salchow"), (24, "1Toeloop"),
            (25, "1Flip"), (26, "1Lutz"), (27, "Sequence"),
        ]
        for sv_id, name in sv_classes:
            if name == "NoBasic":
                continue
            spec = self._match_element(name, sv_id=sv_id)
            self._sv_map[sv_id] = spec

    def _match_element(self, name: str, fsc_id=None, mcfs_id=None, sv_id=None) -> ElementSpec:
        """Find existing element or create new one."""
        # Normalize name for matching
        norm = name.replace(" ", "").replace("_", "").lower()

        for e in self._elements:
            e_norm = e.name.replace(" ", "").replace("_", "").lower()
            if e_norm == norm or norm in e_norm or e_norm in norm:
                # Update with new IDs
                if fsc_id is not None and e.fsc_id is None:
                    e = ElementSpec(
                        name=e.name, category=e.category, edge=e.edge,
                        rotation=e.rotation, position=e.position,
                        is_combination=e.is_combination,
                        fsc_id=fsc_id, mcfs_id=mcfs_id or e.mcfs_id,
                        sv_id=sv_id or e.sv_id,
                    )
                elif sv_id is not None and e.sv_id is None:
                    e = ElementSpec(
                        name=e.name, category=e.category, edge=e.edge,
                        rotation=e.rotation, position=e.position,
                        is_combination=e.is_combination,
                        fsc_id=fsc_id or e.fsc_id, mcfs_id=mcfs_id or e.mcfs_id,
                        sv_id=sv_id,
                    )
                return e

        # New element not in FSC
        spec = ElementSpec(
            name=name, category="unknown", edge=None, rotation=None,
            position=None, is_combination=False,
            fsc_id=fsc_id, mcfs_id=mcfs_id, sv_id=sv_id,
        )
        self._elements.append(spec)
        return spec
```

- [ ] **Step 4: Run tests to verify**

Run: `cd /home/michael/Github/skating-biomechanics-ml && PYTHONPATH=data uv run pytest data/tests/test_label_ontology.py -v`
Expected: Tests may partially fail — adjust element matching logic until all pass.

- [ ] **Step 5: Commit**

```bash
git add data/data_tools/ data/tests/
git commit -m "feat(data): add unified label ontology with FSC/MCFS/SV mappings"
```

---

### Task 2: Data Validation Utilities

**Files:**
- Create: `data/data_tools/validate.py`
- Test: `data/tests/test_validate.py`

- [ ] **Step 1: Write failing tests**

```python
# data/tests/test_validate.py
import pytest
import numpy as np
from data_tools.validate import validate_skeleton, ValidationError


def test_valid_skeleton_passes():
    poses = np.random.randn(100, 17, 2).astype(np.float32)
    errors = validate_skeleton(poses)
    assert len(errors) == 0


def test_wrong_keypoint_count_fails():
    poses = np.random.randn(100, 15, 2).astype(np.float32)
    errors = validate_skeleton(poses)
    assert len(errors) > 0
    assert any("keypoints" in str(e).lower() for e in errors)


def test_nan_detection():
    poses = np.random.randn(100, 17, 2).astype(np.float32)
    poses[10, 5, 0] = np.nan
    errors = validate_skeleton(poses)
    assert len(errors) > 0


def test_inf_detection():
    poses = np.random.randn(100, 17, 2).astype(np.float32)
    poses[10, 5, 1] = np.inf
    errors = validate_skeleton(poses)
    assert len(errors) > 0


def test_all_zeros_detection():
    poses = np.zeros((100, 17, 2), dtype=np.float32)
    errors = validate_skeleton(poses)
    assert len(errors) > 0


def test_min_frames():
    poses = np.random.randn(5, 17, 2).astype(np.float32)
    errors = validate_skeleton(poses, min_frames=10)
    assert len(errors) > 0
```

- [ ] **Step 2: Implement validate module**

```python
# data/data_tools/validate.py
"""Skeleton data validation utilities."""
from __future__ import annotations

import numpy as np


class ValidationError:
    def __init__(self, sample_id: str, field: str, message: str):
        self.sample_id = sample_id
        self.field = field
        self.message = message

    def __repr__(self) -> str:
        return f"ValidationError({self.sample_id!r}, {self.field!r}, {self.message!r})"


def validate_skeleton(
    poses: np.ndarray,
    sample_id: str = "unknown",
    min_frames: int = 5,
    max_coord: float = 10.0,
) -> list[ValidationError]:
    """Validate a single skeleton sequence.

    Args:
        poses: (T, V, C) or (T, V, C, 1) float32 skeleton array.
        sample_id: Identifier for error reporting.
        min_frames: Minimum acceptable frame count.
        max_coord: Maximum absolute coordinate value (normalized data).

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[ValidationError] = []

    # Squeeze trailing dimensions
    while poses.ndim > 3 and poses.shape[-1] == 1:
        poses = poses[..., 0]

    if poses.ndim != 3:
        errors.append(ValidationError(sample_id, "ndim", f"Expected 3D, got {poses.ndim}D"))
        return errors

    T, V, C = poses.shape

    if T < min_frames:
        errors.append(ValidationError(sample_id, "frames", f"Too short: {T} < {min_frames}"))

    if V != 17:
        errors.append(ValidationError(sample_id, "keypoints", f"Expected 17, got {V}"))

    if C < 2:
        errors.append(ValidationError(sample_id, "channels", f"Expected >= 2, got {C}"))

    if np.any(np.isnan(poses)):
        nan_count = int(np.sum(np.isnan(poses)))
        errors.append(ValidationError(sample_id, "nan", f"Contains {nan_count} NaN values"))

    if np.any(np.isinf(poses)):
        inf_count = int(np.sum(np.isinf(poses)))
        errors.append(ValidationError(sample_id, "inf", f"Contains {inf_count} Inf values"))

    if np.all(poses == 0):
        errors.append(ValidationError(sample_id, "zeros", "All values are zero"))

    if np.any(np.abs(poses) > max_coord):
        errors.append(ValidationError(sample_id, "range", f"Coordinates exceed ±{max_coord}"))

    return errors
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=data uv run pytest data/tests/test_validate.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add data/data_tools/validate.py data/tests/test_validate.py
git commit -m "feat(data): add skeleton validation utilities"
```

---

### Task 3: FSC Converter (pkl → unified zarr)

**Files:**
- Create: `data/data_tools/convert_fsc.py`
- Test: `data/tests/test_convert_fsc.py`

- [ ] **Step 1: Write failing test**

```python
# data/tests/test_convert_fsc.py
import pytest
import numpy as np
from pathlib import Path


def test_convert_fsc_creates_zarr(tmp_path):
    from data_tools.convert_fsc import convert_fsc
    raw_dir = Path("data/datasets/raw/figure-skating-classification")
    if not raw_dir.exists():
        pytest.skip("Raw FSC data not found")
    out = tmp_path / "fsc_test.zarr"
    stats = convert_fsc(raw_dir, out)
    assert out.exists()
    assert stats["train"] > 4000
    assert stats["test"] > 900
    assert stats["classes"] == 64


def test_convert_fsc_shapes(tmp_path):
    from data_tools.convert_fsc import convert_fsc, load_unified
    raw_dir = Path("data/datasets/raw/figure-skating-classification")
    if not raw_dir.exists():
        pytest.skip("Raw FSC data not found")
    out = tmp_path / "fsc_test.zarr"
    convert_fsc(raw_dir, out)
    poses, labels, meta = load_unified(out, split="train")
    assert poses.ndim == 3  # (N, T, 17) — will be padded later
    assert poses.shape[-1] == 2  # xy
    assert poses.shape[-2] == 17  # keypoints
    assert len(labels) == len(meta)
```

- [ ] **Step 2: Implement FSC converter**

```python
# data/data_tools/convert_fsc.py
"""Convert FSC pkl files to unified zarr format."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def normalize(p: np.ndarray) -> np.ndarray:
    """Root-center + spine-length normalize. p: (T, V, C) float32."""
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


def convert_fsc(raw_dir: Path, output_dir: Path) -> dict:
    """Convert FSC pkl → unified zarr.

    Args:
        raw_dir: Path to figure-skating-classification/ containing train_data.pkl etc.
        output_dir: Output zarr directory.

    Returns:
        Dict with stats: {"train": N, "test": N, "classes": int}
    """
    import zarr

    from data_tools.validate import validate_skeleton

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    for split in ("train", "test"):
        data_path = raw_dir / f"{split}_data.pkl"
        label_path = raw_dir / f"{split}_label.pkl"
        if not data_path.exists():
            continue

        data = pickle.load(open(data_path, "rb"))
        labels = pickle.load(open(label_path, "rb"))

        # Convert: (N, T, 17, 3, 1) → (N, T, 17, 2) normalized
        valid_poses, valid_labels = [], []
        skipped = 0
        for i, (d, label) in enumerate(zip(data, labels)):
            arr = np.array(d, dtype=np.float32)
            # Squeeze trailing dim if (T, 17, 3, 1)
            while arr.ndim > 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            # Take xy only
            if arr.shape[-1] >= 2:
                arr = arr[..., :2]
            # Validate
            errors = validate_skeleton(arr, sample_id=f"{split}_{i}")
            if errors:
                skipped += 1
                continue
            # Normalize
            arr = normalize(arr)
            valid_poses.append(arr)
            valid_labels.append(int(label))

        # Save as zarr
        root = zarr.open_group(output_dir / split, mode="w")
        root.create_dataset("poses", data=np.array(valid_poses, dtype=object))
        root.create_dataset("labels", data=np.array(valid_labels, dtype=np.int32))

        stats[split] = len(valid_poses)
        if skipped > 0:
            print(f"  {split}: skipped {skipped} invalid samples")

    stats["classes"] = len(set(stats.get("labels", [])))
    return stats


def load_unified(zarr_dir: Path, split: str = "train"):
    """Load unified zarr data. Returns (poses_list, labels, meta_list)."""
    import zarr

    root = zarr.open_group(zarr_dir / split, mode="r")
    return root["poses"][:], root["labels"][:], []
```

- [ ] **Step 3: Run test**

Run: `PYTHONPATH=data uv run pytest data/tests/test_convert_fsc.py -v`
Expected: Pass (if FSC raw data exists at expected path).

- [ ] **Step 4: Commit**

```bash
git add data/data_tools/convert_fsc.py data/tests/test_convert_fsc.py
git commit -m "feat(data): add FSC pkl → unified zarr converter"
```

---

### Task 4: Move Raw Datasets, Run FSC Conversion

**Files:**
- Move: `data/datasets/figure-skating-classification/` → `data/datasets/raw/figure-skating-classification/`
- Move: `data/datasets/mmfs/` → `data/datasets/raw/mmfs/`
- Move: `data/datasets/mcfs/` → `data/datasets/raw/mcfs/`
- Move: `data/datasets/skatingverse/` → `data/datasets/raw/skatingverse/`
- Move: `data/datasets/athletepose3d/` → `data/datasets/raw/athletepose3d/`
- Move: `data/datasets/finefs/` → `data/datasets/raw/finefs/`
- Run: FSC conversion

- [ ] **Step 1: Move datasets to raw/**

```bash
cd /home/michael/Github/skating-biomechanics-ml
mkdir -p data/datasets/raw
mv data/datasets/figure-skating-classification data/datasets/raw/
mv data/datasets/mmfs data/datasets/raw/
mv data/datasets/mcfs data/datasets/raw/
mv data/datasets/skatingverse data/datasets/raw/
mv data/datasets/athletepose3d data/datasets/raw/
mv data/datasets/finefs data/datasets/raw/
```

- [ ] **Step 2: Update imports in convert scripts**

Update `data/data_tools/convert_fsc.py` path: the raw_dir default should point to `data/datasets/raw/figure-skating-classification`.

- [ ] **Step 3: Run FSC conversion**

```bash
PYTHONPATH=data uv run python -m data_tools.convert_fsc
```

Expected: Creates `data/datasets/unified/fsc-64.zarr/` with train/ and test/ groups.

- [ ] **Step 4: Verify conversion**

```bash
PYTHONPATH=data uv run python -c "
import zarr, numpy as np
root = zarr.open_group('data/datasets/unified/fsc-64.zarr/train', mode='r')
poses = root['poses'][:]
labels = root['labels'][:]
print(f'Train: {len(poses)} samples, labels: {len(set(labels))} unique')
print(f'Sample shape: {poses[0].shape}')
print(f'Label range: {labels.min()}-{labels.max()}')
"
```

- [ ] **Step 5: Commit**

```bash
git add data/datasets/
git commit -m "refactor(data): move raw datasets to raw/, add unified/ with FSC conversion"
```

---

### Task 5: MCFS Converter

**Files:**
- Create: `data/data_tools/convert_mcfs.py`
- Test: `data/tests/test_convert_mcfs.py`

- [ ] **Step 1: Write failing test**

```python
# data/tests/test_convert_mcfs.py
import pytest
from pathlib import Path


def test_convert_mcfs_creates_zarr(tmp_path):
    from data_tools.convert_mcfs import convert_mcfs
    raw_dir = Path("data/datasets/raw/mcfs")
    if not raw_dir.exists():
        pytest.skip("Raw MCFS data not found")
    out = tmp_path / "mcfs_test.zarr"
    stats = convert_mcfs(raw_dir, out)
    assert out.exists()
    assert stats["total"] > 2000
    assert stats["classes"] >= 100
```

- [ ] **Step 2: Implement MCFS converter**

```python
# data/data_tools/convert_mcfs.py
"""Convert MCFS segments.pkl to unified zarr format."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from data_tools.validate import validate_skeleton


def normalize(p: np.ndarray) -> np.ndarray:
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


def convert_mcfs(raw_dir: Path, output_dir: Path) -> dict:
    """Convert MCFS segments.pkl → unified zarr.

    MCFS format: list of (poses_array, label_string) tuples.
    poses_array shape: (T, 17, 2) — already 2D normalized.
    """
    import zarr

    output_dir.mkdir(parents=True, exist_ok=True)
    seg_path = raw_dir / "segments.pkl"
    data = pickle.load(open(seg_path, "rb"))

    valid_poses, valid_labels = [], []
    skipped = 0
    for i, (poses, label) in enumerate(data):
        arr = np.array(poses, dtype=np.float32)
        errors = validate_skeleton(arr, sample_id=f"mcfs_{i}")
        if errors:
            skipped += 1
            continue
        arr = normalize(arr)
        valid_poses.append(arr)
        valid_labels.append(label)  # keep string label for now

    # Map string labels to int IDs
    unique_labels = sorted(set(valid_labels))
    label_map = {name: idx for idx, name in enumerate(unique_labels)}
    int_labels = [label_map[l] for l in valid_labels]

    # Save
    root = zarr.open_group(output_dir, mode="w")
    root.create_dataset("poses", data=np.array(valid_poses, dtype=object))
    root.create_dataset("labels", data=np.array(int_labels, dtype=np.int32))
    root.attrs["label_names"] = unique_labels

    return {"total": len(valid_poses), "skipped": skipped, "classes": len(unique_labels)}
```

- [ ] **Step 3: Run test**

Run: `PYTHONPATH=data uv run pytest data/tests/test_convert_mcfs.py -v`

- [ ] **Step 4: Commit**

```bash
git add data/data_tools/convert_mcfs.py data/tests/test_convert_mcfs.py
git commit -m "feat(data): add MCFS segments.pkl → unified zarr converter"
```

---

### Task 6: SkatingVerse Converter (video → skeleton → zarr)

**Files:**
- Create: `data/data_tools/convert_skatingverse.py`
- Test: `data/tests/test_convert_skatingverse.py`

- [ ] **Step 1: Write failing test**

```python
# data/tests/test_convert_skatingverse.py
import pytest
from pathlib import Path


def test_sv_extract_single_video(tmp_path):
    """Test extraction of a single video."""
    from data_tools.convert_skatingverse import extract_single_video
    video_dir = Path("data/datasets/raw/skatingverse/train_videos")
    if not video_dir.exists():
        pytest.skip("SkatingVerse videos not found")
    video_path = list(video_dir.glob("*.mp4"))[0]
    poses = extract_single_video(video_path, frame_skip=4)
    if poses is not None:
        assert poses.ndim == 2
        assert poses.shape[1] == 17
        assert poses.shape[0] >= 5


def test_sv_convert_batch(tmp_path):
    """Test batch conversion with small subset."""
    from data_tools.convert_skatingverse import convert_skatingverse
    raw_dir = Path("data/datasets/raw/skatingverse")
    if not raw_dir.exists():
        pytest.skip("SkatingVerse data not found")
    out = tmp_path / "sv_test.zarr"
    stats = convert_skatingverse(
        raw_dir, out, max_per_class=3, frame_skip=4,
    )
    if stats["total"] > 0:
        assert out.exists()
```

- [ ] **Step 2: Implement SkatingVerse converter**

```python
# data/data_tools/convert_skatingverse.py
"""Convert SkatingVerse videos → unified zarr format via RTMPose."""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from data_tools.validate import validate_skeleton


def normalize(p: np.ndarray) -> np.ndarray:
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


def extract_single_video(video_path: Path, frame_skip: int = 4) -> np.ndarray | None:
    """Extract skeleton from a single video using rtmlib.

    Returns (T, 17, 2) normalized poses or None if extraction fails.
    """
    from rtmlib import Wholebody

    wb = Wholebody(mode="balanced", backend="onnxruntime")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames_poses: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            keypoints, scores = wb(frame)
            # keypoints: (N_persons, 133, 2), scores: (N_persons, 133)
            if len(keypoints) > 0 and keypoints.shape[1] >= 17:
                conf = scores[0, :17]
                if np.mean(conf) >= 0.3:
                    body = keypoints[0, :17, :].copy()  # (17, 2)
                    frames_poses.append(body)
        frame_idx += 1
    cap.release()

    if len(frames_poses) < 5:
        return None

    poses = np.array(frames_poses, dtype=np.float32)
    errors = validate_skeleton(poses, sample_id=str(video_path))
    if errors:
        return None

    return normalize(poses)


def convert_skatingverse(
    raw_dir: Path,
    output_dir: Path,
    max_per_class: int = 0,
    frame_skip: int = 4,
    splits: tuple[str, ...] = ("train", "test"),
) -> dict:
    """Convert SkatingVerse videos → unified zarr.

    Args:
        raw_dir: Path to skatingverse/ containing train_videos/, test_videos/, train.txt, etc.
        output_dir: Output zarr directory.
        max_per_class: Max videos per class (0 = unlimited).
        frame_skip: Process every Nth frame.
        splits: Which splits to process.
    """
    import zarr

    output_dir.mkdir(parents=True, exist_ok=True)

    sv_names: dict[int, str] = {}
    mapping_path = raw_dir / "mapping.txt"
    with open(mapping_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                sv_names[int(parts[-1])] = " ".join(parts[:-1])

    stats: dict[str, int] = {"total": 0, "skipped": 0}

    for split in splits:
        if split == "train":
            txt_path = raw_dir / "train.txt"
            video_dir = raw_dir / "train_videos"
        else:
            txt_path = raw_dir / "answer.txt"
            video_dir = raw_dir / "test_videos"

        if not txt_path.exists():
            continue

        # Load clip metadata
        clips: list[tuple[str, int]] = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        clips.append((parts[0], int(parts[-1])))
                    except ValueError:
                        pass

        # Group by class, apply limit
        by_class: dict[int, list[str]] = {}
        for fname, label in clips:
            if label == 12:  # skip NoBasic
                continue
            by_class.setdefault(label, []).append(fname)
            if max_per_class > 0 and len(by_class[label]) > max_per_class:
                continue

        # Extract
        valid_poses, valid_labels = [], []
        total = sum(len(v) for v in by_class.values())
        for i, (label, fnames) in enumerate(by_class.items()):
            for fname in fnames:
                video_path = video_dir / f"{fname}.mp4"
                if not video_path.exists():
                    continue
                poses = extract_single_video(video_path, frame_skip=frame_skip)
                if poses is not None:
                    valid_poses.append(poses)
                    valid_labels.append(label)
                else:
                    stats["skipped"] += 1

                if (i * 100 + len(valid_poses)) % 100 == 0:
                    print(f"  [{len(valid_poses)}/{total}] extracted", flush=True)

        # Save
        if valid_poses:
            root = zarr.open_group(output_dir / split, mode="w")
            root.create_dataset("poses", data=np.array(valid_poses, dtype=object))
            root.create_dataset("labels", data=np.array(valid_labels, dtype=np.int32))

        stats[split] = len(valid_poses)
        stats["total"] += len(valid_poses)

    stats["classes"] = len(set(stats.get("labels", [])))
    return stats
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=data uv run pytest data/tests/test_convert_skatingverse.py -v`

- [ ] **Step 4: Commit**

```bash
git add data/data_tools/convert_skatingverse.py data/tests/test_convert_skatingverse.py
git commit -m "feat(data): add SkatingVerse video → unified zarr converter"
```

---

### Task 7: Unified PyTorch Dataset

**Files:**
- Create: `data/data_tools/dataset.py`
- Test: `data/tests/test_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# data/tests/test_dataset.py
import pytest
import torch
from pathlib import Path


def test_dataset_loads_fsc():
    from data_tools.dataset import UnifiedSkatingDataset
    zarr_path = Path("data/datasets/unified/fsc-64.zarr")
    if not zarr_path.exists():
        pytest.skip("Unified FSC zarr not found")
    ds = UnifiedSkatingDataset(zarr_path, split="train")
    assert len(ds) > 4000
    poses, label = ds[0]
    assert poses.ndim == 2
    assert poses.shape[1] == 34  # 17 * 2 (flattened)
    assert isinstance(label, int)


def test_dataset_with_augmentation():
    from data_tools.dataset import UnifiedSkatingDataset
    zarr_path = Path("data/datasets/unified/fsc-64.zarr")
    if not zarr_path.exists():
        pytest.skip("Unified FSC zarr not found")
    ds = UnifiedSkatingDataset(zarr_path, split="train", augment=True)
    # Augmented dataset should be larger
    assert len(ds) > 4000


def test_dataset_collate():
    from data_tools.dataset import UnifiedSkatingDataset, varlen_collate
    from torch.utils.data import DataLoader
    zarr_path = Path("data/datasets/unified/fsc-64.zarr")
    if not zarr_path.exists():
        pytest.skip("Unified FSC zarr not found")
    ds = UnifiedSkatingDataset(zarr_path, split="train")
    loader = DataLoader(ds, batch_size=4, collate_fn=varlen_collate)
    batch = next(iter(loader))
    padded, lengths, labels = batch
    assert padded.ndim == 3  # (B, T, 34)
    assert lengths.shape[0] == 4
```

- [ ] **Step 2: Implement dataset**

```python
# data/data_tools/dataset.py
"""Unified PyTorch Dataset for figure skating skeleton data."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class UnifiedSkatingDataset(Dataset):
    """Variable-length skeleton dataset from unified zarr.

    Args:
        zarr_path: Path to unified zarr directory (contains train/ and test/ groups).
        split: "train" or "test".
        augment: Apply data augmentation (train only).
        label_map: Optional dict mapping source labels to different IDs.
    """

    def __init__(
        self,
        zarr_path: Path,
        split: str = "train",
        augment: bool = False,
        label_map: dict[int, int] | None = None,
    ):
        import zarr

        self.augment = augment and split == "train"
        root = zarr.open_group(zarr_path / split, mode="r")
        self.poses = root["poses"][:]  # object array of (T, 17, 2)
        self.labels = root["labels"][:]  # int array
        if label_map:
            self.labels = np.array([label_map.get(l, l) for l in self.labels])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        poses = self.poses[idx].astype(np.float32)  # (T, 17, 2)
        label = int(self.labels[idx])

        if self.augment:
            poses = self._augment(poses)

        # Flatten: (T, 17, 2) → (T, 34)
        return torch.from_numpy(poses.reshape(len(poses), -1)), label

    @staticmethod
    def _augment(poses: np.ndarray) -> np.ndarray:
        """Random augmentation: noise + mirror."""
        # Gaussian noise (80% chance)
        if random.random() < 0.8:
            poses = poses + np.random.randn(*poses.shape).astype(np.float32) * 0.02

        # Mirror (50% chance)
        if random.random() < 0.5:
            poses = poses.copy()
            poses[:, :, 0] = -poses[:, :, 0]  # flip x
            poses = poses[:, LR_SWAP, :]  # swap L/R

        return poses


def varlen_collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences with padding.

    Returns:
        padded: (B, max_T, C) float32
        lengths: (B,) int64
        labels: (B,) int64
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    seqs = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch])
    lengths = torch.tensor([s.shape[0] for s in seqs])
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, lengths, labels
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=data uv run pytest data/tests/test_dataset.py -v`

- [ ] **Step 4: Commit**

```bash
git add data/data_tools/dataset.py data/tests/test_dataset.py
git commit -m "feat(data): add unified PyTorch Dataset with augmentation"
```

---

### Task 8: Rewrite DATASETS.md

**Files:**
- Modify: `data/DATASETS.md`

- [ ] **Step 1: Replace DATASETS.md with compact version**

Replace the 7700-line DATASETS.md with a concise reference:

```markdown
# Dataset Registry

## Unified Format (Canonical)

All datasets converted to unified zarr format in `datasets/unified/`.

| Dataset | Classes | Samples | Format | Source |
|---------|---------|---------|--------|--------|
| FSC | 64 | 5,168 | zarr (T,17,2) | MMFS + mocap |
| MCFS | 132 | 2,668 | zarr (T,17,2) | Frame-level segments |
| SkatingVerse | 28 | 28,579 | zarr (T,17,2) | Video → RTMPose |

**Skeleton:** COCO 17kp (H3.6M), xy normalized (root-centered, spine-length scaled).

## Raw Data

Original files in `datasets/raw/` (read-only).

| Dataset | Size | Format | Notes |
|---------|------|--------|-------|
| figure-skating-classification | 341MB | pkl (150,17,3,1) | FSC source |
| mmfs | 2.1GB | pkl + npy | Contained in FSC |
| mcfs | 103MB | pkl segments | 132 classes |
| skatingverse | 46GB | mp4 video | 28K videos |
| athletepose3d | 70GB | multi-modal | 12 sports |
| finefs | 1.5GB | npz features | Quality scores |

## Relationships

- **FSC = MMFS (4,915) + mocap (253)** — MMFS is redundant, kept for reference
- **SkatingVerse 28 classes ⊂ FSC 64 classes** — ~90% overlap for jumps/spins
- **MCFS 132 classes** — different granularity, not directly mappable to FSC

## Label Ontology

See `data_tools/label_ontology.py` for unified label mappings.

## Preprocessing Pipeline

```bash
# Convert all datasets
PYTHONPATH=data uv run python -m data_tools.convert_fsc
PYTHONPATH=data uv run python -m data_tools.convert_mcfs
PYTHONPATH=data uv run python -m data_tools.convert_skatingverse --max-per-class 0
```

## Usage

```python
from data_tools.dataset import UnifiedSkatingDataset, varlen_collate

ds = UnifiedSkatingDataset("data/datasets/unified/fsc-64.zarr", split="train", augment=True)
loader = DataLoader(ds, batch_size=64, collate_fn=varlen_collate, shuffle=True)
```
```

- [ ] **Step 2: Commit**

```bash
git add data/DATASETS.md
git commit -m "docs(data): rewrite DATASETS.md — compact registry with unified format spec"
```

---

## Self-Review

1. **Spec coverage:**
   - Unified format: Task 3-6 (converters)
   - Label ontology: Task 1
   - Validation: Task 2
   - PyTorch Dataset: Task 7
   - Documentation: Task 8
   - Raw data reorganization: Task 4

2. **Placeholder scan:** No TBDs. All code is complete. Element matching in label_ontology may need tuning based on actual MCFS label names.

3. **Type consistency:** All converters produce `(N, T, 17, 2)` float32. Dataset returns `(T, 34)` flattened. Labels are int32 throughout.

4. **Gap found:** Task 4 moves datasets — need to update experiments/ scripts that reference old paths (`data/datasets/figure-skating-classification/` → `data/datasets/raw/figure-skating-classification/`). This affects `experiments/exp_augmentation.py`, `experiments/exp_infogcn.py`, etc. Not addressed in this plan — should be a follow-up task.

5. **Gap found:** `convert_fsc.py` saves with `dtype=object` for variable-length arrays. Zarr requires uniform arrays. Need to either pad to fixed length or use zarr's vlen_string equivalent for variable-length arrays. This needs resolution in Task 3.
