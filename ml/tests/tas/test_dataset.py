"""Tests for TAS dataset loader.

Note: Tests use direct file loading to avoid import chain issues with cv2/numba.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# Load h36m module directly without triggering __init__.py chain
H36M_PATH = Path(__file__).parent.parent.parent / "src" / "pose_estimation" / "h36m.py"
h36m_spec = importlib.util.spec_from_file_location("h36m", H36M_PATH)
h36m_mod = importlib.util.module_from_spec(h36m_spec)
h36m_spec.loader.exec_module(h36m_mod)
sys.modules["h36m"] = h36m_mod

# Build fake package chain for relative imports
sys.modules["types"] = __import__("types")
ml_mod = types.ModuleType("ml")
ml_src_mod = types.ModuleType("ml.src")
ml_src_pose = types.ModuleType("ml.src.pose_estimation")
ml_src_tas = types.ModuleType("ml.src.tas")
sys.modules["ml"] = ml_mod
sys.modules["ml.src"] = ml_src_mod
sys.modules["ml.src.pose_estimation"] = ml_src_pose
sys.modules["ml.src.tas"] = ml_src_tas
sys.modules["ml.src.pose_estimation.h36m"] = h36m_mod
ml_src_pose.h36m = h36m_mod

# Now load dataset module
data_spec = importlib.util.spec_from_file_location(
    "ml.src.tas.dataset", Path(__file__).parent.parent.parent / "src" / "tas" / "dataset.py"
)
data_mod = importlib.util.module_from_spec(data_spec)
sys.modules["ml.src.tas.dataset"] = data_mod
data_spec.loader.exec_module(data_mod)

# Extract functions
coarse_label = data_mod.coarse_label
op25_to_coco17 = data_mod.op25_to_coco17
normalize_poses = data_mod.normalize_poses
MCFSCoarseDataset = data_mod.MCFSCoarseDataset
pad_collate = data_mod.pad_collate

DATA_DIR = Path("data/datasets/mcfs")


def test_coarse_label():
    assert coarse_label("NONE") == 0
    assert coarse_label("3Flip") == 1
    assert coarse_label("3Axel") == 1
    assert coarse_label("3Salchow") == 1
    assert coarse_label("3Toeloop") == 1
    assert coarse_label("3Lutz") == 1
    assert coarse_label("3Loop") == 1
    assert coarse_label("ChComboSpin4") == 2
    assert coarse_label("FlyCamelSpin4") == 2
    assert coarse_label("StepSequence4") == 3
    assert coarse_label("ChoreoSequence1") == 3
    assert coarse_label("UnknownElement") == 0


def test_op25_to_coco17():
    op25 = np.random.randn(10, 25, 3).astype(np.float64)
    coco17 = op25_to_coco17(op25)
    assert coco17.shape == (10, 17, 2)
    assert coco17.dtype == np.float32


def test_normalize_poses():
    poses = np.random.randn(10, 17, 2).astype(np.float32)
    normalized = normalize_poses(poses)
    assert normalized.shape == (10, 17, 2)
    # Mid-hip should be at origin
    mid = normalized[:, 11:13, :].mean(axis=1)
    np.testing.assert_allclose(mid, 0, atol=1e-5)


def test_mcfs_dataset_exists():
    ds = MCFSCoarseDataset(DATA_DIR / "features", DATA_DIR / "groundTruth")
    assert len(ds) > 0
    poses, labels, length = ds[0]
    assert poses.shape == (length, 17, 2)
    assert labels.shape == (length,)
    assert set(labels.tolist()).issubset({0, 1, 2, 3})
    assert poses.dtype == np.float32
    assert labels.dtype == np.int64


def test_pad_collate():
    import torch

    batch = [
        (np.random.randn(50, 17, 2).astype(np.float32), np.array([0, 1] * 25, dtype=np.int64), 50),
        (np.random.randn(30, 17, 2).astype(np.float32), np.array([0, 2] * 15, dtype=np.int64), 30),
    ]
    poses, labels, lengths = pad_collate(batch)
    assert poses.shape == (2, 50, 17, 2)
    assert labels.shape == (2, 50)
    assert labels[0, 49] != -1  # First sample fills all 50
    assert labels[1, 49] == -1  # Second sample padded
    assert lengths.tolist() == [50, 30]


if __name__ == "__main__":
    test_coarse_label()
    print("✓ coarse_label OK")
    test_op25_to_coco17()
    print("✓ op25_to_coco17 OK")
    test_normalize_poses()
    print("✓ normalize_poses OK")
    test_mcfs_dataset_exists()
    print("✓ mcfs_dataset OK")
    test_pad_collate()
    print("✓ pad_collate OK")
    print("ALL TESTS PASSED")
