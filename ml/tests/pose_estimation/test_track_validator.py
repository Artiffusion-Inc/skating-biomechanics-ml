import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pytest

from src.pose_estimation._track_validator import TrackValidator


def test_no_steal_when_pose_unchanged():
    val = TrackValidator()
    pose = np.ones((17, 3), dtype=np.float32) * 0.5
    pose[:, 2] = 0.9
    assert not val.is_stolen(pose, pose)


def test_steal_on_large_jump_and_anomaly():
    val = TrackValidator()
    prev = np.ones((17, 3), dtype=np.float32) * 0.5
    prev[:, 2] = 0.9
    curr = prev.copy()
    curr[:, 0] += 0.5  # big jump
    curr[0, 1] += 0.3  # distort ratios
    # Need ratios for anomaly detection to trigger
    from src.tracking.skeletal_identity import compute_2d_skeletal_ratios

    prev_ratios = compute_2d_skeletal_ratios(prev)
    assert val.is_stolen(curr, prev, prev_ratios)


def test_migration_score_weights():
    val = TrackValidator()
    prev = np.ones((17, 3), dtype=np.float32) * 0.5
    prev[:, 2] = 0.9
    curr = prev.copy()
    curr[:, 0] += 0.05  # small jump
    score = val.migration_score(curr, prev, elapsed=0)
    assert score < 1.5
