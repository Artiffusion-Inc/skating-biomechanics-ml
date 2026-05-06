import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import pytest

from src.pose_estimation._target_selector import TargetSelector


def test_click_selection_within_window():
    sel = TargetSelector(click_norm=(0.5, 0.5), click_lock_window=6)
    poses = np.zeros((2, 17, 3), dtype=np.float32)
    poses[0, 4, :2] = [0.5, 0.5]  # mid-hip approx
    poses[1, 4, :2] = [0.1, 0.1]
    track_ids = [0, 1]
    result = sel.select_target(poses, track_ids, frame_idx=3)
    assert result == 0


def test_click_selection_outside_window():
    sel = TargetSelector(click_norm=(0.5, 0.5), click_lock_window=6)
    poses = np.zeros((1, 17, 3), dtype=np.float32)
    poses[0, 4, :2] = [0.5, 0.5]
    track_ids = [0]
    result = sel.select_target(poses, track_ids, frame_idx=10)
    assert result is None


def test_auto_select_by_hits():
    sel = TargetSelector()
    hit_counts = {0: 5, 1: 10}
    assert sel.auto_select_by_hits(hit_counts) == 1
