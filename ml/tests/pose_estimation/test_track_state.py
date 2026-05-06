import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import pytest

from src.pose_estimation._track_state import TrackState


def test_track_state_init_custom_tracker():
    ts = TrackState(fps=30.0, tracking_backend="custom")
    assert ts.custom_tracker is not None
    assert ts.sports2d_tracker is None
    assert ts.deepsort_tracker is None


def test_track_state_init_sports2d():
    ts = TrackState(fps=30.0, tracking_mode="sports2d")
    assert ts.sports2d_tracker is not None
    assert ts.custom_tracker is None


def test_record_frame_data():
    ts = TrackState(fps=30.0)
    poses = np.random.rand(2, 17, 3).astype(np.float32)
    track_ids = [0, 1]
    ts.record_frame(5, poses, track_ids)
    assert ts.track_hit_counts[0] == 1
    assert ts.track_hit_counts[1] == 1
    assert 5 in ts.frame_track_data
    assert ts.frame_track_data[5][0].shape == (17, 3)


def test_auto_select_by_hits():
    ts = TrackState(fps=30.0)
    ts.track_hit_counts = {0: 5, 1: 10, 2: 3}
    selected = ts.auto_select_target()
    assert selected == 1
