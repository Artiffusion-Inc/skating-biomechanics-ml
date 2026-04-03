"""Tests for skeleton drawer Sports2D-style updates."""

import numpy as np
import pytest

from src.visualization.skeleton.drawer import draw_skeleton


def _frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestSports2DBoneColors:
    def test_left_bone_is_green(self):
        """Left-side bones should use Sports2D green color."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[4] = [0.5, 0.4]  # LHIP (normalized)
        pose[5] = [0.5, 0.6]  # LKNEE (normalized)
        frame = _frame()
        draw_skeleton(frame, pose, 480, 640, confidence_threshold=0.0, line_width=10)
        # Center of the line - should be green (0, 255, 0) in BGR
        center = frame[240, 320]
        assert center[1] > 200, f"Expected green, got BGR={center}"  # Green channel high

    def test_right_bone_is_orange(self):
        """Right-side bones should use Sports2D orange color."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[1] = [0.55, 0.4]  # RHIP (normalized)
        pose[2] = [0.55, 0.6]  # RKNEE (normalized)
        frame = _frame()
        draw_skeleton(frame, pose, 480, 640, confidence_threshold=0.0, line_width=10)
        # Center of the line - should be orange (0, 128, 255) in BGR
        center = frame[240, 352]
        # Orange has high red channel (255 in BGR[2])
        assert center[2] > 200, f"Expected orange (high red), got BGR={center}"
