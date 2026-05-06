import numpy as np
import pytest

from src.pose_estimation._frame_processor import FrameProcessor


def test_convert_empty():
    fp = FrameProcessor(output_format="normalized")
    result = fp.convert_keypoints(np.zeros((0, 17, 2)), np.zeros((0, 17)), 640, 480)
    assert result.shape == (0, 17, 3)


def test_convert_single_person():
    fp = FrameProcessor(output_format="normalized")
    kps = np.array([[[100.0, 200.0]] * 17], dtype=np.float32)
    scores = np.ones((1, 17), dtype=np.float32) * 0.8
    result = fp.convert_keypoints(kps, scores, 640, 480)
    assert result.shape == (1, 17, 3)
    assert result[0, 0, 0] == pytest.approx(100 / 640, abs=1e-4)
    assert result[0, 0, 1] == pytest.approx(200 / 480, abs=1e-4)
