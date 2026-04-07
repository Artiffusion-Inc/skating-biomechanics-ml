"""Tests for FootTrackNet wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest

if TYPE_CHECKING:
    from src.ml.foot_tracker import FootTracker


class TestFootTracker:
    """Tests for person and foot detection."""

    @staticmethod
    def _make_tracker(mock_session, input_name: str = "images") -> FootTracker:
        """Create a FootTracker bypassing __init__ with required attributes set."""
        from src.ml.foot_tracker import FootTracker

        tracker = FootTracker.__new__(FootTracker)
        tracker._session = mock_session
        tracker._input_name = input_name
        tracker._input_size = (640, 480)
        return tracker

    def test_detect_returns_detections(self):
        """detect() returns list of detection dicts."""
        mock_session = mock.MagicMock()
        # Two detections: person (class 0) and foot (class 1)
        mock_session.run.return_value = [
            np.array(
                [
                    [100, 100, 300, 400, 0.9, 0],
                    [200, 300, 250, 450, 0.8, 1],
                ],
                dtype=np.float32,
            )
        ]

        tracker = self._make_tracker(mock_session)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = tracker.detect(frame)

        assert len(detections) == 2
        assert detections[0]["class_id"] == 0  # person
        assert detections[1]["class_id"] == 1  # foot
        assert detections[0]["confidence"] == pytest.approx(0.9)
        assert detections[1]["confidence"] == pytest.approx(0.8)

    def test_detect_filters_low_confidence(self):
        """detect() filters out detections below 0.3 confidence."""
        mock_session = mock.MagicMock()
        mock_session.run.return_value = [
            np.array(
                [
                    [100, 100, 300, 400, 0.9, 0],  # above threshold
                    [200, 300, 250, 450, 0.2, 1],  # below threshold
                ],
                dtype=np.float32,
            )
        ]

        tracker = self._make_tracker(mock_session)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = tracker.detect(frame)

        assert len(detections) == 1
        assert detections[0]["confidence"] == pytest.approx(0.9)

    def test_detect_scales_bbox_to_original_size(self):
        """Bbox coordinates are scaled from model input size to original frame size."""
        mock_session = mock.MagicMock()
        # Detection at (100, 100, 300, 400) in 640x480 model input
        mock_session.run.return_value = [np.array([[100, 100, 300, 400, 0.9, 0]], dtype=np.float32)]

        tracker = self._make_tracker(mock_session)

        # Original frame is 1280x960 (2x model input)
        frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        detections = tracker.detect(frame)

        assert len(detections) == 1
        bbox = detections[0]["bbox"]
        # 640x480 -> 1280x960 is 2x scale
        assert bbox[0] == 200  # 100 * 2
        assert bbox[1] == 200  # 100 * 2
        assert bbox[2] == 600  # 300 * 2
        assert bbox[3] == 800  # 400 * 2

    def test_detect_empty_output(self):
        """detect() returns empty list when no detections."""
        mock_session = mock.MagicMock()
        mock_session.run.return_value = [np.zeros((0, 6), dtype=np.float32)]

        tracker = self._make_tracker(mock_session)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = tracker.detect(frame)

        assert detections == []

    def test_init_from_registry(self):
        """FootTracker loads from ModelRegistry."""
        from src.ml.foot_tracker import FootTracker
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry(device="cpu")
        reg.register("foot_tracker", vram_mb=30, path="/tmp/foot_tracker.onnx")

        mock_session = mock.MagicMock()
        mock_session.get_input_details.return_value = [
            {"name": "images", "shape": [1, 3, 480, 640], "type": "float32"}
        ]

        with mock.patch("src.ml.model_registry.ort.InferenceSession", return_value=mock_session):
            tracker = FootTracker(reg)
            assert tracker._session is mock_session
            assert tracker._input_name == "images"
            assert tracker._input_size == (640, 480)
