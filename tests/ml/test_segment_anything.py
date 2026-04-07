"""Tests for SAM2 segmentation wrapper."""

from unittest import mock

import numpy as np


class TestSegmentAnything:
    """Tests for SAM2 image segmentation."""

    def test_segment_returns_mask(self):
        """segment() returns (H, W) bool mask."""
        from src.ml.segment_anything import SegmentAnything

        mock_session = mock.MagicMock()
        # SAM2 returns masks and scores
        mock_session.run.return_value = [
            np.ones((1, 1, 256, 256), dtype=np.float32),  # masks
            np.array([[0.95]]),  # scores (iou_predictions)
        ]

        est = SegmentAnything.__new__(SegmentAnything)
        est._session = mock_session
        est._input_size = 1024
        est._input_names = ["image", "point_coords", "point_labels"]

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mask = est.segment(frame, point=(320, 240))

        assert mask.shape == (480, 640)
        assert mask.dtype == bool

    def test_segment_with_no_point_returns_empty(self):
        """segment() with point=None returns None (no prompt)."""
        from src.ml.segment_anything import SegmentAnything

        est = SegmentAnything.__new__(SegmentAnything)
        result = est.segment(np.zeros((480, 640, 3), dtype=np.uint8), point=None)
        assert result is None

    def test_segment_resize_back_to_original(self):
        """Mask is resized to original frame size."""
        from src.ml.segment_anything import SegmentAnything

        mock_session = mock.MagicMock()
        mock_session.run.return_value = [
            np.ones((1, 1, 256, 256), dtype=np.float32),
            np.array([[0.95]]),
        ]

        est = SegmentAnything.__new__(SegmentAnything)
        est._session = mock_session
        est._input_size = 1024
        est._input_names = ["image", "point_coords", "point_labels"]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mask = est.segment(frame, point=(640, 360))

        assert mask.shape == (720, 1280)


class TestSegmentationLayer:
    """Tests for segmentation mask visualization."""

    def test_render_adds_mask_overlay(self):
        """SegmentationMaskLayer renders semi-transparent mask."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.segmentation_layer import SegmentationMaskLayer

        layer = SegmentationMaskLayer(opacity=0.3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:400, 200:500] = True

        ctx = LayerContext(frame_width=640, frame_height=480)
        ctx.custom_data["seg_mask"] = mask

        result = layer.render(frame, ctx)
        assert result.shape == (480, 640, 3)
        # Masked region should have color
        assert not np.all(result == 0)

    def test_render_no_mask_returns_unchanged(self):
        """No-op when no seg_mask in context."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.segmentation_layer import SegmentationMaskLayer

        layer = SegmentationMaskLayer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        ctx = LayerContext(frame_width=640, frame_height=480)

        result = layer.render(frame, ctx)
        assert np.array_equal(result, frame)
