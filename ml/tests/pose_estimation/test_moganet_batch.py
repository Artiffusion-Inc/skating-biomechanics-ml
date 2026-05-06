import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

"""Tests for MogaNet batch inference module."""

import numpy as np
import pytest

from src.pose_estimation.moganet_batch import (
    MOGANET_INPUT_SIZE,
    MogaNetBatch,
    decode_heatmaps,
    preprocess_crops,
    rescale_keypoints,
)


class TestPreprocessCrops:
    def test_output_shape(self):
        """Verify output is (B, 3, 288, 384) float32."""
        crops = [np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8) for _ in range(4)]
        tensor = preprocess_crops(crops)
        assert tensor.shape == (4, 3, MOGANET_INPUT_SIZE[1], MOGANET_INPUT_SIZE[0])
        assert tensor.dtype == np.float32
        assert len(tensor) == 4

    def test_single_crop(self):
        """Single crop yields batch_size=1."""
        crop = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        tensor = preprocess_crops([crop])
        assert tensor.shape[0] == 1

    def test_bgr_to_rgb(self):
        """BGR blue channel becomes RGB blue (channel index 2 in CHW)."""
        crop = np.zeros((50, 50, 3), dtype=np.uint8)
        crop[:, :, 0] = 255  # Blue channel in BGR
        tensor = preprocess_crops([crop])
        # After BGR->RGB, the original BGR blue ends up in RGB blue (CHW channel 2)
        # The crop is scaled and centered: 50x50 -> scale=5.76 -> 288x288, pad_left=48
        # Crop content region in CHW: channels=all, height=0:288, width=48:336
        # Pixel (0, 0) in content = tensor[0, 2, 0, 48] should be RGB blue (value ~2.64)
        assert tensor[0, 2, 0, 48] > 2.0, f"Expected >2.0, got {tensor[0, 2, 0, 48]}"

    def test_normalization_applied(self):
        """Verify ImageNet mean/std normalization is applied."""
        crop = np.full((50, 50, 3), 128, dtype=np.uint8)
        tensor = preprocess_crops([crop])
        # 50x50 -> scale=5.76 -> new_h=288, new_w=288, pad_left=(384-288)/2=48
        # Content pixel at tensor[0, 0, 0, 48] should be normalized
        # (128/255 - 0.485) / 0.229
        expected = ((128.0 / 255.0) - 0.485) / 0.229
        assert tensor[0, 0, 0, 48] == pytest.approx(expected, abs=0.02)

    def test_letterbox_preserves_aspect_ratio(self):
        """Tall crop should be letterboxed (padded left/right)."""
        # Very tall crop: 400x100 (HxW), so aspect ratio constrained by height
        crop = np.random.randint(0, 255, (400, 100, 3), dtype=np.uint8)
        tensor = preprocess_crops([crop])
        # scale = min(384/100, 288/400) = min(3.84, 0.72) = 0.72
        # new_h = 400 * 0.72 = 288, new_w = 100 * 0.72 = 72
        # pad_left = (384 - 72) / 2 = 156
        # Check that left padding area is normalized 0-value
        # 0 normalized in R: (0 - 0.485) / 0.229 approx -2.12
        left_pad = tensor[0, 0, :, :156].mean()
        right_start = 384 - 156
        right_pad = tensor[0, 0, :, right_start:].mean()
        assert left_pad == pytest.approx(-2.12, abs=0.1)
        assert right_pad == pytest.approx(-2.12, abs=0.1)

    def test_empty_crops(self):
        """Empty list returns empty tensor."""
        tensor = preprocess_crops([])
        assert tensor.shape == (0, 3, MOGANET_INPUT_SIZE[1], MOGANET_INPUT_SIZE[0])


class TestDecodeHeatmaps:
    def test_single_peak_per_joint(self):
        """Single clear peak per joint, verify correct coordinate scaling."""
        batch_size = 2
        num_joints = 17
        heatmap_h, heatmap_w = 72, 96

        heatmaps = np.zeros((batch_size, num_joints, heatmap_h, heatmap_w), dtype=np.float32)
        for b in range(batch_size):
            for j in range(num_joints):
                y_peak = (j * 4 + 5) % heatmap_h
                x_peak = (j * 7 + 3) % heatmap_w
                heatmaps[b, j, y_peak, x_peak] = 1.0

        keypoints, scores = decode_heatmaps(heatmaps)

        assert keypoints.shape == (batch_size, num_joints, 2)
        assert scores.shape == (batch_size, num_joints)

        for b in range(batch_size):
            for j in range(num_joints):
                y_peak = (j * 4 + 5) % heatmap_h
                x_peak = (j * 7 + 3) % heatmap_w
                # Scale from heatmap to model input space
                expected_x = x_peak * MOGANET_INPUT_SIZE[0] / heatmap_w
                expected_y = y_peak * MOGANET_INPUT_SIZE[1] / heatmap_h
                assert keypoints[b, j, 0] == pytest.approx(expected_x, abs=0.5)
                assert keypoints[b, j, 1] == pytest.approx(expected_y, abs=0.5)
                assert scores[b, j] == pytest.approx(1.0)

    def test_no_peaks(self):
        """All-zero heatmaps produce near-zero scores."""
        heatmaps = np.zeros((1, 17, 72, 96), dtype=np.float32)
        _keypoints, scores = decode_heatmaps(heatmaps)
        assert np.all(scores < 0.01)


class TestRescaleKeypoints:
    def test_no_letterbox_no_offset(self):
        """Crop exactly matches input size, bbox at origin: no change."""
        crop = np.random.randint(0, 255, (288, 384, 3), dtype=np.uint8)
        keypoints = np.full((1, 17, 2), 100.0, dtype=np.float32)
        keypoints[0, 0] = [100.0, 200.0]
        bboxes = [(0, 0, 384, 288)]
        rescaled = rescale_keypoints(keypoints, [crop], bboxes)
        assert rescaled.shape == (1, 17, 2)
        # Scale = 1.0, no padding, bbox at origin -> same coords
        assert rescaled[0, 0, 0] == pytest.approx(100.0)
        assert rescaled[0, 0, 1] == pytest.approx(200.0)

    def test_with_bbox_offset(self):
        """Bbox origin offset is added to keypoints."""
        crop = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        keypoints = np.array([[50.0, 75.0]] * 17, dtype=np.float32).reshape(1, 17, 2)
        bboxes = [(100, 50, 150, 200)]

        # Compute expected rescaling
        input_w, input_h = 384, 288
        crop_h, crop_w = 200, 150
        scale = min(input_w / crop_w, input_h / crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        pad_left = (input_w - new_w) / 2
        pad_top = (input_h - new_h) / 2

        expected_x = (50.0 - pad_left) / scale + 100.0
        expected_y = (75.0 - pad_top) / scale + 50.0

        rescaled = rescale_keypoints(keypoints, [crop], bboxes)
        assert rescaled[0, 0, 0] == pytest.approx(expected_x, abs=0.5)
        assert rescaled[0, 0, 1] == pytest.approx(expected_y, abs=0.5)

    def test_multiple_crops(self):
        """Different crops and bboxes handled correctly."""
        crops = [
            np.random.randint(0, 255, (288, 384, 3), dtype=np.uint8),
            np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8),
        ]
        keypoints = np.zeros((2, 17, 2), dtype=np.float32)
        keypoints[0, :] = [100.0, 150.0]
        keypoints[1, :] = [50.0, 75.0]
        bboxes = [(0, 0, 384, 288), (100, 50, 150, 200)]

        rescaled = rescale_keypoints(keypoints, crops, bboxes)

        # First crop: no letterbox, no offset
        assert rescaled[0, 0, 0] == pytest.approx(100.0)
        assert rescaled[0, 0, 1] == pytest.approx(150.0)

        # Second crop: letterbox + offset
        crop_h, crop_w = 200, 150
        scale = min(384 / crop_w, 288 / crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        pad_left = (384 - new_w) / 2
        pad_top = (288 - new_h) / 2
        expected_x = (50.0 - pad_left) / scale + 100.0
        expected_y = (75.0 - pad_top) / scale + 50.0
        assert rescaled[1, 0, 0] == pytest.approx(expected_x, abs=0.5)
        assert rescaled[1, 0, 1] == pytest.approx(expected_y, abs=0.5)


class TestMogaNetBatchInit:
    def test_init_without_model_raises(self):
        """Missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MogaNetBatch(model_path="/nonexistent/model.onnx", device="cpu")
