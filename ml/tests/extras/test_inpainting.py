"""Tests for LAMA image inpainting wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.extras.inpainting import INPUT_SIZE, ImageInpainter


def _fake_resize(img, size, **_kwargs):
    """Return a zero array with the target shape."""
    h, w = size[1], size[0]
    if img.ndim == 3:
        return np.zeros((h, w, img.shape[2]), dtype=img.dtype)
    return np.zeros((h, w), dtype=img.dtype)


def _fake_cvt_color(img, _code):
    """Passthrough cvtColor mock."""
    return img


def _make_inpainter():
    """Build an ImageInpainter with a fully mocked ONNX session."""
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = []
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_session.get_outputs.return_value = [mock_output]
    # ONNX output is NCHW float32
    mock_session.run.return_value = [
        np.ones((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32) * 0.5
    ]

    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_session
    return ImageInpainter(mock_registry), mock_session, mock_registry


class TestImageInpainter:
    """Tests for ImageInpainter.inpaint."""

    def test_inpaint_normal_case(self):
        """Inpainting with a non-empty mask should call the model and composite."""
        inpainter, mock_session, mock_registry = _make_inpainter()

        frame = np.ones((100, 80, 3), dtype=np.uint8) * 128
        mask = np.zeros((100, 80), dtype=bool)
        mask[40:60, 30:50] = True

        with patch("src.extras.inpainting.cv2.resize", side_effect=_fake_resize):
            with patch("src.extras.inpainting.cv2.cvtColor", side_effect=_fake_cvt_color):
                result = inpainter.inpaint(frame, mask)

        mock_registry.get.assert_called_once_with("lama")
        mock_session.run.assert_called_once()
        assert result.shape == (100, 80, 3)
        assert result.dtype == np.uint8
        # Masked region should come from model output (zeros after fake resize)
        np.testing.assert_array_equal(result[40:60, 30:50], 0)
        # Unmasked region should stay original
        np.testing.assert_array_equal(result[0:10, 0:10], 128)

    def test_inpaint_zero_mask_noop(self):
        """Inpainting with an all-False mask should return the original frame."""
        inpainter, _mock_session, _mock_registry = _make_inpainter()

        frame = np.ones((64, 64, 3), dtype=np.uint8) * 200
        mask = np.zeros((64, 64), dtype=bool)

        with patch("src.extras.inpainting.cv2.resize", side_effect=_fake_resize):
            with patch("src.extras.inpainting.cv2.cvtColor", side_effect=_fake_cvt_color):
                result = inpainter.inpaint(frame, mask)

        np.testing.assert_array_equal(result, frame)

    def test_inpaint_full_mask(self):
        """Inpainting with an all-True mask should use model output everywhere."""
        inpainter, _mock_session, _mock_registry = _make_inpainter()

        frame = np.ones((64, 64, 3), dtype=np.uint8) * 200
        mask = np.ones((64, 64), dtype=bool)

        with patch("src.extras.inpainting.cv2.resize", side_effect=_fake_resize):
            with patch("src.extras.inpainting.cv2.cvtColor", side_effect=_fake_cvt_color):
                result = inpainter.inpaint(frame, mask)

        # Model output is 0.5 -> 128 after clip * 255 -> but fake resize returns zeros
        np.testing.assert_array_equal(result, 0)
