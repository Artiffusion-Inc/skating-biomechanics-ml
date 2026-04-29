"""Tests for RobustVideoMatting wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.extras.video_matting import NUM_RECURRENT, VideoMatting


def _fake_cvt_color(img, _code):
    """Passthrough cvtColor mock."""
    return img


def _fake_resize(img, size, **_kwargs):
    """Return a ones array with the target shape."""
    h, w = size[1], size[0]
    if img.ndim == 2:
        return np.ones((h, w), dtype=img.dtype)
    return np.ones((h, w, img.shape[2]), dtype=img.dtype)


def _make_matting(pha_shape=(1, 1, 128, 128)):
    """Build a VideoMatting instance with a fully mocked ONNX session."""
    input_names = ["src", "downsample_ratio", "r1i", "r2i", "r3i", "r4i"]
    mock_inputs = [MagicMock(name=n) for n in input_names]
    for m, n in zip(mock_inputs, input_names, strict=True):
        m.name = n

    output_names = ["pha", "r1o", "r2o", "r3o", "r4o"]
    mock_outputs = [MagicMock(name=n) for n in output_names]
    for m, n in zip(mock_outputs, output_names, strict=True):
        m.name = n

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = mock_inputs
    mock_session.get_outputs.return_value = mock_outputs

    pha = np.ones(pha_shape, dtype=np.float32) * 0.8
    rec = np.zeros((1, 1, 1, 1), dtype=np.float32)
    mock_session.run.return_value = [pha, rec, rec, rec, rec]

    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_session

    return VideoMatting(mock_registry), mock_session, mock_registry


class TestVideoMatting:
    """Tests for VideoMatting.matting."""

    def test_matting_normal_case(self):
        """Normal matting call should return alpha matte in [0, 1]."""
        matting, mock_session, mock_registry = _make_matting()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        with patch("src.extras.video_matting.cv2.cvtColor", side_effect=_fake_cvt_color):
            with patch("src.extras.video_matting.cv2.resize", side_effect=_fake_resize):
                result = matting.matting(frame)

        mock_registry.get.assert_called_once_with("video_matting")
        mock_session.run.assert_called_once()
        assert result.shape == (480, 640)
        assert result.dtype == np.float32
        assert np.all((result >= 0) & (result <= 1))
        # Because fake_resize returns ones and clip keeps them at 1
        np.testing.assert_allclose(result, 1.0)

    def test_matting_downsample_ratio_passed(self):
        """_downsample_ratio should be forwarded to the ONNX session."""
        matting, mock_session, _mock_registry = _make_matting(pha_shape=(1, 1, 480, 640))

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        with patch("src.extras.video_matting.cv2.cvtColor", side_effect=_fake_cvt_color):
            result = matting.matting(frame)

        assert result.shape == (480, 640)
        call_args = mock_session.run.call_args
        inputs_dict = call_args[0][1]
        assert "downsample_ratio" in inputs_dict
        np.testing.assert_array_equal(
            inputs_dict["downsample_ratio"],
            np.array([0.25], dtype=np.float32),
        )

    def test_matting_recurrent_state_update(self):
        """Recurrent states should be updated after each call."""
        matting, _mock_session, _mock_registry = _make_matting()

        assert len(matting._rec_states) == 0

        frame = np.ones((240, 320, 3), dtype=np.uint8)
        with patch("src.extras.video_matting.cv2.cvtColor", side_effect=_fake_cvt_color):
            with patch("src.extras.video_matting.cv2.resize", side_effect=_fake_resize):
                matting.matting(frame)

        assert len(matting._rec_states) == NUM_RECURRENT
        for state in matting._rec_states:
            assert isinstance(state, np.ndarray)

    def test_matting_reset(self):
        """reset() should clear recurrent states."""
        matting, _mock_session, _mock_registry = _make_matting()

        frame = np.ones((240, 320, 3), dtype=np.uint8)
        with patch("src.extras.video_matting.cv2.cvtColor", side_effect=_fake_cvt_color):
            with patch("src.extras.video_matting.cv2.resize", side_effect=_fake_resize):
                matting.matting(frame)

        assert len(matting._rec_states) == NUM_RECURRENT
        matting.reset()
        assert len(matting._rec_states) == 0

    def test_matting_with_mask_parameter(self):
        """Mask parameter is accepted even though current implementation ignores it."""
        matting, mock_session, _mock_registry = _make_matting(pha_shape=(1, 1, 480, 640))

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        mask = np.ones((480, 640), dtype=np.uint8)

        with patch("src.extras.video_matting.cv2.cvtColor", side_effect=_fake_cvt_color):
            result = matting.matting(frame, mask=mask)

        assert result.shape == (480, 640)
        mock_session.run.assert_called_once()
