"""Tests for NVENC auto-detection in H264Writer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.video_writer import H264Writer, _probe_nvenc


class TestProbeNvenc:
    """Tests for _probe_nvenc helper."""

    @patch("src.utils.video_writer.av.open")
    def test_returns_true_when_nvenc_available(self, mock_open: MagicMock) -> None:
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_container.add_stream.return_value = mock_stream
        mock_open.return_value = mock_container

        assert _probe_nvenc() is True

    @patch("src.utils.video_writer.av.open", side_effect=Exception("no nvenc"))
    def test_returns_false_when_nvenc_unavailable(self, mock_open: MagicMock) -> None:
        assert _probe_nvenc() is False

    @patch("src.utils.video_writer.av.open")
    def test_returns_false_when_add_stream_raises(self, mock_open: MagicMock) -> None:
        mock_container = MagicMock()
        mock_container.add_stream.side_effect = Exception("codec not found")
        mock_open.return_value = mock_container

        assert _probe_nvenc() is False


class TestH264WriterInit:
    """Tests for H264Writer codec selection."""

    @patch("src.utils.video_writer._probe_nvenc", return_value=True)
    @patch("src.utils.video_writer.av.open")
    def test_auto_selects_nvenc_when_available(
        self, mock_open: MagicMock, mock_probe: MagicMock
    ) -> None:
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_container.add_stream.return_value = mock_stream
        mock_open.return_value = mock_container

        H264Writer("/tmp/test.mp4", 1280, 720, 30.0)

        mock_probe.assert_called_once()
        mock_container.add_stream.assert_called_once()
        # The codec passed to add_stream should be "h264_nvenc"
        args, _kwargs = mock_container.add_stream.call_args
        assert args[0] == "h264_nvenc"

    @patch("src.utils.video_writer._probe_nvenc", return_value=False)
    @patch("src.utils.video_writer.av.open")
    def test_auto_selects_libx264_when_nvenc_unavailable(
        self, mock_open: MagicMock, mock_probe: MagicMock
    ) -> None:
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_container.add_stream.return_value = mock_stream
        mock_open.return_value = mock_container

        H264Writer("/tmp/test.mp4", 1280, 720, 30.0)

        mock_probe.assert_called_once()
        args, _kwargs = mock_container.add_stream.call_args
        assert args[0] == "libx264"

    @patch("src.utils.video_writer._probe_nvenc")
    @patch("src.utils.video_writer.av.open")
    def test_explicit_codec_skips_probe(self, mock_open: MagicMock, mock_probe: MagicMock) -> None:
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_container.add_stream.return_value = mock_stream
        mock_open.return_value = mock_container

        H264Writer("/tmp/test.mp4", 1280, 720, 30.0, codec="libx264")

        mock_probe.assert_not_called()
        args, _kwargs = mock_container.add_stream.call_args
        assert args[0] == "libx264"

    @patch("src.utils.video_writer._probe_nvenc", return_value=True)
    @patch("src.utils.video_writer.av.open")
    def test_nvenc_uses_constqp_options(self, mock_open: MagicMock, mock_probe: MagicMock) -> None:
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_container.add_stream.return_value = mock_stream
        mock_open.return_value = mock_container

        H264Writer("/tmp/test.mp4", 1280, 720, 30.0)

        assert mock_stream.options == {"preset": "p4", "rc": "constqp", "qp": "28"}

    @patch("src.utils.video_writer._probe_nvenc", return_value=False)
    @patch("src.utils.video_writer.av.open")
    def test_libx264_uses_preset_and_crf(self, mock_open: MagicMock, mock_probe: MagicMock) -> None:
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_container.add_stream.return_value = mock_stream
        mock_open.return_value = mock_container

        H264Writer("/tmp/test.mp4", 1280, 720, 30.0, preset="ultrafast", crf=18)

        assert mock_stream.options == {"preset": "ultrafast", "crf": "18"}


class TestH264WriterWrite:
    """Tests for H264Writer.write without cvtColor."""

    @patch("src.utils.video_writer._probe_nvenc", return_value=False)
    @patch("src.utils.video_writer.av.open")
    @patch("src.utils.video_writer.av.VideoFrame")
    def test_write_uses_bgr24_format(
        self, mock_frame_cls: MagicMock, mock_open: MagicMock, mock_probe: MagicMock
    ) -> None:
        import numpy as np

        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.encode.return_value = []
        mock_container.add_stream.return_value = mock_stream
        mock_open.return_value = mock_container
        mock_av_frame = MagicMock()
        mock_frame_cls.from_ndarray.return_value = mock_av_frame

        writer = H264Writer("/tmp/test.mp4", 1280, 720, 30.0)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        writer.write(frame)

        mock_frame_cls.from_ndarray.assert_called_once_with(frame, format="bgr24")
        mock_stream.encode.assert_called_once_with(mock_av_frame)
