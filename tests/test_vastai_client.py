"""Tests for Vast.ai client module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.vastai.client import VastResult, _get_worker_url


def test_get_worker_url_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"url": "https://worker-1.vast.ai:8000"}
    mock_resp.raise_for_status = MagicMock()

    with patch("src.vastai.client.httpx.post", return_value=mock_resp) as mock_post:
        url = _get_worker_url("skating-ml-gpu", "test-key")

    assert url == "https://worker-1.vast.ai:8000"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["endpoint"] == "skating-ml-gpu"


def test_vast_result_fields():
    r = VastResult(
        video_path="/tmp/out.mp4",
        poses_path="/tmp/out.npy",
        csv_path=None,
        stats={"frames": 100},
    )
    assert r.video_path == "/tmp/out.mp4"
    assert r.csv_path is None
    assert r.stats == {"frames": 100}
