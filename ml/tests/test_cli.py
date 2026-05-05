import argparse
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_scripts_dir = str(Path(__file__).parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from cli import (
    CREDS_PATH,
    CLIError,
    _ensure_access_token,
    _load_credentials,
    _save_credentials,
    analyze,
    login,
    upload_video,
)


@pytest.fixture(autouse=True)
def clean_creds():
    if CREDS_PATH.exists():
        CREDS_PATH.unlink()
    yield
    if CREDS_PATH.exists():
        CREDS_PATH.unlink()


class TestCredentials:
    def test_save_and_load(self):
        creds = {"access_token": "tok123", "refresh_token": "ref456"}
        _save_credentials(creds)
        loaded = _load_credentials()
        assert loaded == creds
        assert CREDS_PATH.stat().st_mode & 0o777 == 0o600

    def test_load_missing(self):
        if CREDS_PATH.exists():
            CREDS_PATH.unlink()
        assert _load_credentials() is None


class TestEnsureAccessToken:
    async def test_valid_token(self):
        _save_credentials({"access_token": "valid", "refresh_token": "ref"})
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
        token = await _ensure_access_token(mock_client)
        assert token == "valid"

    async def test_no_creds(self):
        if CREDS_PATH.exists():
            CREDS_PATH.unlink()
        mock_client = AsyncMock()
        with pytest.raises(CLIError) as exc_info:
            await _ensure_access_token(mock_client)
        assert exc_info.value.code == 2


class TestLogin:
    @patch("builtins.input", return_value="test@example.com")
    @patch("getpass.getpass", return_value="secret")
    @patch("httpx.AsyncClient")
    async def test_login_success(self, mock_client_cls, mock_getpass, mock_input):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "tok", "refresh_token": "ref"}
        mock_client_cls.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_resp
        )

        await login(argparse.Namespace())

        creds = _load_credentials()
        assert creds["access_token"] == "tok"
        assert creds["refresh_token"] == "ref"

    @patch("builtins.input", return_value="test@example.com")
    @patch("getpass.getpass", return_value="wrong")
    @patch("httpx.AsyncClient")
    async def test_login_failure(self, mock_client_cls, mock_getpass, mock_input):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"message": "Invalid credentials"}
        mock_client_cls.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_resp
        )

        with pytest.raises(CLIError) as exc_info:
            await login(argparse.Namespace())
        assert exc_info.value.code == 1


class TestUploadVideo:
    async def test_upload_success(self):
        mock_init = MagicMock()
        mock_init.status_code = 200
        mock_init.json.return_value = {
            "upload_id": "uid123",
            "key": "uploads/u/123/v.mp4",
            "chunk_size": 5242880,
            "parts": [
                {"part_number": 1, "url": "https://r2.example.com/part1"},
            ],
        }

        mock_put = MagicMock()
        mock_put.status_code = 200
        mock_put.headers = {"etag": '"etag123"'}

        mock_complete = MagicMock()
        mock_complete.status_code = 200
        mock_complete.json.return_value = {
            "status": "completed",
            "key": "uploads/u/123/v.mp4",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_init, mock_complete])
        mock_client.put = AsyncMock(return_value=mock_put)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            tmp_path = Path(f.name)
        try:
            key = await upload_video(mock_client, "tok", str(tmp_path))
            assert key == "uploads/u/123/v.mp4"
        finally:
            tmp_path.unlink(missing_ok=True)


class TestAnalyze:
    async def test_analyze_success(self):
        mock_queue = MagicMock()
        mock_queue.status_code = 200
        mock_queue.json.return_value = {"task_id": "task123"}

        mock_running = MagicMock()
        mock_running.status_code = 200
        mock_running.json.return_value = {
            "status": "running",
            "progress": 0.5,
            "message": "GPU...",
        }

        mock_completed = MagicMock()
        mock_completed.status_code = 200
        mock_completed.json.return_value = {
            "status": "completed",
            "progress": 1.0,
            "message": "Done",
            "result": {"video_path": "key123", "stats": {"fps": 30.0}},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_queue, mock_completed])
        mock_client.get = AsyncMock(side_effect=[mock_running, mock_completed])

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("cli._ensure_access_token", return_value="tok"):
            with patch("cli.upload_video", return_value="key123"):
                with patch("httpx.AsyncClient", mock_client_cls):
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                        f.write(b"video")
                        tmp = f.name
                    try:
                        args = argparse.Namespace(
                            video=tmp,
                            element="waltz_jump",
                            person_click=None,
                            frame_skip=1,
                            layer=3,
                            tracking="auto",
                            export=True,
                            session_id=None,
                            depth=False,
                            optical_flow=False,
                            segment=False,
                            foot_track=False,
                            matting=False,
                            inpainting=False,
                        )

                        captured = io.StringIO()
                        with patch("sys.stdout", new=captured):
                            await analyze(args)

                        output = captured.getvalue().strip()
                        result = json.loads(output)
                        assert result["video_path"] == "key123"
                    finally:
                        Path(tmp).unlink(missing_ok=True)
