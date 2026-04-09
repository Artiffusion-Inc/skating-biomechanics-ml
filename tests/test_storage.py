"""Tests for storage abstraction layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.storage import delete_object, download_file, upload_file


@patch("src.storage._client")
def test_upload_file_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_client.return_value = mock_s3
    with patch("src.storage.get_settings"):
        upload_file("/tmp/test.mp4", "input/test.mp4")
    mock_s3.upload_file.assert_called_once()


@patch("src.storage._client")
def test_download_file_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_client.return_value = mock_s3
    with patch("src.storage.get_settings"):
        download_file("output/test.mp4", "/tmp/result.mp4")
    mock_s3.download_file.assert_called_once()


@patch("src.storage._client")
def test_delete_object_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_client.return_value = mock_s3
    with patch("src.storage.get_settings"):
        delete_object("input/test.mp4")
    mock_s3.delete_object.assert_called_once()
