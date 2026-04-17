"""Tests for R2 storage operations."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


class TestPresignedURL:
    """Tests for presigned URL generation."""

    def test_get_object_url_default_method(self):
        """Should generate GET presigned URL by default."""
        with patch("app.storage._client") as mock_client:
            # Create a MagicMock that will return our test URL
            mock_client.return_value.generate_presigned_url.return_value = (
                "https://test.r2.dev/test-bucket/test-key?signature=abc"
            )

            from app.storage import get_object_url

            url = get_object_url("test-key")

            # Verify the method was called correctly
            mock_client.return_value.generate_presigned_url.assert_called_once()
            call_args = mock_client.return_value.generate_presigned_url.call_args
            assert call_args[0][0] == "get_object"
            assert call_args[1]["Params"]["Key"] == "test-key"

    def test_get_object_url_put_method(self):
        """Should generate PUT presigned URL when requested."""
        with patch("app.storage._client") as mock_client:
            mock_client.return_value.generate_presigned_url.return_value = (
                "https://test.r2.dev/test-bucket/test-key?signature=abc"
            )

            from app.storage import get_object_url

            url = get_object_url("test-key", method="put_object")

            # Verify put_object method was requested
            call_args = mock_client.return_value.generate_presigned_url.call_args
            assert call_args[0][0] == "put_object"

    def test_get_object_url_custom_expires(self):
        """Should respect custom expiration time."""
        with patch("app.storage._client") as mock_client:
            mock_client.return_value.generate_presigned_url.return_value = "url"

            from app.storage import get_object_url

            url = get_object_url("test-key", expires=7200)

            # Verify custom expires was passed
            call_args = mock_client.return_value.generate_presigned_url.call_args
            assert call_args[1]["ExpiresIn"] == 7200


class TestAsyncUpload:
    """Tests for async upload operations."""

    @pytest.mark.asyncio
    async def test_upload_file_async(self):
        """Should upload file using httpx with presigned URL."""
        with patch("app.storage._client") as mock_boto:
            mock_boto.return_value.generate_presigned_url.return_value = "https://test.url"

            # Create test file
            with tempfile.NamedTemporaryFile(delete=False, mode="wb") as f:
                f.write(b"test data")
                test_path = f.name

            try:
                # Mock httpx.AsyncClient at module level
                mock_httpx_client = MagicMock()
                mock_httpx_instance = AsyncMock()
                mock_httpx_instance.__aenter__.return_value = mock_httpx_instance

                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_httpx_instance.put = AsyncMock(return_value=mock_response)

                mock_httpx_client.return_value = mock_httpx_instance

                with patch("httpx.AsyncClient", mock_httpx_client):
                    from app.storage import upload_file_async

                    result = await upload_file_async(test_path, "test-key")

                    assert result == "test-key"
                    # Verify httpx put was called
                    mock_httpx_instance.put.assert_called_once()

            finally:
                Path(test_path).unlink()

    @pytest.mark.asyncio
    async def test_upload_bytes_async(self):
        """Should upload bytes using httpx."""
        with patch("app.storage._client") as mock_boto:
            mock_boto.return_value.generate_presigned_url.return_value = "https://test.url"

            mock_httpx_client = MagicMock()
            mock_httpx_instance = AsyncMock()
            mock_httpx_instance.__aenter__.return_value = mock_httpx_instance

            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_httpx_instance.put = AsyncMock(return_value=mock_response)

            mock_httpx_client.return_value = mock_httpx_instance

            with patch("httpx.AsyncClient", mock_httpx_client):
                from app.storage import upload_bytes_async

                result = await upload_bytes_async(b"test data", "test-key")

                assert result == "test-key"
                mock_httpx_instance.put.assert_called_once()


class TestAsyncDownload:
    """Tests for async download operations."""

    @pytest.mark.asyncio
    async def test_download_file_async(self):
        """Should download file using httpx."""
        with patch("app.storage._client") as mock_boto:
            mock_boto.return_value.generate_presigned_url.return_value = "https://test.url"

            mock_httpx_client = MagicMock()
            mock_httpx_instance = AsyncMock()
            mock_httpx_instance.__aenter__.return_value = mock_httpx_instance

            mock_response = Mock()
            mock_response.content = b"downloaded data"
            mock_response.raise_for_status = Mock()
            mock_httpx_instance.get = AsyncMock(return_value=mock_response)

            mock_httpx_client.return_value = mock_httpx_instance

            with patch("httpx.AsyncClient", mock_httpx_client):
                from app.storage import download_file_async

                with tempfile.TemporaryDirectory() as tmpdir:
                    result = await download_file_async("test-key", Path(tmpdir) / "test.mp4")

                    assert result == str(Path(tmpdir) / "test.mp4")
                    mock_httpx_instance.get.assert_called_once()

                    # Verify file was written
                    assert (Path(tmpdir) / "test.mp4").read_bytes() == b"downloaded data"
