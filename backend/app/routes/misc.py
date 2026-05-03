"""Health check and file serving routes (R2 streaming proxy)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from litestar import Controller, get
from litestar.exceptions import ClientException
from litestar.response import Stream

from app.storage import object_exists_async, stream_object_async


class MiscController(Controller):
    path = ""
    tags: ClassVar[list[str]] = ["misc"]

    # Content-type mapping by extension
    _CONTENT_TYPES = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".npy": "application/octet-stream",
        ".csv": "text/csv",
    }

    @get("/health")
    async def health(self) -> dict:
        return {"status": "ok"}

    @get("/outputs/{key:path}")
    async def serve_output(self, key: str) -> Stream:
        """Stream file from R2 as a proxy (frontend never talks to R2 directly)."""
        if not await object_exists_async(key):
            raise ClientException(
                status_code=404,
                detail="File not found",
            )

        body, length, ctype = await stream_object_async(key)
        # Prefer extension-based content type over what S3 reports
        ext = Path(key).suffix.lower()
        if ext in self._CONTENT_TYPES:
            ctype = self._CONTENT_TYPES[ext]

        return Stream(
            body.iter_chunks(chunk_size=8192),
            media_type=ctype,
            headers={"Content-Length": str(length)},
        )
