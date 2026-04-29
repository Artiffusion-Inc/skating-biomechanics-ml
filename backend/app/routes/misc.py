"""Health check and file serving routes (R2 streaming proxy)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.routes import raise_api_error
from app.storage import object_exists_async, stream_object_async

router = APIRouter(tags=["misc"])

# Content-type mapping by extension
_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".npy": "application/octet-stream",
    ".csv": "text/csv",
}


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/outputs/{key:path}")
async def serve_output(key: str):
    """Stream file from R2 as a proxy (frontend never talks to R2 directly)."""
    if not await object_exists_async(key):
        raise_api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            error="NotFound",
            message="File not found",
            details={"key": key},
        )

    body, length, ctype = await stream_object_async(key)
    # Prefer extension-based content type over what S3 reports
    ext = Path(key).suffix.lower()
    if ext in _CONTENT_TYPES:
        ctype = _CONTENT_TYPES[ext]

    return StreamingResponse(
        content=body.iter_chunks(chunk_size=8192),
        media_type=ctype,
        headers={"Content-Length": str(length)},
    )
