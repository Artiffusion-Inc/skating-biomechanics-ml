"""App-level exception handlers for Litestar."""

from __future__ import annotations

from litestar.exceptions import HTTPException
from litestar.response import Response

from app.schemas import ErrorResponse


def http_exception_handler(request, exc: HTTPException) -> Response:
    """Map Litestar HTTPException to structured ErrorResponse."""
    body = ErrorResponse(
        error=exc.detail if isinstance(exc.detail, str) else "Error",
        message=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        path=str(request.url.path),
    )
    return Response(
        content=body.model_dump(),
        status_code=exc.status_code,
        media_type="application/json",
    )
