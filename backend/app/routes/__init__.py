"""Route utilities — shared helpers for all API route modules."""

from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException, Request, status

from app.schemas import ErrorResponse


def raise_api_error(
    status_code: int,
    error: str,
    message: str,
    details: dict | list | None = None,
    request: Request | None = None,
) -> NoReturn:
    """Raise an HTTPException with a structured ErrorResponse body.

    Usage:
        raise_api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            error="NotFound",
            message="User not found",
            details={"id": user_id},
            request=request,
        )
    """
    body = ErrorResponse(
        error=error,
        message=message,
        details=details,
        path=str(request.url.path) if request else "",
    )
    raise HTTPException(status_code=status_code, detail=body.model_dump())
