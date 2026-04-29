"""Tests for shared Pydantic schemas."""

from __future__ import annotations

import pytest
from app.schemas import ErrorResponse, ValidationErrorDetail
from pydantic import ValidationError


class TestErrorResponse:
    def test_minimal_error(self):
        e = ErrorResponse(error="NotFound", message="User not found")
        assert e.error == "NotFound"
        assert e.message == "User not found"
        assert e.details is None
        assert e.path == ""

    def test_with_details_dict(self):
        e = ErrorResponse(
            error="ValidationError",
            message="Bad input",
            details={"field": "email"},
        )
        assert e.details == {"field": "email"}

    def test_with_details_list(self):
        e = ErrorResponse(
            error="ValidationError",
            message="Bad input",
            details=[
                ValidationErrorDetail(field="email", message="Invalid", value="bad"),
            ],
        )
        assert len(e.details) == 1  # type: ignore[arg-type]
        assert e.details[0].field == "email"  # type: ignore[index]

    def test_model_dump(self):
        e = ErrorResponse(error="Test", message="Msg", path="/api/v1/users")
        d = e.model_dump()
        assert d["error"] == "Test"
        assert d["message"] == "Msg"
        assert d["path"] == "/api/v1/users"
        assert d["details"] is None
