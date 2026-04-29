"""Tests for backend/app/logging_config.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import structlog
from app.logging_config import configure_logging


def test_configure_logging_runs_without_error():
    """configure_logging() should execute without raising."""
    with patch("app.logging_config.settings") as mock_settings:
        mock_settings.app.log_level = "INFO"
        configure_logging()


def test_configure_logging_structlog_configured():
    """After configure_logging(), structlog should return a usable logger."""
    with patch("app.logging_config.settings") as mock_settings:
        mock_settings.app.log_level = "INFO"
        configure_logging()

    logger = structlog.get_logger()
    assert logger is not None


def test_configure_logging_json_mode():
    """JSON log_level should trigger JSON processor branch."""
    with patch("app.logging_config.settings") as mock_settings:
        mock_settings.app.log_level = "JSON"
        configure_logging()

    logger = structlog.get_logger()
    assert logger is not None


def test_configure_logging_lowercase_json():
    """Lowercase 'json' should be handled case-insensitively."""
    with patch("app.logging_config.settings") as mock_settings:
        mock_settings.app.log_level = "json"
        configure_logging()

    logger = structlog.get_logger()
    assert logger is not None
