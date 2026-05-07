"""Unit tests for EmailService (resend SDK)."""

from unittest.mock import MagicMock, patch

import pytest
from app.services.email import EmailService


@pytest.fixture
def mock_settings():
    """Patch get_settings to return a mock with resend config."""
    with patch("app.services.email.get_settings") as mock_get:
        settings = MagicMock()
        settings.resend.api_key.get_secret_value.return_value = "re_test_key"
        settings.resend.from_email = "noreply@skatelab.ru"
        settings.resend.from_name = "SkateLab"
        settings.resend.reset_url_template = "https://skatelab.ru/reset-password?token={token}"
        settings.resend.verify_url_template = "https://skatelab.ru/verify-email?token={token}"
        settings.resend.dashboard_url = "https://skatelab.ru/dashboard"
        mock_get.return_value = settings
        yield settings


class TestEmailService:
    def test_get_template_id_ru(self, mock_settings):
        svc = EmailService()
        tid = svc._get_template_id("password_reset", "ru")
        assert tid == "password-reset-ru"

    def test_get_template_id_en(self, mock_settings):
        svc = EmailService()
        tid = svc._get_template_id("password_reset", "en")
        assert tid == "password-reset-en"

    def test_get_template_id_fallback_to_en(self, mock_settings):
        svc = EmailService()
        tid = svc._get_template_id("password_reset", "fr")
        assert tid == "password-reset-en"

    @pytest.mark.asyncio
    async def test_send_password_reset(self, mock_settings):
        with patch("app.services.email.resend") as mock_resend:
            mock_resend.Emails.send.return_value = {"id": "email_123"}
            svc = EmailService()
            result = await svc.send_password_reset(
                to="user@example.com", token="abc123", locale="ru"
            )
            assert result == "email_123"
            call_args = mock_resend.Emails.send.call_args[0][0]
            assert call_args["template"]["id"] == "password-reset-ru"
            assert "RESET_URL" in call_args["template"]["variables"]
            assert "abc123" in call_args["template"]["variables"]["RESET_URL"]

    @pytest.mark.asyncio
    async def test_send_email_verification(self, mock_settings):
        with patch("app.services.email.resend") as mock_resend:
            mock_resend.Emails.send.return_value = {"id": "email_456"}
            svc = EmailService()
            result = await svc.send_email_verification(
                to="user@example.com", token="verify_tok", locale="en"
            )
            assert result == "email_456"
            call_args = mock_resend.Emails.send.call_args[0][0]
            assert call_args["template"]["id"] == "email-verification-en"
            assert "VERIFY_URL" in call_args["template"]["variables"]

    @pytest.mark.asyncio
    async def test_send_coaching_invite(self, mock_settings):
        with patch("app.services.email.resend") as mock_resend:
            mock_resend.Emails.send.return_value = {"id": "email_789"}
            svc = EmailService()
            result = await svc.send_coaching_invite(
                to="skater@example.com",
                inviter_name="Coach Alex",
                connection_type="coaching",
                locale="ru",
            )
            assert result == "email_789"
            call_args = mock_resend.Emails.send.call_args[0][0]
            assert call_args["template"]["id"] == "coaching-invite-ru"
            variables = call_args["template"]["variables"]
            assert variables["INVITER_NAME"] == "Coach Alex"
            assert variables["CONNECTION_TYPE"] == "coaching"

    @pytest.mark.asyncio
    async def test_send_password_reset_suppresses_errors(self, mock_settings):
        """Email failures must not raise — fire-and-forget pattern."""
        with patch("app.services.email.resend") as mock_resend:
            mock_resend.Emails.send.side_effect = Exception("Resend API down")
            svc = EmailService()
            result = await svc.send_password_reset(to="user@example.com", token="abc", locale="ru")
            assert result is None
