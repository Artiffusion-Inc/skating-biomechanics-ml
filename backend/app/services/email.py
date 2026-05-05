"""Email service using Resend API."""

from __future__ import annotations

import httpx
from app.config import get_settings


async def send_password_reset_email(to_email: str, token: str) -> None:
    """Send a password-reset email via Resend.

    Args:
        to_email: Recipient address.
        token: Raw reset token (will be inserted into the URL template).

    Raises:
        RuntimeError: If the API key is missing or the API returns an error.
    """
    settings = get_settings()
    api_key = settings.resend.api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("RESEND_API_KEY not configured")

    reset_url = settings.resend.reset_url_template.format(token=token)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "from": f"{settings.resend.from_name} <{settings.resend.from_email}>",
                "to": [to_email],
                "subject": "Сброс пароля — Skating AI Coach",
                "html": (
                    f"<p>Здравствуйте!</p>"
                    f"<p>Для сброса пароля перейдите по ссылке:</p>"
                    f'<p><a href="{reset_url}">Сбросить пароль</a></p>'
                    f"<p>Ссылка действительна 1 час.</p>"
                    f"<p>Если вы не запрашивали сброс пароля, проигнорируйте это письмо.</p>"
                ),
            },
        )
        _ = resp.raise_for_status()
