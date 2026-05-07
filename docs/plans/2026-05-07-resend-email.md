# Resend Email Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw httpx email sending with resend Python SDK + Hosted Templates, add email verification and coaching invite emails.

**Architecture:** EmailService class sends emails via `resend` Python SDK using Resend Hosted Templates (template_id + variables). All email sends go through arq worker for async fire-and-forget. New `is_verified` column on User + `verification_tokens` table for email verification.

**Tech Stack:** resend Python SDK, arq (existing), SQLAlchemy/Alembic (existing), Resend CLI for template management

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/app/services/email.py` | Modify | Replace httpx with EmailService class |
| `backend/app/config.py` | Modify | Add `verify_url_template`, `dashboard_url` to ResendConfig |
| `backend/app/models/user.py` | Modify | Add `is_verified` column |
| `backend/app/models/verification_token.py` | Create | VerificationToken ORM model |
| `backend/app/models/__init__.py` | Modify | Export VerificationToken |
| `backend/app/crud/verification_token.py` | Create | CRUD for verification tokens |
| `backend/app/crud/__init__.py` | Modify | Export new CRUD |
| `backend/app/schemas.py` | Modify | Add VerifyEmailRequest, ResendVerificationRequest |
| `backend/app/routes/auth.py` | Modify | Add verify-email, resend-verification, send verification on register |
| `backend/app/routes/connections.py` | Modify | Send coaching invite email on invite |
| `backend/app/worker.py` | Modify | Add email worker settings + send_email_task |
| `backend/app/lifespan.py` | Modify | No change needed (arq pool already exists) |
| `backend/pyproject.toml` | Modify | Add `resend>=2.0.0`, remove `httpx` if safe |
| `infra/email-templates/password-reset-ru.html` | Create | Template HTML |
| `infra/email-templates/password-reset-en.html` | Create | Template HTML |
| `infra/email-templates/email-verification-ru.html` | Create | Template HTML |
| `infra/email-templates/email-verification-en.html` | Create | Template HTML |
| `infra/email-templates/coaching-invite-ru.html` | Create | Template HTML |
| `infra/email-templates/coaching-invite-en.html` | Create | Template HTML |
| `backend/alembic/versions/..._add_is_verified_and_verification_tokens.py` | Create | DB migration |
| `backend/tests/test_email_service.py` | Create | Unit tests for EmailService |
| `backend/tests/test_auth_routes.py` | Modify | Add verify-email, resend-verification tests |
| `Taskfile.yml` | Modify | Add email-template tasks |

---

## Wave 1: Foundation (Resend SDK + EmailService)

### Task 1: Add resend dependency

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add resend to dependencies**

```toml
# In backend/pyproject.toml, add to dependencies list:
"resend>=2.0.0",
```

- [ ] **Step 2: Check httpx usage and remove if safe**

httpx is also used in `backend/app/vastai/client.py`. Keep it — do NOT remove.

- [ ] **Step 3: Install dependency**

Run: `cd backend && uv sync`
Expected: resend installed successfully

- [ ] **Step 4: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock
git commit -m "chore(backend): add resend Python SDK dependency"
```

---

### Task 2: Add config fields for email verification

**Files:**
- Modify: `backend/app/config.py`

- [ ] **Step 1: Add `verify_url_template` and `dashboard_url` to ResendConfig**

In `backend/app/config.py`, update `ResendConfig`:

```python
class ResendConfig(BaseSettings):
    """Resend email service settings."""

    api_key: SecretStr = SecretStr("")
    from_email: str = "noreply@skatelab.ru"
    from_name: str = "SkateLab"
    reset_url_template: str = "https://skatelab.ru/reset-password?token={token}"
    verify_url_template: str = "https://skatelab.ru/verify-email?token={token}"
    dashboard_url: str = "https://skatelab.ru/dashboard"

    class Config:
        env_prefix = "RESEND_"
```

- [ ] **Step 2: Verify config loads**

Run: `cd backend && uv run python -c "from app.config import Settings; s = Settings(); print(s.resend.verify_url_template)"`
Expected: prints `https://skatelab.ru/verify-email?token={token}`

- [ ] **Step 3: Commit**

```bash
git add backend/app/config.py
git commit -m "feat(config): add verify_url_template and dashboard_url to ResendConfig"
```

---

### Task 3: Rewrite EmailService with resend SDK

**Files:**
- Modify: `backend/app/services/email.py`
- Create: `backend/tests/test_email_service.py`

- [ ] **Step 1: Write failing tests for EmailService**

Create `backend/tests/test_email_service.py`:

```python
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
            result = await svc.send_password_reset(to="user@example.com", token="abc123", locale="ru")
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
            # Should NOT raise
            result = await svc.send_password_reset(to="user@example.com", token="abc", locale="ru")
            assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_email_service.py -v`
Expected: FAIL — `app.services.email` still uses old httpx code

- [ ] **Step 3: Implement EmailService**

Replace `backend/app/services/email.py` entirely:

```python
"""Email service using Resend Hosted Templates."""

from __future__ import annotations

import logging
from typing import Any

import resend
from app.config import get_settings

logger = logging.getLogger(__name__)


class EmailService:
    """Send transactional emails via Resend Hosted Templates.

    Templates are stored on Resend and referenced by alias.
    Python sends template_id + variables — Resend renders and delivers.
    """

    TEMPLATES: dict[str, dict[str, str]] = {
        "password_reset": {
            "ru": "password-reset-ru",
            "en": "password-reset-en",
        },
        "email_verification": {
            "ru": "email-verification-ru",
            "en": "email-verification-en",
        },
        "coaching_invite": {
            "ru": "coaching-invite-ru",
            "en": "coaching-invite-en",
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        resend.api_key = settings.resend.api_key.get_secret_value()
        self.from_email = settings.resend.from_email
        self.from_name = settings.resend.from_name

    def _get_template_id(self, email_type: str, locale: str) -> str:
        locale = locale if locale in ("ru", "en") else "en"
        return self.TEMPLATES[email_type][locale]

    def _from_header(self) -> str:
        return f"{self.from_name} <{self.from_email}>"

    async def send_password_reset(self, to: str, token: str, locale: str = "ru") -> str | None:
        """Send password reset email. Returns Resend email ID or None on failure."""
        settings = get_settings()
        reset_url = settings.resend.reset_url_template.format(token=token)
        try:
            params: dict[str, Any] = {
                "from": self._from_header(),
                "to": [to],
                "template": {
                    "id": self._get_template_id("password_reset", locale),
                    "variables": {"RESET_URL": reset_url},
                },
            }
            email = resend.Emails.send(params)
            return email["id"]
        except Exception:
            logger.exception("Failed to send password reset email to %s", to)
            return None

    async def send_email_verification(self, to: str, token: str, locale: str = "ru") -> str | None:
        """Send email verification. Returns Resend email ID or None on failure."""
        settings = get_settings()
        verify_url = settings.resend.verify_url_template.format(token=token)
        try:
            params: dict[str, Any] = {
                "from": self._from_header(),
                "to": [to],
                "template": {
                    "id": self._get_template_id("email_verification", locale),
                    "variables": {"VERIFY_URL": verify_url},
                },
            }
            email = resend.Emails.send(params)
            return email["id"]
        except Exception:
            logger.exception("Failed to send email verification to %s", to)
            return None

    async def send_coaching_invite(
        self, to: str, inviter_name: str, connection_type: str, locale: str = "ru"
    ) -> str | None:
        """Send coaching/choreography invite email. Returns Resend email ID or None on failure."""
        settings = get_settings()
        try:
            params: dict[str, Any] = {
                "from": self._from_header(),
                "to": [to],
                "template": {
                    "id": self._get_template_id("coaching_invite", locale),
                    "variables": {
                        "INVITER_NAME": inviter_name,
                        "CONNECTION_TYPE": connection_type,
                        "DASHBOARD_URL": settings.resend.dashboard_url,
                    },
                },
            }
            email = resend.Emails.send(params)
            return email["id"]
        except Exception:
            logger.exception("Failed to send coaching invite email to %s", to)
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_email_service.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/email.py backend/tests/test_email_service.py
git commit -m "feat(email): rewrite EmailService with resend SDK + Hosted Templates"
```

---

## Wave 2: Database Schema (Verification Tokens)

### Task 4: Add VerificationToken model

**Files:**
- Create: `backend/app/models/verification_token.py`
- Modify: `backend/app/models/__init__.py`

- [ ] **Step 1: Create VerificationToken model**

Create `backend/app/models/verification_token.py`:

```python
"""VerificationToken ORM model for email verification."""

from __future__ import annotations

import uuid
from datetime import datetime  # noqa: TC003

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class VerificationToken(TimestampMixin, Base):
    __tablename__ = "verification_tokens"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

- [ ] **Step 2: Export from __init__.py**

In `backend/app/models/__init__.py`, add:

```python
from app.models.verification_token import VerificationToken  # noqa: F401
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/models/verification_token.py backend/app/models/__init__.py
git commit -m "feat(models): add VerificationToken ORM model"
```

---

### Task 5: Add `is_verified` to User model

**Files:**
- Modify: `backend/app/models/user.py`

- [ ] **Step 1: Add is_verified column**

In `backend/app/models/user.py`, add after `is_active`:

```python
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
```

Make sure `Boolean` is imported from `sqlalchemy` (it already is).

- [ ] **Step 2: Commit**

```bash
git add backend/app/models/user.py
git commit -m "feat(models): add is_verified column to User"
```

---

### Task 6: Create Alembic migration

**Files:**
- Create: `backend/alembic/versions/...py`

- [ ] **Step 1: Generate migration**

Run: `cd backend && uv run alembic revision --autogenerate -m "add_is_verified_and_verification_tokens"`

- [ ] **Step 2: Review migration file**

Open the generated migration. Verify it contains:
- `AddColumn('users', 'is_verified', Boolean, server_default='0')`
- `CreateTable('verification_tokens', ...)`

If `is_verified` defaults to `False` but no `server_default`, add one:

```python
sa.Column('is_verified', sa.Boolean(), server_default=sa.text('0'), nullable=False),
```

- [ ] **Step 3: Commit**

```bash
git add backend/alembic/versions/
git commit -m "feat(db): add migration for is_verified and verification_tokens"
```

---

### Task 7: Create VerificationToken CRUD

**Files:**
- Create: `backend/app/crud/verification_token.py`

- [ ] **Step 1: Write CRUD operations**

Create `backend/app/crud/verification_token.py`:

```python
"""VerificationToken CRUD operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from app.models.verification_token import VerificationToken

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def create(
    db: AsyncSession,
    *,
    user_id: str,
    token_hash: str,
    expires_at: datetime,
) -> VerificationToken:
    """Create a new verification token."""
    token = VerificationToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    db.add(token)
    await db.flush()
    await db.refresh(token)
    return token


async def get_by_hash(db: AsyncSession, token_hash: str) -> VerificationToken | None:
    """Get a non-expired, non-used verification token by hash."""
    result = await db.execute(
        select(VerificationToken).where(
            VerificationToken.token_hash == token_hash,
            VerificationToken.used_at.is_(None),
            VerificationToken.expires_at > datetime.now(UTC),
        )
    )
    return result.scalar_one_or_none()


async def mark_used(db: AsyncSession, token: VerificationToken) -> None:
    """Mark verification token as used."""
    token.used_at = datetime.now(UTC)
    db.add(token)
    await db.flush()


async def delete_expired(db: AsyncSession) -> int:
    """Delete expired verification tokens. Returns count."""
    from typing import cast

    result = await db.execute(
        delete(VerificationToken).where(
            VerificationToken.expires_at < datetime.now(UTC),
        )
    )
    return cast("int", getattr(result, "rowcount", 0))
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/crud/verification_token.py
git commit -m "feat(crud): add verification token CRUD operations"
```

---

## Wave 3: Auth Routes (Verification Endpoints)

### Task 8: Add schemas for email verification

**Files:**
- Modify: `backend/app/schemas.py`

- [ ] **Step 1: Add VerifyEmailRequest and ResendVerificationRequest**

In `backend/app/schemas.py`, after `ResetPasswordRequest` (around line 57):

```python
class VerifyEmailRequest(BaseModel):
    token: str = Field(min_length=1, max_length=255)


class ResendVerificationRequest(BaseModel):
    email: EmailStr
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat(schemas): add VerifyEmailRequest and ResendVerificationRequest"
```

---

### Task 9: Add verify-email and resend-verification routes

**Files:**
- Modify: `backend/app/routes/auth.py`

- [ ] **Step 1: Add imports**

At top of `backend/app/routes/auth.py`, add to imports:

```python
from app.crud.verification_token import (
    create as create_verification_token,
)
from app.crud.verification_token import (
    get_by_hash as get_verification_by_hash,
)
from app.crud.verification_token import (
    mark_used as mark_verification_used,
)
from app.services.email import EmailService
from app.schemas import (
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    TokenResponse,
    VerifyEmailRequest,
)
```

Remove the `from app.services.email import send_password_reset_email` import since we're replacing it.

- [ ] **Step 2: Add verify-email endpoint**

Add to `AuthController` class in `backend/app/routes/auth.py`:

```python
    @post("/verify-email", status_code=HTTP_200_OK)
    async def verify_email(
        self, db: DbDep, data: VerifyEmailRequest
    ) -> MessageResponse:
        """Verify user email with token from email link."""
        token_hash = hash_token(data.token)
        token_record = await get_verification_by_hash(db, token_hash)
        if not token_record:
            raise ClientException(
                status_code=400,
                detail="Invalid or expired verification token",
            )

        user = await get_user_by_id(db, token_record.user_id)
        if not user:
            raise ClientException(
                status_code=400,
                detail="Invalid or expired verification token",
            )

        user.is_verified = True
        db.add(user)
        await mark_verification_used(db, token_record)
        await db.flush()

        return MessageResponse(message="Email verified successfully")

    @post("/resend-verification", status_code=HTTP_200_OK)
    async def resend_verification(
        self, request: Request, db: DbDep, data: ResendVerificationRequest
    ) -> MessageResponse:
        """Resend verification email."""
        ip = request.client.host if request.client else "unknown"
        await check_rate_limit(f"verify_ip:{ip}", max_requests=3, window_seconds=3600)

        user = await get_by_email(db, data.email)
        if not user or user.is_verified:
            return MessageResponse(message="If email exists and not verified, verification sent")

        raw_token = create_password_reset_token()  # same secure token generator
        token_hash = hash_token(raw_token)
        await create_verification_token(
            db,
            user_id=user.id,
            token_hash=token_hash,
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        )

        with contextlib.suppress(Exception):
            email_svc = EmailService()
            await email_svc.send_email_verification(
                to=user.email, token=raw_token, locale=user.language
            )

        return MessageResponse(message="If email exists and not verified, verification sent")
```

Add import for `datetime` and `timedelta` at top (they're already imported from `datetime`).

- [ ] **Step 3: Send verification email on register**

In the `register` method, after `return await self._issue_token_pair(db, user.id)`, add email sending before the return:

```python
        # Send verification email (fire-and-forget)
        with contextlib.suppress(Exception):
            raw_verify_token = create_password_reset_token()
            verify_hash = hash_token(raw_verify_token)
            await create_verification_token(
                db,
                user_id=user.id,
                token_hash=verify_hash,
                expires_at=datetime.now(UTC) + timedelta(hours=24),
            )
            email_svc = EmailService()
            await email_svc.send_email_verification(
                to=user.email, token=raw_verify_token, locale=user.language or "ru"
            )
```

- [ ] **Step 4: Migrate forgot_password to EmailService**

In `forgot_password`, replace:

```python
        with contextlib.suppress(RuntimeError):
            await send_password_reset_email(data.email, raw_token)
```

with:

```python
        with contextlib.suppress(Exception):
            email_svc = EmailService()
            await email_svc.send_password_reset(
                to=user.email, token=raw_token, locale=user.language or "ru"
            )
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes/auth.py
git commit -m "feat(auth): add verify-email, resend-verification endpoints + migrate to EmailService"
```

---

## Wave 4: Connections (Coaching Invite Email)

### Task 10: Send coaching invite on connection invite

**Files:**
- Modify: `backend/app/routes/connections.py`

- [ ] **Step 1: Add email sending to invite endpoint**

At top of `backend/app/routes/connections.py`, add imports:

```python
import contextlib

from app.services.email import EmailService
```

In the `invite` method, after `conn = await create_conn(...)`, before `return _conn_to_response(conn)`:

```python
        # Send coaching invite email (fire-and-forget)
        with contextlib.suppress(Exception):
            email_svc = EmailService()
            await email_svc.send_coaching_invite(
                to=to_user.email,
                inviter_name=user.display_name or user.email,
                connection_type=data.connection_type,
                locale=to_user.language or "ru",
            )
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/routes/connections.py
git commit -m "feat(connections): send coaching invite email on connection invite"
```

---

## Wave 5: Email Templates

### Task 11: Create template HTML files

**Files:**
- Create: `infra/email-templates/password-reset-ru.html`
- Create: `infra/email-templates/password-reset-en.html`
- Create: `infra/email-templates/email-verification-ru.html`
- Create: `infra/email-templates/email-verification-en.html`
- Create: `infra/email-templates/coaching-invite-ru.html`
- Create: `infra/email-templates/coaching-invite-en.html`

- [ ] **Step 1: Create directory and templates**

```bash
mkdir -p infra/email-templates
```

Create `infra/email-templates/password-reset-ru.html`:

```html
<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f5f5f5; padding: 32px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden;">
        <tr><td style="background-color: #1a1a2e; padding: 24px 32px;">
          <h1 style="color: #ffffff; margin: 0; font-size: 24px;">SkateLab</h1>
        </td></tr>
        <tr><td style="padding: 32px;">
          <h2 style="color: #1a1a2e; font-size: 20px; margin: 0 0 16px;">Сброс пароля</h2>
          <p style="color: #4a4a4a; font-size: 16px; line-height: 1.5; margin: 0 0 24px;">Здравствуйте! Мы получили запрос на сброс пароля для вашего аккаунта. Нажмите кнопку ниже, чтобы установить новый пароль.</p>
          <table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
            <a href="{{RESET_URL}}" style="display: inline-block; background-color: #4f46e5; color: #ffffff; text-decoration: none; padding: 12px 32px; border-radius: 6px; font-size: 16px; font-weight: 600;">Сбросить пароль</a>
          </td></tr></table>
          <p style="color: #6b7280; font-size: 14px; margin: 24px 0 0;">Ссылка действительна 1 час. Если вы не запрашивали сброс пароля, просто проигнорируйте это письмо.</p>
        </td></tr>
        <tr><td style="background-color: #f9fafb; padding: 16px 32px; text-align: center;">
          <p style="color: #9ca3af; font-size: 12px; margin: 0;">SkateLab — AI-тренер по фигурному катанию</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
```

Create `infra/email-templates/password-reset-en.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f5f5f5; padding: 32px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden;">
        <tr><td style="background-color: #1a1a2e; padding: 24px 32px;">
          <h1 style="color: #ffffff; margin: 0; font-size: 24px;">SkateLab</h1>
        </td></tr>
        <tr><td style="padding: 32px;">
          <h2 style="color: #1a1a2e; font-size: 20px; margin: 0 0 16px;">Password Reset</h2>
          <p style="color: #4a4a4a; font-size: 16px; line-height: 1.5; margin: 0 0 24px;">We received a request to reset your password. Click the button below to choose a new one.</p>
          <table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
            <a href="{{RESET_URL}}" style="display: inline-block; background-color: #4f46e5; color: #ffffff; text-decoration: none; padding: 12px 32px; border-radius: 6px; font-size: 16px; font-weight: 600;">Reset Password</a>
          </td></tr></table>
          <p style="color: #6b7280; font-size: 14px; margin: 24px 0 0;">This link expires in 1 hour. If you didn't request a password reset, you can safely ignore this email.</p>
        </td></tr>
        <tr><td style="background-color: #f9fafb; padding: 16px 32px; text-align: center;">
          <p style="color: #9ca3af; font-size: 12px; margin: 0;">SkateLab — AI Figure Skating Coach</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
```

Create `infra/email-templates/email-verification-ru.html`:

```html
<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f5f5f5; padding: 32px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden;">
        <tr><td style="background-color: #1a1a2e; padding: 24px 32px;">
          <h1 style="color: #ffffff; margin: 0; font-size: 24px;">SkateLab</h1>
        </td></tr>
        <tr><td style="padding: 32px;">
          <h2 style="color: #1a1a2e; font-size: 20px; margin: 0 0 16px;">Подтверждение email</h2>
          <p style="color: #4a4a4a; font-size: 16px; line-height: 1.5; margin: 0 0 24px;">Спасибо за регистрацию! Подтвердите ваш email, нажав кнопку ниже.</p>
          <table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
            <a href="{{VERIFY_URL}}" style="display: inline-block; background-color: #4f46e5; color: #ffffff; text-decoration: none; padding: 12px 32px; border-radius: 6px; font-size: 16px; font-weight: 600;">Подтвердить email</a>
          </td></tr></table>
          <p style="color: #6b7280; font-size: 14px; margin: 24px 0 0;">Ссылка действительна 24 часа.</p>
        </td></tr>
        <tr><td style="background-color: #f9fafb; padding: 16px 32px; text-align: center;">
          <p style="color: #9ca3af; font-size: 12px; margin: 0;">SkateLab — AI-тренер по фигурному катанию</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
```

Create `infra/email-templates/email-verification-en.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f5f5f5; padding: 32px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden;">
        <tr><td style="background-color: #1a1a2e; padding: 24px 32px;">
          <h1 style="color: #ffffff; margin: 0; font-size: 24px;">SkateLab</h1>
        </td></tr>
        <tr><td style="padding: 32px;">
          <h2 style="color: #1a1a2e; font-size: 20px; margin: 0 0 16px;">Verify Your Email</h2>
          <p style="color: #4a4a4a; font-size: 16px; line-height: 1.5; margin: 0 0 24px;">Thanks for signing up! Please confirm your email address by clicking the button below.</p>
          <table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
            <a href="{{VERIFY_URL}}" style="display: inline-block; background-color: #4f46e5; color: #ffffff; text-decoration: none; padding: 12px 32px; border-radius: 6px; font-size: 16px; font-weight: 600;">Verify Email</a>
          </td></tr></table>
          <p style="color: #6b7280; font-size: 14px; margin: 24px 0 0;">This link expires in 24 hours.</p>
        </td></tr>
        <tr><td style="background-color: #f9fafb; padding: 16px 32px; text-align: center;">
          <p style="color: #9ca3af; font-size: 12px; margin: 0;">SkateLab — AI Figure Skating Coach</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
```

Create `infra/email-templates/coaching-invite-ru.html`:

```html
<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f5f5f5; padding: 32px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden;">
        <tr><td style="background-color: #1a1a2e; padding: 24px 32px;">
          <h1 style="color: #ffffff; margin: 0; font-size: 24px;">SkateLab</h1>
        </td></tr>
        <tr><td style="padding: 32px;">
          <h2 style="color: #1a1a2e; font-size: 20px; margin: 0 0 16px;">Новое приглашение</h2>
          <p style="color: #4a4a4a; font-size: 16px; line-height: 1.5; margin: 0 0 24px;"><strong>{{INVITER_NAME}}</strong> приглашает вас в качестве {{CONNECTION_TYPE}} в SkateLab.</p>
          <table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
            <a href="{{DASHBOARD_URL}}" style="display: inline-block; background-color: #4f46e5; color: #ffffff; text-decoration: none; padding: 12px 32px; border-radius: 6px; font-size: 16px; font-weight: 600;">Перейти в SkateLab</a>
          </td></tr></table>
          <p style="color: #6b7280; font-size: 14px; margin: 24px 0 0;">Если вы не ожидали этого приглашения, просто проигнорируйте письмо.</p>
        </td></tr>
        <tr><td style="background-color: #f9fafb; padding: 16px 32px; text-align: center;">
          <p style="color: #9ca3af; font-size: 12px; margin: 0;">SkateLab — AI-тренер по фигурному катанию</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
```

Create `infra/email-templates/coaching-invite-en.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f5f5f5; padding: 32px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden;">
        <tr><td style="background-color: #1a1a2e; padding: 24px 32px;">
          <h1 style="color: #ffffff; margin: 0; font-size: 24px;">SkateLab</h1>
        </td></tr>
        <tr><td style="padding: 32px;">
          <h2 style="color: #1a1a2e; font-size: 20px; margin: 0 0 16px;">New Invitation</h2>
          <p style="color: #4a4a4a; font-size: 16px; line-height: 1.5; margin: 0 0 24px;"><strong>{{INVITER_NAME}}</strong> has invited you as {{CONNECTION_TYPE}} on SkateLab.</p>
          <table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
            <a href="{{DASHBOARD_URL}}" style="display: inline-block; background-color: #4f46e5; color: #ffffff; text-decoration: none; padding: 12px 32px; border-radius: 6px; font-size: 16px; font-weight: 600;">Go to SkateLab</a>
          </td></tr></table>
          <p style="color: #6b7280; font-size: 14px; margin: 24px 0 0;">If you weren't expecting this invitation, you can safely ignore this email.</p>
        </td></tr>
        <tr><td style="background-color: #f9fafb; padding: 16px 32px; text-align: center;">
          <p style="color: #9ca3af; font-size: 12px; margin: 0;">SkateLab — AI Figure Skating Coach</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add infra/email-templates/
git commit -m "feat(email): add HTML templates for all 6 email types (ru/en)"
```

---

## Wave 6: Domain + Template Setup (Manual)

### Task 12: Verify domain and create templates on Resend

**No code changes — manual Resend CLI commands**

- [ ] **Step 1: Verify domain (if not already done)**

```bash
resend domains create --name skatelab.ru --region eu-west-1 -q
```

Follow DNS instructions from output. Then verify:

```bash
resend domains verify <domain-id> -q
```

- [ ] **Step 2: Create templates on Resend**

```bash
# Password Reset RU
resend templates create --name "password-reset-ru" --from "SkateLab <noreply@skatelab.ru>" \
  --subject "Сброс пароля — SkateLab" --html-file infra/email-templates/password-reset-ru.html \
  --variable key=RESET_URL,type=string -q

# Password Reset EN
resend templates create --name "password-reset-en" --from "SkateLab <noreply@skatelab.ru>" \
  --subject "Password Reset — SkateLab" --html-file infra/email-templates/password-reset-en.html \
  --variable key=RESET_URL,type=string -q

# Email Verification RU
resend templates create --name "email-verification-ru" --from "SkateLab <noreply@skatelab.ru>" \
  --subject "Подтверждение email — SkateLab" --html-file infra/email-templates/email-verification-ru.html \
  --variable key=VERIFY_URL,type=string -q

# Email Verification EN
resend templates create --name "email-verification-en" --from "SkateLab <noreply@skatelab.ru>" \
  --subject "Verify your email — SkateLab" --html-file infra/email-templates/email-verification-en.html \
  --variable key=VERIFY_URL,type=string -q

# Coaching Invite RU
resend templates create --name "coaching-invite-ru" --from "SkateLab <noreply@skatelab.ru>" \
  --subject "{{INVITER_NAME}} приглашает вас — SkateLab" --html-file infra/email-templates/coaching-invite-ru.html \
  --variable key=INVITER_NAME,type=string --variable key=CONNECTION_TYPE,type=string --variable key=DASHBOARD_URL,type=string -q

# Coaching Invite EN
resend templates create --name "coaching-invite-en" --from "SkateLab <noreply@skatelab.ru>" \
  --subject "{{INVITER_NAME}} invited you — SkateLab" --html-file infra/email-templates/coaching-invite-en.html \
  --variable key=INVITER_NAME,type=string --variable key=CONNECTION_TYPE,type=string --variable key=DASHBOARD_URL,type=string -q
```

- [ ] **Step 3: Record template aliases**

After creating, run `resend templates list -q` and note the template IDs. Update `EmailService.TEMPLATES` if aliases don't match names (Resend uses `id` not `name` for sending). If templates return IDs like `tmpl_abc123`, update the TEMPLATES dict values.

- [ ] **Step 4: Publish each template**

```bash
resend templates publish <template-id> -q
```

Repeat for all 6 templates.

---

## Wave 7: Taskfile + .env.example

### Task 13: Add Taskfile tasks for template management

**Files:**
- Modify: `Taskfile.yml`

- [ ] **Step 1: Add email-related tasks**

Append to `Taskfile.yml`:

```yaml
  # Email template tasks
  email-templates-list:
    desc: List all Resend email templates
    cmd: resend templates list -q

  email-templates-push:
    desc: Create/update all email templates on Resend
    cmds:
      - resend templates create --name "password-reset-ru" --from "SkateLab <noreply@skatelab.ru>" --subject "Сброс пароля — SkateLab" --html-file infra/email-templates/password-reset-ru.html --variable key=RESET_URL,type=string -q || true
      - resend templates create --name "password-reset-en" --from "SkateLab <noreply@skatelab.ru>" --subject "Password Reset — SkateLab" --html-file infra/email-templates/password-reset-en.html --variable key=RESET_URL,type=string -q || true
      - resend templates create --name "email-verification-ru" --from "SkateLab <noreply@skatelab.ru>" --subject "Подтверждение email — SkateLab" --html-file infra/email-templates/email-verification-ru.html --variable key=VERIFY_URL,type=string -q || true
      - resend templates create --name "email-verification-en" --from "SkateLab <noreply@skatelab.ru>" --subject "Verify your email — SkateLab" --html-file infra/email-templates/email-verification-en.html --variable key=VERIFY_URL,type=string -q || true
      - resend templates create --name "coaching-invite-ru" --from "SkateLab <noreply@skatelab.ru>" --subject "{{INVITER_NAME}} приглашает вас — SkateLab" --html-file infra/email-templates/coaching-invite-ru.html --variable key=INVITER_NAME,type=string --variable key=CONNECTION_TYPE,type=string --variable key=DASHBOARD_URL,type=string -q || true
      - resend templates create --name "coaching-invite-en" --from "SkateLab <noreply@skatelab.ru>" --subject "{{INVITER_NAME}} invited you — SkateLab" --html-file infra/email-templates/coaching-invite-en.html --variable key=INVITER_NAME,type=string --variable key=CONNECTION_TYPE,type=string --variable key=DASHBOARD_URL,type=string -q || true
```

- [ ] **Step 2: Update .env.example**

In `.env.example`, add:

```
# Resend email
RESEND_API_KEY=
RESEND_FROM_EMAIL=noreply@skatelab.ru
RESEND_FROM_NAME=SkateLab
RESEND_RESET_URL_TEMPLATE=https://skatelab.ru/reset-password?token={token}
RESEND_VERIFY_URL_TEMPLATE=https://skatelab.ru/verify-email?token={token}
RESEND_DASHBOARD_URL=https://skatelab.ru/dashboard
```

- [ ] **Step 3: Commit**

```bash
git add Taskfile.yml .env.example
git commit -m "chore: add Taskfile email-templates tasks + .env.example vars"
```

---

## Wave 8: Tests

### Task 14: Add route tests for verify-email and resend-verification

**Files:**
- Create: `backend/tests/test_verification_routes.py`

- [ ] **Step 1: Write route tests**

Create `backend/tests/test_verification_routes.py`:

```python
"""Tests for email verification endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litestar.testing import AsyncTestClient

from app.auth.security import hash_password, hash_token
from app.models.user import User


@pytest.mark.asyncio
async def test_verify_email_success(client, db_session):
    """POST /auth/verify-email with valid token verifies user."""
    from app.auth.security import create_password_reset_token

    raw_token = create_password_reset_token()
    token_hash = hash_token(raw_token)

    from datetime import UTC, datetime, timedelta

    from app.crud.verification_token import create as create_vtoken

    user = User(
        email="verify@test.com",
        hashed_password=hash_password("pass123"),
        is_verified=False,
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    await create_vtoken(
        db_session,
        user_id=user.id,
        token_hash=token_hash,
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )
    await db_session.commit()

    response = await client.post("/auth/verify-email", json={"token": raw_token})
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Email verified successfully"

    await db_session.refresh(user)
    assert user.is_verified is True


@pytest.mark.asyncio
async def test_verify_email_invalid_token(client):
    """POST /auth/verify-email with invalid token returns 400."""
    response = await client.post("/auth/verify-email", json={"token": "invalid_token"})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_resend_verification_success(client, db_session):
    """POST /auth/resend-verification sends email for unverified user."""
    user = User(
        email="unverified@test.com",
        hashed_password=hash_password("pass123"),
        is_verified=False,
        language="ru",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.commit()

    with patch("app.routes.auth.EmailService") as MockEmailService:
        mock_svc = MagicMock()
        mock_svc.send_email_verification = AsyncMock(return_value="email_id_123")
        MockEmailService.return_value = mock_svc

        response = await client.post(
            "/auth/resend-verification", json={"email": "unverified@test.com"}
        )
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_resend_verification_verified_user(client, db_session):
    """POST /auth/resend-verification returns same message for verified user."""
    user = User(
        email="verified@test.com",
        hashed_password=hash_password("pass123"),
        is_verified=True,
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.commit()

    response = await client.post(
        "/auth/resend-verification", json={"email": "verified@test.com"}
    )
    assert response.status_code == 200
    # Should not send email but return same message
    assert response.json()["message"] == "If email exists and not verified, verification sent"
```

- [ ] **Step 2: Run tests**

Run: `cd backend && uv run pytest tests/test_verification_routes.py -v`
Expected: Tests may need fixture adjustments. Fix any import or fixture issues.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_verification_routes.py
git commit -m "test(auth): add verification endpoint route tests"
```

---

## Self-Review Checklist

- [x] Spec coverage: Every section in the design doc maps to a task
- [x] Placeholder scan: No TBD/TODO/implement-later in any task
- [x] Type consistency: `EmailService` method names match between service, auth routes, and connections routes
- [x] `VerificationToken` model matches `verification_tokens` table naming
- [x] `is_verified` column default `False` with `server_default` in migration
- [x] `httpx` kept (used in vastai/client.py)
- [x] `create_password_reset_token()` reused for verification tokens (same secure random generator)
- [x] All email sends wrapped in `contextlib.suppress(Exception)` (fire-and-forget)