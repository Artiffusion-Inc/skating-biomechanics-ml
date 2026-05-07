# Resend Email Integration Design

**Date:** 2026-05-07
**Status:** Draft

## Summary

Replace raw httpx email sending with `resend` Python SDK + Resend Hosted Templates. Three email types: password reset (existing), email verification (new), coaching invites (new). Templates stored on Resend, rendered server-side — no local HTML rendering, no subprocess, no Jinja2.

## Architecture

```
User action → EmailService.send_*()
                ↓
         resend Python SDK
                ↓
    Resend API (template_id + variables)
                ↓
        Resend renders HTML → sends email
```

**No local template rendering.** Python sends `template_id` + `variables` dict. Resend substitutes `{{{VAR}}}` placeholders and delivers the email.

## Components

### 1. EmailService (`backend/app/services/email.py`)

Replace current `send_password_reset_email()` with class-based service:

```python
import resend
from app.config import get_settings

class EmailService:
    """Send transactional emails via Resend Hosted Templates."""

    # Template aliases — map to Resend template IDs
    TEMPLATES = {
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

    def __init__(self):
        settings = get_settings()
        resend.api_key = settings.resend.api_key.get_secret_value()
        self.from_email = settings.resend.from_email
        self.from_name = settings.resend.from_name

    def _get_template_id(self, email_type: str, locale: str) -> str:
        locale = locale if locale in ("ru", "en") else "en"
        return self.TEMPLATES[email_type][locale]

    async def send_password_reset(self, to: str, token: str, locale: str = "ru") -> str:
        """Send password reset email. Returns Resend email ID."""
        reset_url = get_settings().resend.reset_url_template.format(token=token)
        params: resend.Emails.SendParams = {
            "from": f"{self.from_name} <{self.from_email}>",
            "to": [to],
            "template": {
                "id": self._get_template_id("password_reset", locale),
                "variables": {"RESET_URL": reset_url},
            },
        }
        email = resend.Emails.send(params)
        return email["id"]

    async def send_email_verification(self, to: str, token: str, locale: str = "ru") -> str:
        """Send email verification. Returns Resend email ID."""
        verify_url = get_settings().resend.verify_url_template.format(token=token)
        params: resend.Emails.SendParams = {
            "from": f"{self.from_name} <{self.from_email}>",
            "to": [to],
            "template": {
                "id": self._get_template_id("email_verification", locale),
                "variables": {"VERIFY_URL": verify_url},
            },
        }
        email = resend.Emails.send(params)
        return email["id"]

    async def send_coaching_invite(
        self, to: str, inviter_name: str, connection_type: str, locale: str = "ru"
    ) -> str:
        """Send coaching/choreography invite email. Returns Resend email ID."""
        params: resend.Emails.SendParams = {
            "from": f"{self.from_name} <{self.from_email}>",
            "to": [to],
            "template": {
                "id": self._get_template_id("coaching_invite", locale),
                "variables": {
                    "INVITER_NAME": inviter_name,
                    "CONNECTION_TYPE": connection_type,
                    "DASHBOARD_URL": get_settings().resend.dashboard_url,
                },
            },
        }
        email = resend.Emails.send(params)
        return email["id"]
```

### 2. Config Changes (`backend/app/config.py`)

Add to `ResendConfig`:

```python
class ResendConfig(BaseSettings):
    api_key: SecretStr = SecretStr("")
    from_email: str = "noreply@skatelab.ru"
    from_name: str = "SkateLab"
    reset_url_template: str = "https://skatelab.ru/reset-password?token={token}"
    verify_url_template: str = "https://skatelab.ru/verify-email?token={token}"  # NEW
    dashboard_url: str = "https://skatelab.ru/dashboard"  # NEW
```

### 3. User Model Changes (`backend/app/models/user.py`)

Add `is_verified` field:

```python
is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
```

Add `VerificationToken` model (new file `backend/app/models/verification_token.py`):

```python
class VerificationToken(TimestampMixin, Base):
    __tablename__ = "verification_tokens"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

### 4. Auth Route Changes (`backend/app/routes/auth.py`)

- Register endpoint: send verification email after user creation
- New `POST /auth/verify-email` endpoint: validate token, set `is_verified=True`
- `POST /auth/resend-verification` endpoint: resend verification email with rate limiting
- Middleware/guard: optionally block unverified users from protected routes (configurable)

### 5. Connections Route Changes (`backend/app/routes/connections.py`)

- `POST /connections/invite`: after creating connection, send coaching invite email

### 6. Email Sending Strategy

**All email sending through arq worker** (fire-and-forget from route handlers):

```python
# In route handler:
await arq_pool.enqueue_job("send_email", email_type="coaching_invite", to=user.email, ...)

# In worker:
async def send_email_task(ctx, email_type: str, **kwargs):
    service = EmailService()
    await getattr(service, f"send_{email_type}")(**kwargs)
```

**Why arq, not inline:** email sending should not block HTTP responses. If Resend API is slow/down, users shouldn't wait. arq provides retries with `Retry` middleware.

**Exception:** password reset in `/forgot-password` is already fire-and-forget (suppressed errors). This pattern generalizes to arq.

### 7. Template Management

Templates managed via Resend CLI:

```bash
# Create template
resend templates create --name "password-reset-ru" --from "SkateLab <noreply@skatelab.ru>" \
  --subject "Сброс пароля — SkateLab" \
  --html-file templates/password-reset-ru.html

# List templates
resend templates list -q

# Update template
resend templates update <template-id> --html-file templates/password-reset-ru.html
```

Template HTML files stored in `infra/email-templates/` for VCS tracking:

```
infra/email-templates/
├── password-reset-ru.html
├── password-reset-en.html
├── email-verification-ru.html
├── email-verification-en.html
├── coaching-invite-ru.html
└── coaching-invite-en.html
```

Variables use Handlebars syntax: `{{{RESET_URL}}}`, `{{{VERIFY_URL}}}`, `{{{INVITER_NAME}}}`.

### 8. Domain Setup

Verify `skatelab.ru` domain in Resend (or use existing verified domain). DNS records needed:
- SPF, DKIM, DMARC TXT records
- Resend provides these on `resend domains verify`

## Email Types

### Password Reset (existing → migrate)

| Field | Value |
|-------|-------|
| Template aliases | `password-reset-ru`, `password-reset-en` |
| Variables | `RESET_URL` |
| Subject RU | Сброс пароля — SkateLab |
| Subject EN | Password Reset — SkateLab |
| Trigger | `POST /auth/forgot-password` |

### Email Verification (new)

| Field | Value |
|-------|-------|
| Template aliases | `email-verification-ru`, `email-verification-en` |
| Variables | `VERIFY_URL` |
| Subject RU | Подтверждение email — SkateLab |
| Subject EN | Verify your email — SkateLab |
| Trigger | `POST /auth/register` (auto) + `POST /auth/resend-verification` |
| Token TTL | 24 hours |

### Coaching Invite (new)

| Field | Value |
|-------|-------|
| Template aliases | `coaching-invite-ru`, `coaching-invite-en` |
| Variables | `INVITER_NAME`, `CONNECTION_TYPE`, `DASHBOARD_URL` |
| Subject RU | {{INVITER_NAME}} приглашает вас — SkateLab |
| Subject EN | {{INVITER_NAME}} invited you — SkateLab |
| Trigger | `POST /connections/invite` |

## Database Migration

New Alembic migration adds:

1. `users.is_verified` column (Boolean, default False)
2. `verification_tokens` table (id, user_id, token_hash, expires_at, used_at)

## Dependency Changes

`backend/pyproject.toml`:

- Add: `resend>=2.0.0` (Python SDK)
- Remove: `httpx` (if no longer used elsewhere — check first)

## Error Handling

- `resend` SDK raises `resend.exceptions.*` on API errors
- `EmailService` wraps errors with structured logging (structlog)
- arq worker retries on transient failures (Resend 429/5xx)
- Email failures never block user-facing flows — always fire-and-forget or arq-enqueued

## Testing Strategy

1. **Unit tests**: Mock `resend.Emails.send`, verify correct template_id and variables
2. **Integration tests**: Use Resend test mode API key (`re_test_*`) or `--dry-run` flag
3. **Template validation**: `resend templates list -q` in CI to verify all expected templates exist
4. **E2E**: Manual test with real email addresses on staging

## Task Breakdown

1. Add `resend` dependency, refactor `EmailService` class
2. Create 6 templates on Resend (3 types × 2 locales)
3. Store template HTML in `infra/email-templates/`
4. Add `is_verified` column + `verification_tokens` table
5. Add `/auth/verify-email` and `/auth/resend-verification` endpoints
6. Integrate email sending into `/auth/register` and `/connections/invite`
7. Add arq task for async email sending
8. Add `verify_url_template` and `dashboard_url` to config
9. Migrate existing password reset from httpx to SDK
10. Write tests
11. Update `Taskfile.yaml` with template management tasks