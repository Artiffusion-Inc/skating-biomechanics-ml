"""Auth API routes: register, login, refresh, logout."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from litestar import Controller, post
from litestar.exceptions import ClientException
from litestar.status_codes import HTTP_201_CREATED, HTTP_204_NO_CONTENT

from app.auth.deps import DbDep
from app.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)
from app.config import get_settings
from app.crud.refresh_token import create as create_refresh_token_crud
from app.crud.refresh_token import get_active_by_hash, revoke
from app.crud.user import create as create_user
from app.crud.user import get_by_email
from app.schemas import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)

settings = get_settings()


class AuthController(Controller):
    path = "/auth"
    tags: ClassVar[list[str]] = ["auth"]

    async def _issue_token_pair(
        self, db: DbDep, user_id: str, family_id: str | None = None
    ) -> TokenResponse:
        """Create and persist a new access + refresh token pair."""
        access = create_access_token(user_id=user_id)
        refresh_str = create_refresh_token()
        fam = family_id or str(uuid.uuid4())
        await create_refresh_token_crud(
            db,
            user_id=user_id,
            token_hash=hash_token(refresh_str),
            family_id=fam,
            expires_at=datetime.now(UTC) + timedelta(days=settings.jwt.refresh_token_expire_days),
        )
        return TokenResponse(access_token=access, refresh_token=refresh_str)

    @post("/register", status_code=HTTP_201_CREATED)
    async def register(self, db: DbDep, data: RegisterRequest) -> TokenResponse:
        """Register a new user."""
        existing = await get_by_email(db, data.email)
        if existing:
            raise ClientException(
                status_code=409,
                detail="Email already registered",
            )

        user = await create_user(
            db,
            email=data.email,
            hashed_password=hash_password(data.password),
            display_name=data.display_name,
        )
        return await self._issue_token_pair(db, user.id)

    @post("/login")
    async def login(self, db: DbDep, data: LoginRequest) -> TokenResponse:
        """Authenticate and return tokens."""
        user = await get_by_email(db, data.email)
        if not user or not verify_password(data.password, user.hashed_password):
            raise ClientException(
                status_code=401,
                detail="Invalid email or password",
            )
        return await self._issue_token_pair(db, user.id)

    @post("/refresh")
    async def refresh(self, db: DbDep, data: RefreshRequest) -> TokenResponse:
        """Rotate refresh token and issue new token pair."""
        token_hash = hash_token(data.refresh_token)
        existing = await get_active_by_hash(db, token_hash)
        if not existing:
            raise ClientException(
                status_code=401,
                detail="Invalid or expired refresh token",
            )
        await revoke(db, existing)
        return await self._issue_token_pair(db, existing.user_id, family_id=existing.family_id)

    @post("/logout", status_code=HTTP_204_NO_CONTENT)
    async def logout(self, db: DbDep, data: RefreshRequest) -> None:
        """Revoke a refresh token (client should also discard access token)."""
        token_hash = hash_token(data.refresh_token)
        existing = await get_active_by_hash(db, token_hash)
        if existing:
            await revoke(db, existing)
