"""Authentication dependencies for Litestar."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from litestar.exceptions import NotAuthorizedException
from litestar.params import Dependency
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.user import User

if TYPE_CHECKING:
    from litestar import Request


async def retrieve_user_handler(token, connection) -> User | None:
    """Litestar JWTAuth callback: decode token and fetch user from DB."""
    user_id = getattr(token, "sub", None)
    if user_id is None and callable(getattr(token, "get", None)):
        user_id = token.get("sub")
    if not user_id:
        return None

    # Tests inject session via app.state; prod creates a fresh one.
    db = getattr(connection.app.state, "test_db_session", None)
    if db is None:
        from app.database import async_session_factory

        async with async_session_factory() as db:
            result = await db.execute(
                select(User).where(User.id == user_id, User.is_active.is_(True))
            )
            return result.scalar_one_or_none()
    result = await db.execute(select(User).where(User.id == user_id, User.is_active.is_(True)))
    return result.scalar_one_or_none()


async def get_current_user(request: Request, db_session: AsyncSession) -> User:
    """Return the currently authenticated user.

    Used as a dependency provider for routes that need the user object.
    When APP_SKIP_AUTH=true, returns the first active user.
    """
    if get_settings().app.skip_auth:
        result = await db_session.execute(
            select(User).where(User.is_active.is_(True)).order_by(User.created_at).limit(1)
        )
        user = result.scalar_one_or_none()
        if user is None:
            raise NotAuthorizedException("No active users in database. Create one first.")
        return user

    # Normal JWT flow: request.user is set by JWTAuth middleware
    user = request.user
    if user is None or not getattr(user, "is_active", False):
        raise NotAuthorizedException("Could not validate credentials")
    return user


# Type alias for clean injection in route signatures
CurrentUser = Annotated[User, Dependency()]
DbDep = Annotated[AsyncSession, Dependency()]
