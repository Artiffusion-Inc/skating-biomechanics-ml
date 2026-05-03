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

settings = get_settings()


async def retrieve_user_handler(token: dict, connection) -> User | None:
    """Litestar JWTAuth callback: decode token dict and fetch user from DB.

    Litestar JWTAuth decodes the JWT and passes the payload dict here.
    """
    user_id = token.get("sub")
    if not user_id:
        return None

    # We need a DB session — get it from the connection scope state
    db: AsyncSession = connection.scope.get("db_session")
    if db is None:
        return None

    result = await db.execute(select(User).where(User.id == user_id, User.is_active.is_(True)))
    return result.scalar_one_or_none()


async def get_current_user(request: Request, db_session: AsyncSession) -> User:
    """Return the currently authenticated user.

    Used as a dependency provider for routes that need the user object.
    When APP_SKIP_AUTH=true, returns the first active user.
    """
    if settings.app.skip_auth:
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
