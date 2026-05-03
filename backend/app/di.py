"""Litestar dependency providers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from litestar.di import Provide

from app.config import Settings, get_settings
from app.database import async_session_factory

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession


async def provide_settings() -> Settings:
    """Provide cached app settings."""
    return get_settings()


@asynccontextmanager
async def provide_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async SQLAlchemy session with auto-commit/rollback."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except (OSError, RuntimeError, ValueError):
            await session.rollback()
            raise


dependencies = {
    "settings": Provide(provide_settings, sync_to_thread=False),
    "db_session": Provide(provide_db_session, sync_to_thread=False),
}
