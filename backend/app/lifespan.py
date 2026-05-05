"""Lifespan context manager for Litestar app."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import redis.asyncio as aioredis
from arq import create_pool
from arq.connections import RedisSettings
from litestar.stores.redis import RedisStore
from litestar.stores.registry import StoreRegistry

from app.config import get_settings
from app.task_manager import close_valkey_pool, init_valkey_pool

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from litestar import Litestar


def _parse_redis_url(url: str) -> dict:
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 6379,
        "database": int((parsed.path or "/0").lstrip("/") or 0),
        "password": parsed.password or None,
    }


@asynccontextmanager
async def app_lifespan(app: Litestar) -> AsyncGenerator[None, None]:
    """Initialize and tear down shared resources."""
    settings = get_settings()

    # Valkey pool (used by task_manager module)
    await init_valkey_pool()

    # Response cache store via Litestar StoreRegistry
    url = settings.valkey.build_url()
    redis_client = aioredis.Redis.from_url(url, decode_responses=False)
    root_store = RedisStore(redis=redis_client)
    app.stores = StoreRegistry(default_factory=root_store.with_namespace)

    # arq pool for background job enqueue
    arq_cfg = _parse_redis_url(url)
    app.state.arq_pool = await create_pool(
        RedisSettings(
            host=arq_cfg["host"],
            port=arq_cfg["port"],
            database=arq_cfg["database"],
            password=arq_cfg["password"],
        )
    )

    try:
        yield
    finally:
        await app.state.arq_pool.close()
        await close_valkey_pool()
