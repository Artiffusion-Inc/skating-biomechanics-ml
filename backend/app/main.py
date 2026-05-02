"""FastAPI application for the figure skating biomechanics web UI."""

from __future__ import annotations

from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import structlog
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

from app.config import get_settings
from app.logging_config import configure_logging
from app.rate_limit import limiter
from app.routes import (
    auth,
    choreography,
    connections,
    detect,
    metrics,
    misc,
    models,
    process,
    sessions,
    uploads,
    users,
)
from app.task_manager import close_valkey_pool, init_valkey_pool

configure_logging()
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_valkey_pool()

    # Initialize response cache backend
    redis = aioredis.from_url(settings.valkey.build_url())
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

    # arq pool singleton for job enqueue
    app.state.arq_pool = await create_pool(
        RedisSettings(
            host=settings.valkey.host,
            port=settings.valkey.port,
            database=settings.valkey.db,
            password=settings.valkey.password.get_secret_value(),
        )
    )
    yield
    await app.state.arq_pool.close()
    await close_valkey_pool()


app = FastAPI(
    title="AI Тренер — Фигурное катание",
    description="FastAPI backend for figure skating biomechanics analysis. Auth, sessions, metrics, uploads, detection, processing, and choreography planning.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(auth.router, prefix="/auth")
api_v1.include_router(users.router, prefix="/users")
api_v1.include_router(detect.router, prefix="/detect")
api_v1.include_router(models.router, prefix="/models")
api_v1.include_router(process.router, prefix="/process")
api_v1.include_router(misc.router)
api_v1.include_router(sessions.router, prefix="/sessions")
api_v1.include_router(metrics.router, prefix="/metrics")
api_v1.include_router(connections.router, prefix="/connections")
api_v1.include_router(uploads.router, prefix="/uploads")
api_v1.include_router(choreography.router, prefix="/choreography")
app.include_router(api_v1)
