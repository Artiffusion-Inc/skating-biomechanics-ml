"""Litestar application for the figure skating biomechanics web UI."""

from __future__ import annotations

import structlog
from litestar import Litestar, Router
from litestar.config.cors import CORSConfig
from litestar.config.response_cache import ResponseCacheConfig
from litestar.exceptions import HTTPException
from litestar.middleware.rate_limit import RateLimitConfig

from app.config import get_settings
from app.exceptions import http_exception_handler
from app.lifespan import app_lifespan
from app.logging_config import configure_logging
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

configure_logging()
logger = structlog.get_logger()


def create_app() -> Litestar:
    """Build and return the Litestar application."""
    settings = get_settings()

    # Assemble routers under /api/v1
    api_v1 = Router(
        path="/api/v1",
        route_handlers=[
            auth.router,
            users.router,
            detect.router,
            models.router,
            process.router,
            misc.router,
            sessions.router,
            metrics.router,
            connections.router,
            uploads.router,
            choreography.router,
        ],
    )

    cors_config = CORSConfig(
        allow_origins=settings.cors.origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rate_limit_config = RateLimitConfig(
        rate_limit=("minute", 60),
        exclude=["/api/v1/health", "/api/v1/docs", "/api/v1/redoc", "/api/v1/openapi.json"],
    )

    return Litestar(
        route_handlers=[api_v1],
        lifespan=[app_lifespan],
        cors_config=cors_config,
        response_cache_config=ResponseCacheConfig(default_expiration=60),
        middleware=[rate_limit_config.middleware],
        exception_handlers={HTTPException: http_exception_handler},
        debug=settings.app.log_level == "DEBUG",
        openapi_config=None,
    )


# Importable ASGI application
app = create_app()
