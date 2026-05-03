"""Route module exports — each submodule provides a Litestar Router."""

from __future__ import annotations

from litestar import Router

from app.routes.auth import AuthController

auth = Router(path="/auth", route_handlers=[AuthController])
# Other routers will be added in subsequent tasks
