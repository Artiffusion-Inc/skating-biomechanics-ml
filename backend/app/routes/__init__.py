"""Route module exports — each submodule provides a Litestar Router."""

from __future__ import annotations

from litestar import Router

from app.routes.auth import AuthController
from app.routes.metrics import MetricsController
from app.routes.sessions import SessionsController
from app.routes.users import UsersController

auth = Router(path="/auth", route_handlers=[AuthController])
metrics = Router(path="/metrics", route_handlers=[MetricsController])
sessions = Router(path="/sessions", route_handlers=[SessionsController])
users = Router(path="/users", route_handlers=[UsersController])
# Other routers will be added in subsequent tasks
