"""Route module exports — each submodule provides a Litestar Router."""

from __future__ import annotations

from litestar import Router

from app.routes.auth import AuthController
from app.routes.users import UsersController

auth = Router(path="/auth", route_handlers=[AuthController])
users = Router(path="/users", route_handlers=[UsersController])
# Other routers will be added in subsequent tasks
