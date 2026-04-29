"""Flexible connection API routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, status

from app.auth.deps import CurrentUser, DbDep
from app.crud.connection import (
    create as create_conn,
)
from app.crud.connection import (
    get_active as get_active_conn,
)
from app.crud.connection import (
    get_by_id as get_conn_by_id,
)
from app.crud.connection import (
    list_for_user,
    list_pending_for_user,
)
from app.crud.user import get_by_email
from app.models.connection import ConnectionStatus, ConnectionType
from app.routes import raise_api_error
from app.schemas import ConnectionListResponse, ConnectionResponse, InviteRequest

if TYPE_CHECKING:
    from app.models.connection import Connection


router = APIRouter(tags=["connections"])


def _conn_to_response(conn: Connection) -> ConnectionResponse:
    data = ConnectionResponse.model_validate(conn)
    data.from_user_name = conn.from_user.display_name if conn.from_user else None
    data.to_user_name = conn.to_user.display_name if conn.to_user else None
    return data


@router.post("/invite", response_model=ConnectionResponse, status_code=status.HTTP_201_CREATED)
async def invite(body: InviteRequest, user: CurrentUser, db: DbDep):
    """User invites another user to a connection."""
    to_user = await get_by_email(db, body.to_user_email)
    if not to_user:
        raise_api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            error="NotFound",
            message="User not found",
            details={"email": body.to_user_email},
        )

    conn_type = ConnectionType(body.connection_type)
    existing = await get_active_conn(
        db, from_user_id=user.id, to_user_id=to_user.id, connection_type=conn_type
    )
    if existing:
        raise_api_error(
            status_code=status.HTTP_409_CONFLICT,
            error="Conflict",
            message="Connection already exists",
        )

    conn = await create_conn(
        db,
        from_user_id=user.id,
        to_user_id=to_user.id,
        connection_type=conn_type,
        initiated_by=user.id,
    )
    return _conn_to_response(conn)


@router.post("/{conn_id}/accept", response_model=ConnectionResponse)
async def accept_invite(conn_id: str, user: CurrentUser, db: DbDep):
    """Invitee accepts a connection."""
    conn = await get_conn_by_id(db, conn_id)
    if not conn:
        raise_api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            error="NotFound",
            message="Connection not found",
            details={"conn_id": conn_id},
        )
    if conn.to_user_id != user.id:
        raise_api_error(
            status_code=status.HTTP_403_FORBIDDEN,
            error="Forbidden",
            message="Not authorized",
            details={"conn_id": conn_id},
        )
    if conn.status != ConnectionStatus.INVITED:
        raise_api_error(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="BadRequest",
            message="Not an active invite",
            details={"conn_id": conn_id},
        )
    conn.status = ConnectionStatus.ACTIVE
    db.add(conn)
    await db.flush()
    await db.refresh(conn)
    return _conn_to_response(conn)


@router.post("/{conn_id}/end", response_model=ConnectionResponse)
async def end_connection(conn_id: str, user: CurrentUser, db: DbDep):
    """Either party ends the connection."""
    conn = await get_conn_by_id(db, conn_id)
    if not conn:
        raise_api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            error="NotFound",
            message="Connection not found",
            details={"conn_id": conn_id},
        )
    if user.id not in (conn.from_user_id, conn.to_user_id):
        raise_api_error(
            status_code=status.HTTP_403_FORBIDDEN,
            error="Forbidden",
            message="Not authorized",
            details={"conn_id": conn_id},
        )
    if conn.status == ConnectionStatus.ENDED:
        raise_api_error(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="BadRequest",
            message="Already ended",
            details={"conn_id": conn_id},
        )
    conn.status = ConnectionStatus.ENDED
    conn.ended_at = datetime.now(UTC)
    db.add(conn)
    await db.flush()
    await db.refresh(conn)
    return _conn_to_response(conn)


@router.get("", response_model=ConnectionListResponse)
async def list_connections(user: CurrentUser, db: DbDep):
    """List all connections for the current user."""
    conns = await list_for_user(db, user.id)
    return ConnectionListResponse(connections=[_conn_to_response(c) for c in conns])


@router.get("/pending", response_model=ConnectionListResponse)
async def list_pending(user: CurrentUser, db: DbDep):
    """List pending invites received by the current user."""
    conns = await list_pending_for_user(db, user.id)
    return ConnectionListResponse(connections=[_conn_to_response(c) for c in conns])
