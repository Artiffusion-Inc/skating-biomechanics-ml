"""Tests for workspace ORM models."""

import pytest
from app.models.user import User
from app.models.workspace import (
    Subscription,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_create_workspace(db_session: AsyncSession):
    ws = Workspace(name="Ice Academy", slug="ice-academy")
    db_session.add(ws)
    await db_session.flush()
    await db_session.refresh(ws)

    assert ws.name == "Ice Academy"
    assert ws.slug == "ice-academy"
    assert ws.is_active is True
    assert ws.id is not None


@pytest.mark.asyncio
async def test_workspace_member_unique(db_session: AsyncSession):
    user = User(email="test@ice.com", hashed_password="hash")
    ws = Workspace(name="Ice", slug="ice")
    db_session.add_all([user, ws])
    await db_session.flush()

    m1 = WorkspaceMember(workspace_id=ws.id, user_id=user.id, role=WorkspaceRole.COACH)
    db_session.add(m1)
    await db_session.flush()

    m2 = WorkspaceMember(workspace_id=ws.id, user_id=user.id, role=WorkspaceRole.STUDENT)
    db_session.add(m2)
    with pytest.raises(IntegrityError):  # noqa: PT012
        await db_session.flush()


@pytest.mark.asyncio
async def test_subscription_placeholder(db_session: AsyncSession):
    ws = Workspace(name="Ice", slug="ice2")
    db_session.add(ws)
    await db_session.flush()

    sub = Subscription(workspace_id=ws.id)
    db_session.add(sub)
    await db_session.flush()
    await db_session.refresh(sub)

    assert sub.plan == "free"
    assert sub.status.value == "trial"
    assert sub.workspace_id == ws.id
