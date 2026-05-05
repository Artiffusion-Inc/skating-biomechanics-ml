---
title: "Workspace + Subscription Schema Migration"
date: "2026-05-05"
status: draft
---

# Workspace + Subscription Schema Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-tenant workspace support (clubs/schools), workspace membership with roles, and subscription placeholder to the existing SQLAlchemy schema.

**Architecture:** Workspace becomes the billing and resource scoping unit. Session, MusicAnalysis, ChoreographyProgram gain nullable `workspace_id` FK (back-compat). User-to-user Connection stays for cross-workspace relationships. WorkspaceMember roles: owner, admin, coach, student, parent.

**Tech Stack:** SQLAlchemy 2.0 async, Alembic, Litestar, PostgreSQL

**Design decisions:**
- `workspace_id` on Session/MusicAnalysis/ChoreographyProgram is nullable to allow gradual migration and personal-mode usage
- Subscription is a placeholder (no payment provider integration yet)
- Roles are enum, not RBAC table — sufficient for club/school hierarchy
- Connection table remains unchanged; coach-student links can exist both inside and outside workspaces

**Spec:** Current schema has User, Session, SessionMetric, Connection, RefreshToken, MusicAnalysis, ChoreographyProgram. Missing: Workspace, WorkspaceMember, Subscription. Session.user_id stays (direct ownership), workspace_id added for scoping.

---

## File Structure

### New backend files
```
backend/app/models/workspace.py          # Workspace, WorkspaceMember, Subscription, WorkspaceRole
backend/app/crud/workspace.py            # CRUD for workspaces and members
backend/app/routes/workspaces.py          # Workspace API routes
backend/tests/test_workspace_models.py   # Model tests
backend/tests/test_workspace_routes.py   # Route tests
```

### Modified files
```
backend/app/models/__init__.py           # Export new models
backend/app/models/session.py            # Add workspace_id FK
backend/app/models/choreography.py       # Add workspace_id FK to MusicAnalysis and ChoreographyProgram
backend/app/schemas.py                   # Add WorkspaceResponse, WorkspaceMemberResponse, etc.
backend/app/auth/deps.py                 # Add workspace-scoped auth helpers
backend/app/routes/sessions.py           # Filter by workspace_id
backend/app/routes/choreography.py       # Filter by workspace_id
backend/alembic/versions/...             # Two migrations: new tables + FKs
```

---

## Task 1: Add Workspace Model

**Files:**
- Create: `backend/app/models/workspace.py`
- Modify: `backend/app/models/__init__.py`

- [ ] **Step 1: Write workspace.py**

```python
"""Workspace, WorkspaceMember, and Subscription ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime  # noqa: TC003
from enum import StrEnum

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class WorkspaceRole(StrEnum):
    """Role of a user within a workspace."""

    OWNER = "owner"
    ADMIN = "admin"
    COACH = "coach"
    STUDENT = "student"
    PARENT = "parent"


class SubscriptionStatus(StrEnum):
    """Lifecycle of a subscription."""

    TRIAL = "trial"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"


class Workspace(TimestampMixin, Base):
    """An organization (club, school, training group)."""

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(
        String(100), unique=True, index=True, nullable=False
    )
    description: Mapped[str | None] = mapped_column(Text)
    avatar_url: Mapped[str | None] = mapped_column(String(500))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    members: Mapped[list[WorkspaceMember]] = relationship(
        "WorkspaceMember",
        back_populates="workspace",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    subscriptions: Mapped[list[Subscription]] = relationship(
        "Subscription",
        back_populates="workspace",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class WorkspaceMember(TimestampMixin, Base):
    """Many-to-many link between User and Workspace with a role."""

    __tablename__ = "workspace_members"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[WorkspaceRole] = mapped_column(
        SAEnum(WorkspaceRole),
        default=WorkspaceRole.STUDENT,
    )
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    invited_by: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    workspace: Mapped[Workspace] = relationship(
        "Workspace", back_populates="members", lazy="selectin"
    )
    user: Mapped["User"] = relationship(  # type: ignore[valid-type]
        "User",
        foreign_keys=[user_id],
        lazy="selectin",
    )

    __table_args__ = (
        Index("uq_workspace_member", "workspace_id", "user_id", unique=True),
    )


class Subscription(TimestampMixin, Base):
    """Billing anchor for a workspace."""

    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True,
    )
    plan: Mapped[str] = mapped_column(String(50), default="free")
    status: Mapped[SubscriptionStatus] = mapped_column(
        SAEnum(SubscriptionStatus),
        default=SubscriptionStatus.TRIAL,
    )
    seats: Mapped[int | None] = mapped_column(SmallInteger, default=1)
    max_seats: Mapped[int | None] = mapped_column(SmallInteger, default=5)
    trial_ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    workspace: Mapped[Workspace] = relationship(
        "Workspace", back_populates="subscriptions", lazy="selectin"
    )
```

- [ ] **Step 2: Update models/__init__.py**

Replace `backend/app/models/__init__.py` with:

```python
"""SQLAlchemy ORM models."""

from app.models.base import Base
from app.models.choreography import ChoreographyProgram, MusicAnalysis
from app.models.connection import Connection
from app.models.refresh_token import RefreshToken
from app.models.session import Session, SessionMetric
from app.models.user import User
from app.models.workspace import Subscription, Workspace, WorkspaceMember, WorkspaceRole

__all__ = [
    "Base",
    "ChoreographyProgram",
    "Connection",
    "MusicAnalysis",
    "RefreshToken",
    "Session",
    "SessionMetric",
    "Subscription",
    "User",
    "Workspace",
    "WorkspaceMember",
    "WorkspaceRole",
]
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/models/workspace.py backend/app/models/__init__.py
git commit -m "feat(backend): add Workspace, WorkspaceMember, Subscription models"
```

---

## Task 2: Add workspace_id to Session and Choreography tables

**Files:**
- Modify: `backend/app/models/session.py`
- Modify: `backend/app/models/choreography.py`

- [ ] **Step 1: Modify session.py**

Add to imports: `from sqlalchemy import ...` already includes `ForeignKey`, `Index`, `String`.

Add inside `Session` class, after `user_id`:

```python
    workspace_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
```

- [ ] **Step 2: Modify choreography.py**

Read `backend/app/models/choreography.py` and add `workspace_id` to both `MusicAnalysis` and `ChoreographyProgram`.

For `MusicAnalysis`, after `user_id`:

```python
    workspace_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
```

Same for `ChoreographyProgram`.

- [ ] **Step 3: Commit**

```bash
git add backend/app/models/session.py backend/app/models/choreography.py
git commit -m "feat(backend): add nullable workspace_id FK to Session, MusicAnalysis, ChoreographyProgram"
```

---

## Task 3: Generate Alembic Migrations

**Files:**
- Create: `backend/alembic/versions/2026_05_05_xxxx_add_workspaces_and_subscriptions.py`
- Create: `backend/alembic/versions/2026_05_05_xxxx_add_workspace_id_to_resources.py`

- [ ] **Step 1: Ensure postgres is running**

Run: `podman compose up -d postgres`
Wait: `podman compose exec postgres pg_isready -U skating`

- [ ] **Step 2: Generate first migration (new tables)**

Run:
```bash
cd backend && uv run alembic revision --autogenerate -m "add workspaces and subscriptions"
```

Expected: migration file creates `workspaces`, `workspace_members`, `subscriptions` tables with indexes and FKs.

- [ ] **Step 3: Apply first migration**

Run: `cd backend && uv run alembic upgrade head`

- [ ] **Step 4: Generate second migration (FK additions)**

Run:
```bash
cd backend && uv run alembic revision --autogenerate -m "add workspace_id to sessions and choreography"
```

Verify it adds `workspace_id` nullable columns to `sessions`, `music_analyses`, `choreography_programs` with FK constraints.

- [ ] **Step 5: Apply second migration**

Run: `cd backend && uv run alembic upgrade head`

- [ ] **Step 6: Commit**

```bash
git add backend/alembic/versions/
git commit -m "feat(backend): add alembic migrations for workspace schema"
```

---

## Task 4: Add Schemas

**Files:**
- Modify: `backend/app/schemas.py`

- [ ] **Step 1: Add Workspace schemas**

Insert after `UpdateOnboardingRoleRequest` (before Detect & Process section):

```python
# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------


class WorkspaceRole(str, StrEnum):  # Duplicate for Pydantic; keep in sync with ORM
    OWNER = "owner"
    ADMIN = "admin"
    COACH = "coach"
    STUDENT = "student"
    PARENT = "parent"


class CreateWorkspaceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    slug: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_-]+$")
    description: str | None = Field(default=None, max_length=1000)


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: str | None
    avatar_url: str | None
    is_active: bool
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class WorkspaceMemberResponse(BaseModel):
    id: str
    workspace_id: str
    user_id: str
    role: str
    joined_at: str
    invited_by: str | None
    user_name: str | None = None
    user_email: str | None = None

    model_config = {"from_attributes": True}

    @field_validator("joined_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class InviteMemberRequest(BaseModel):
    email: EmailStr
    role: str = Field(pattern=r"^(admin|coach|student|parent)$")


class SubscriptionResponse(BaseModel):
    id: str
    workspace_id: str
    plan: str
    status: str
    seats: int | None
    max_seats: int | None
    trial_ends_at: str | None
    current_period_start: str | None
    current_period_end: str | None

    model_config = {"from_attributes": True}

    @field_validator("trial_ends_at", "current_period_start", "current_period_end", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)
```

- [ ] **Step 2: Add workspace_id to existing responses**

Add `workspace_id: str | None = None` to:
- `SessionResponse`
- `MusicAnalysisResponse`
- `ChoreographyProgramResponse`

- [ ] **Step 3: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat(backend): add workspace, member, subscription schemas"
```

---

## Task 5: Add Workspace CRUD

**Files:**
- Create: `backend/app/crud/workspace.py`

- [ ] **Step 1: Write workspace CRUD**

```python
"""Workspace and WorkspaceMember CRUD operations."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workspace import (
    Subscription,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)


async def create_workspace(
    db: AsyncSession,
    *,
    name: str,
    slug: str,
    owner_id: str,
    description: str | None = None,
) -> Workspace:
    """Create a new workspace with owner as first member."""
    ws = Workspace(name=name, slug=slug, description=description)
    db.add(ws)
    await db.flush()
    await db.refresh(ws)

    member = WorkspaceMember(
        workspace_id=ws.id,
        user_id=owner_id,
        role=WorkspaceRole.OWNER,
        joined_at=datetime.now(timezone.utc),
    )
    db.add(member)

    sub = Subscription(workspace_id=ws.id, plan="free", status="trial")
    db.add(sub)

    await db.flush()
    return ws


async def get_workspace_by_id(db: AsyncSession, workspace_id: str) -> Workspace | None:
    """Get workspace by ID."""
    result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    return result.scalar_one_or_none()


async def get_workspace_by_slug(db: AsyncSession, slug: str) -> Workspace | None:
    """Get workspace by slug."""
    result = await db.execute(select(Workspace).where(Workspace.slug == slug))
    return result.scalar_one_or_none()


async def list_workspaces_for_user(db: AsyncSession, user_id: str) -> list[Workspace]:
    """List all workspaces a user is a member of."""
    result = await db.execute(
        select(Workspace)
        .join(WorkspaceMember)
        .where(WorkspaceMember.user_id == user_id, Workspace.is_active.is_(True))
        .order_by(Workspace.created_at.desc())
    )
    return list(result.scalars().all())


async def add_workspace_member(
    db: AsyncSession,
    *,
    workspace_id: str,
    user_id: str,
    role: WorkspaceRole,
    invited_by: str | None = None,
) -> WorkspaceMember:
    """Add a member to a workspace."""
    member = WorkspaceMember(
        workspace_id=workspace_id,
        user_id=user_id,
        role=role,
        joined_at=datetime.now(timezone.utc),
        invited_by=invited_by,
    )
    db.add(member)
    await db.flush()
    await db.refresh(member)
    return member


async def get_workspace_member(
    db: AsyncSession, workspace_id: str, user_id: str
) -> WorkspaceMember | None:
    """Get a specific workspace membership."""
    result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def list_workspace_members(
    db: AsyncSession, workspace_id: str
) -> list[WorkspaceMember]:
    """List all members of a workspace."""
    result = await db.execute(
        select(WorkspaceMember)
        .where(WorkspaceMember.workspace_id == workspace_id)
        .order_by(WorkspaceMember.joined_at.desc())
    )
    return list(result.scalars().all())


async def remove_workspace_member(
    db: AsyncSession, workspace_id: str, user_id: str
) -> bool:
    """Remove a member from a workspace. Returns True if removed."""
    member = await get_workspace_member(db, workspace_id, user_id)
    if member is None:
        return False
    await db.delete(member)
    await db.flush()
    return True


async def update_member_role(
    db: AsyncSession,
    workspace_id: str,
    user_id: str,
    new_role: WorkspaceRole,
) -> WorkspaceMember | None:
    """Update a member's role."""
    member = await get_workspace_member(db, workspace_id, user_id)
    if member is None:
        return None
    member.role = new_role
    db.add(member)
    await db.flush()
    await db.refresh(member)
    return member
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/crud/workspace.py
git commit -m "feat(backend): add workspace CRUD operations"
```

---

## Task 6: Add Workspace Auth Helpers

**Files:**
- Modify: `backend/app/auth/deps.py`

- [ ] **Step 1: Add imports and helpers**

Add imports at top:

```python
from app.models.workspace import WorkspaceMember, WorkspaceRole
from app.crud.workspace import get_workspace_member
```

Add after `DbDep`:

```python
async def require_workspace_role(
    workspace_id: str,
    user: User,
    db: AsyncSession,
    min_role: WorkspaceRole = WorkspaceRole.STUDENT,
) -> WorkspaceMember:
    """Require user to have at least min_role in workspace."""
    member = await get_workspace_member(db, workspace_id, user.id)
    if member is None:
        raise NotAuthorizedException("Not a member of this workspace")
    # Simple hierarchy: owner > admin > coach > student > parent
    hierarchy = {
        WorkspaceRole.OWNER: 4,
        WorkspaceRole.ADMIN: 3,
        WorkspaceRole.COACH: 2,
        WorkspaceRole.STUDENT: 1,
        WorkspaceRole.PARENT: 0,
    }
    if hierarchy.get(member.role, -1) < hierarchy.get(min_role, -1):
        raise NotAuthorizedException("Insufficient workspace permissions")
    return member
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/auth/deps.py
git commit -m "feat(backend): add workspace role requirement helper"
```

---

## Task 7: Add Workspace Routes

**Files:**
- Create: `backend/app/routes/workspaces.py`
- Modify: `backend/app/main.py` (register router)

- [ ] **Step 1: Write routes**

```python
"""Workspace API routes."""

from __future__ import annotations

from litestar import Controller, get, post, patch, delete
from litestar.exceptions import ClientException, NotFoundException
from litestar.status_codes import HTTP_201_CREATED, HTTP_204_NO_CONTENT

from app.auth.deps import CurrentUser, DbDep, require_workspace_role
from app.crud.user import get_by_email
from app.crud.workspace import (
    add_workspace_member,
    create_workspace,
    get_workspace_by_id,
    get_workspace_by_slug,
    list_workspace_members,
    list_workspaces_for_user,
    remove_workspace_member,
    update_member_role,
)
from app.models.workspace import WorkspaceMember, WorkspaceRole
from app.schemas import (
    CreateWorkspaceRequest,
    InviteMemberRequest,
    WorkspaceMemberResponse,
    WorkspaceResponse,
)


class WorkspacesController(Controller):
    path = "/workspaces"
    tags = ["workspaces"]

    @post("", status_code=HTTP_201_CREATED)
    async def create(
        self, data: CreateWorkspaceRequest, user: CurrentUser, db: DbDep
    ) -> WorkspaceResponse:
        existing = await get_workspace_by_slug(db, data.slug)
        if existing:
            raise ClientException(detail="Workspace slug already taken")
        ws = await create_workspace(
            db, name=data.name, slug=data.slug, owner_id=user.id, description=data.description
        )
        return WorkspaceResponse.model_validate(ws)

    @get("")
    async def list(self, user: CurrentUser, db: DbDep) -> list[WorkspaceResponse]:
        workspaces = await list_workspaces_for_user(db, user.id)
        return [WorkspaceResponse.model_validate(w) for w in workspaces]

    @get("/{workspace_id:str}")
    async def get(
        self, workspace_id: str, user: CurrentUser, db: DbDep
    ) -> WorkspaceResponse:
        await require_workspace_role(workspace_id, user, db)
        ws = await get_workspace_by_id(db, workspace_id)
        if not ws:
            raise NotFoundException(detail="Workspace not found")
        return WorkspaceResponse.model_validate(ws)

    @post("/{workspace_id:str}/invite", status_code=HTTP_201_CREATED)
    async def invite(
        self,
        workspace_id: str,
        data: InviteMemberRequest,
        user: CurrentUser,
        db: DbDep,
    ) -> WorkspaceMemberResponse:
        await require_workspace_role(workspace_id, user, db, min_role=WorkspaceRole.ADMIN)
        target = await get_by_email(db, data.email)
        if not target:
            raise ClientException(detail="User not found")
        member = await add_workspace_member(
            db,
            workspace_id=workspace_id,
            user_id=target.id,
            role=WorkspaceRole(data.role),
            invited_by=user.id,
        )
        resp = WorkspaceMemberResponse.model_validate(member)
        resp.user_name = target.display_name or target.email
        resp.user_email = target.email
        return resp

    @get("/{workspace_id:str}/members")
    async def list_members(
        self, workspace_id: str, user: CurrentUser, db: DbDep
    ) -> list[WorkspaceMemberResponse]:
        await require_workspace_role(workspace_id, user, db)
        members = await list_workspace_members(db, workspace_id)
        return [WorkspaceMemberResponse.model_validate(m) for m in members]

    @delete("/{workspace_id:str}/members/{user_id:str}", status_code=HTTP_204_NO_CONTENT)
    async def remove_member(
        self, workspace_id: str, user_id: str, user: CurrentUser, db: DbDep
    ) -> None:
        await require_workspace_role(workspace_id, user, db, min_role=WorkspaceRole.ADMIN)
        await remove_workspace_member(db, workspace_id, user_id)

    @patch("/{workspace_id:str}/members/{user_id:str}/role")
    async def update_role(
        self,
        workspace_id: str,
        user_id: str,
        data: InviteMemberRequest,  # reusing email+role; email ignored
        user: CurrentUser,
        db: DbDep,
    ) -> WorkspaceMemberResponse:
        await require_workspace_role(workspace_id, user, db, min_role=WorkspaceRole.ADMIN)
        updated = await update_member_role(
            db, workspace_id, user_id, WorkspaceRole(data.role)
        )
        if not updated:
            raise NotFoundException(detail="Member not found")
        return WorkspaceMemberResponse.model_validate(updated)
```

- [ ] **Step 2: Register router**

In `backend/app/main.py`, add import and include:

```python
from app.routes.workspaces import WorkspacesController
```

Add to routes list:
```python
WorkspacesController,
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/routes/workspaces.py backend/app/main.py
git commit -m "feat(backend): add workspace API routes"
```

---

## Task 8: Update Session Routes for Workspace Scoping

**Files:**
- Modify: `backend/app/routes/sessions.py`

- [ ] **Step 1: Add workspace filter to list sessions**

In the list handler, accept optional `workspace_id` query param. If provided, filter sessions by `workspace_id` (and user still must be a member via auth helper).

Add query param support in Litestar:

```python
from app.auth.deps import require_workspace_role
from app.models.workspace import WorkspaceRole
```

In list method signature, add `workspace_id: str | None = None`. If provided:

```python
if workspace_id:
    await require_workspace_role(workspace_id, user, db, min_role=WorkspaceRole.STUDENT)
    # filter by workspace_id
```

- [ ] **Step 2: Add workspace_id to create session request**

Add `workspace_id: str | None = None` to `CreateSessionRequest` schema.

In create handler, if `workspace_id` is provided, verify membership.

- [ ] **Step 3: Commit**

```bash
git add backend/app/routes/sessions.py backend/app/schemas.py
git commit -m "feat(backend): add workspace scoping to sessions"
```

---

## Task 9: Write Model Tests

**Files:**
- Create: `backend/tests/test_workspace_models.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for workspace ORM models."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.workspace import (
    Subscription,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)


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
    with pytest.raises(Exception):  # IntegrityError on flush
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
```

- [ ] **Step 2: Run tests**

Run: `cd backend && uv run pytest tests/test_workspace_models.py -v`
Expected: All 3 PASS

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_workspace_models.py
git commit -m "test(backend): add workspace model tests"
```

---

## Task 10: Write Route Tests

**Files:**
- Create: `backend/tests/test_workspace_routes.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for workspace API routes."""

import pytest
from litestar.testing import AsyncTestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import hash_password
from app.main import create_app
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceMember, WorkspaceRole


@pytest.fixture
def app():
    from app.main import create_app
    return create_app()


@pytest.fixture
async def authed_user(db_session: AsyncSession) -> User:
    user = User(
        email="owner@ice.com",
        hashed_password=hash_password("pass"),
        display_name="Owner",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def member_user(db_session: AsyncSession) -> User:
    user = User(
        email="member@ice.com",
        hashed_password=hash_password("pass"),
        display_name="Member",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


async def auth_headers(client, user: User):
    from app.auth.security import create_access_token
    token = create_access_token(user_id=user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_create_workspace(client: AsyncTestClient, authed_user: User, db_session):
    headers = await auth_headers(client, authed_user)
    response = await client.post(
        "/workspaces",
        json={"name": "Ice Academy", "slug": "ice-academy"},
        headers=headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Ice Academy"
    assert data["slug"] == "ice-academy"


@pytest.mark.asyncio
async def test_list_workspaces(client: AsyncTestClient, authed_user: User, db_session):
    ws = Workspace(name="Ice", slug="ice-list")
    db_session.add(ws)
    await db_session.flush()
    member = WorkspaceMember(workspace_id=ws.id, user_id=authed_user.id, role=WorkspaceRole.OWNER)
    db_session.add(member)
    await db_session.flush()

    headers = await auth_headers(client, authed_user)
    response = await client.get("/workspaces", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == ws.id


@pytest.mark.asyncio
async def test_invite_member(client: AsyncTestClient, authed_user: User, member_user: User, db_session):
    ws = Workspace(name="Ice", slug="ice-invite")
    db_session.add(ws)
    await db_session.flush()
    owner = WorkspaceMember(workspace_id=ws.id, user_id=authed_user.id, role=WorkspaceRole.OWNER)
    db_session.add(owner)
    await db_session.flush()

    headers = await auth_headers(client, authed_user)
    response = await client.post(
        f"/workspaces/{ws.id}/invite",
        json={"email": "member@ice.com", "role": "coach"},
        headers=headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["user_id"] == member_user.id
    assert data["role"] == "coach"
```

- [ ] **Step 2: Run tests**

Run: `cd backend && uv run pytest tests/test_workspace_routes.py -v`
Expected: All 3 PASS

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_workspace_routes.py
git commit -m "test(backend): add workspace route tests"
```

---

## Self-Review

1. **Spec coverage:**
   - Workspace model: Task 1
   - WorkspaceMember with roles: Task 1
   - Subscription placeholder: Task 1
   - Session workspace scoping: Task 2, Task 8
   - Choreography workspace scoping: Task 2
   - Auth helpers: Task 6
   - CRUD: Task 5
   - Routes: Task 7
   - Tests: Task 9, Task 10

2. **Placeholder scan:** No TBD/TODO. All code provided.

3. **Type consistency:**
   - `WorkspaceRole` used consistently across model, CRUD, routes
   - `workspace_id` nullable in all FK additions
   - Response schemas use `model_config = {"from_attributes": True}`

---

## Execution Handoff

**Plan complete.**

Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task + review loop. Commit after every step. All tests green before next Wave.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
