"""Tests for Connection ORM model."""

import pytest
from app.models.connection import Connection, ConnectionStatus, ConnectionType
from app.models.user import User
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
async def two_users(db_session: AsyncSession):
    """Create two users for connection tests."""
    coach = User(email="coach@example.com", hashed_password="hashed", display_name="Coach")
    skater = User(email="skater@example.com", hashed_password="hashed", display_name="Skater")
    db_session.add(coach)
    db_session.add(skater)
    await db_session.flush()
    return coach, skater


@pytest.mark.asyncio
async def test_create_connection(db_session: AsyncSession, two_users):
    """Test creating a connection with all fields, defaults (status=INVITED)."""
    coach, skater = two_users
    conn = Connection(
        from_user_id=coach.id,
        to_user_id=skater.id,
        initiated_by=coach.id,
    )
    db_session.add(conn)
    await db_session.flush()
    await db_session.refresh(conn)

    assert conn.id is not None
    assert len(conn.id) == 36  # UUID
    assert conn.from_user_id == coach.id
    assert conn.to_user_id == skater.id
    assert conn.connection_type == ConnectionType.COACHING
    assert conn.status == ConnectionStatus.INVITED
    assert conn.initiated_by == coach.id
    assert conn.ended_at is None
    assert conn.created_at is not None


@pytest.mark.asyncio
async def test_connection_type_enum_values():
    """Test ConnectionType enum has correct values."""
    assert ConnectionType.COACHING == "coaching"
    assert ConnectionType.CHOREOGRAPHY == "choreography"


@pytest.mark.asyncio
async def test_connection_status_enum_values():
    """Test ConnectionStatus enum has correct values."""
    assert ConnectionStatus.INVITED == "invited"
    assert ConnectionStatus.ACTIVE == "active"
    assert ConnectionStatus.ENDED == "ended"


@pytest.mark.asyncio
async def test_connection_choreography_type(db_session: AsyncSession, two_users):
    """Test connection works with CHOREOGRAPHY type."""
    coach, skater = two_users
    conn = Connection(
        from_user_id=coach.id,
        to_user_id=skater.id,
        connection_type=ConnectionType.CHOREOGRAPHY,
        initiated_by=coach.id,
    )
    db_session.add(conn)
    await db_session.flush()
    await db_session.refresh(conn)

    assert conn.connection_type == ConnectionType.CHOREOGRAPHY
    assert conn.status == ConnectionStatus.INVITED


@pytest.mark.asyncio
async def test_connection_read_back(db_session: AsyncSession, two_users):
    """Test persist and read back via SQLAlchemy select."""
    coach, skater = two_users
    conn = Connection(
        from_user_id=coach.id,
        to_user_id=skater.id,
        initiated_by=coach.id,
    )
    db_session.add(conn)
    await db_session.flush()

    result = await db_session.execute(select(Connection).where(Connection.from_user_id == coach.id))
    fetched = result.scalar_one()

    assert fetched.id == conn.id
    assert fetched.from_user_id == coach.id
    assert fetched.to_user_id == skater.id
    assert fetched.connection_type == ConnectionType.COACHING
    assert fetched.status == ConnectionStatus.INVITED
    assert fetched.initiated_by == coach.id
