"""Tests for auth API routes."""

import pytest
from app.auth.security import hash_password
from app.models.user import User
from sqlalchemy.ext.asyncio import AsyncSession


async def test_register(client, db_session: AsyncSession):
    """Test successful registration."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "new@example.com", "password": "securepass123"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

    # Verify user was created
    from sqlalchemy import select

    result = await db_session.execute(select(User).where(User.email == "new@example.com"))
    user = result.scalar_one_or_none()
    assert user is not None
    assert user.hashed_password != "securepass123"


async def test_register_duplicate_email(client, db_session: AsyncSession):
    """Test registration with duplicate email returns 409."""
    user = User(email="exists@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()

    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "exists@example.com", "password": "securepass123"},
    )
    assert response.status_code == 409


async def test_register_short_password(client):
    """Test registration with short password returns 400 (Litestar validation)."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "new@example.com", "password": "short"},
    )
    assert response.status_code == 400


async def test_login(client, db_session: AsyncSession):
    """Test successful login."""
    user = User(email="login@example.com", hashed_password=hash_password("pass123"))
    db_session.add(user)
    await db_session.flush()

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": "pass123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


async def test_login_wrong_password(client, db_session: AsyncSession):
    """Test login with wrong password returns 401."""
    user = User(email="login@example.com", hashed_password=hash_password("correct"))
    db_session.add(user)
    await db_session.flush()

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": "wrong"},
    )
    assert response.status_code == 401


async def test_login_nonexistent_email(client):
    """Test login with nonexistent email returns 401."""
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "pass123"},
    )
    assert response.status_code == 401


async def test_refresh_tokens(client, db_session: AsyncSession):
    """Test refresh token rotation."""
    user = User(email="refresh@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    # Login to get tokens
    login_resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "refresh@example.com", "password": "pass"},
    )
    tokens = login_resp.json()
    old_refresh = tokens["refresh_token"]

    # Use refresh token
    refresh_resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": old_refresh},
    )
    assert refresh_resp.status_code == 200
    new_tokens = refresh_resp.json()
    assert new_tokens["refresh_token"] != old_refresh
    assert "access_token" in new_tokens

    # Old refresh token should be revoked
    second_refresh = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": old_refresh},
    )
    assert second_refresh.status_code == 401


async def test_refresh_with_completely_unknown_token(client):
    """Test refresh with a token that was never issued returns 401."""
    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "a" * 64},
    )
    assert response.status_code == 401
    assert "Invalid or expired" in response.json()["message"]


async def test_logout_with_valid_token(client, db_session: AsyncSession):
    """Test logout revokes the refresh token."""
    user = User(email="logout@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    # Login to get tokens
    login_resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "logout@example.com", "password": "pass"},
    )
    tokens = login_resp.json()
    refresh_token = tokens["refresh_token"]

    # Logout
    logout_resp = await client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": refresh_token},
    )
    assert logout_resp.status_code == 204

    # Refresh should now fail (token revoked)
    refresh_resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_resp.status_code == 401


async def test_logout_with_nonexistent_token(client):
    """Test logout with a token that was never issued returns 204 (idempotent)."""
    response = await client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": "b" * 64},
    )
    assert response.status_code == 204
