"""Auth-related Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    avatar_url: str | None
    bio: str | None
    height_cm: int | None
    weight_kg: float | None
    language: str
    timezone: str
    theme: str
    is_active: bool
    created_at: str

    model_config = {"from_attributes": True}


class UpdateProfileRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=100)
    bio: str | None = None
    height_cm: int | None = Field(default=None, ge=50, le=250)
    weight_kg: float | None = Field(default=None, ge=20, le=300)


class UpdateSettingsRequest(BaseModel):
    language: str | None = Field(default=None, max_length=10)
    timezone: str | None = Field(default=None, max_length=50)
    theme: str | None = Field(default=None, pattern=r"^(light|dark|system)$")
