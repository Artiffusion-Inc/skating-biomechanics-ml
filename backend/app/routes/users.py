"""User API routes: profile and settings."""

from fastapi import APIRouter

from app.auth.deps import CurrentUser, DbDep
from app.crud.user import update
from app.schemas import (
    UpdateOnboardingRoleRequest,
    UpdateProfileRequest,
    UpdateSettingsRequest,
    UserResponse,
)

router = APIRouter(tags=["users"])


@router.get("/me", response_model=UserResponse)
async def get_me(user: CurrentUser):
    """Get current user profile."""
    return user


@router.patch("/me", response_model=UserResponse)
async def update_profile(body: UpdateProfileRequest, user: CurrentUser, db: DbDep):
    """Update current user profile."""
    updated = await update(
        db,
        user,
        display_name=body.display_name,
        bio=body.bio,
        height_cm=body.height_cm,
        weight_kg=body.weight_kg,
    )
    return updated


@router.patch("/me/settings", response_model=UserResponse)
async def update_settings(body: UpdateSettingsRequest, user: CurrentUser, db: DbDep):
    """Update current user preferences."""
    updated = await update(
        db,
        user,
        language=body.language,
        timezone=body.timezone,
        theme=body.theme,
    )
    return updated


@router.patch("/me/onboarding", response_model=UserResponse)
async def update_onboarding_role(body: UpdateOnboardingRoleRequest, user: CurrentUser, db: DbDep):
    """Update user's onboarding role."""
    updated = await update(db, user, onboarding_role=body.onboarding_role)
    return updated
