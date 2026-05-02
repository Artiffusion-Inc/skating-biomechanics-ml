#!/usr/bin/env python3
"""Seed demo data into the local database for frontend visual testing.

Usage:
    cd backend && uv run python scripts/seed_demo.py

Creates:
    - 1 skater user (demo@skating.ai)
    - 1 coach user (coach@skating.ai)
    - 20 sessions for the skater over last 90 days
    - Biomechanical metrics per session (some PRs)
    - Coach-student connection
"""

from __future__ import annotations

import asyncio
import random
import sys
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

# Make backend package importable
sys.path.insert(0, "/home/michael/Github/skating-biomechanics-ml/backend")

from app.auth.security import hash_password
from app.database import async_session_factory
from app.metrics_registry import METRIC_REGISTRY
from app.models import (
    Connection,
    Session,
    SessionMetric,
    User,
)
from app.models.connection import ConnectionStatus, ConnectionType

SKATER_EMAIL = "demo@skating.ai"
SKATER_PASSWORD = "demo1234"
COACH_EMAIL = "coach@skating.ai"
COACH_PASSWORD = "coach1234"

ELEMENT_TYPES = ["waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"]

SESSION_TITLES_RU = {
    "waltz_jump": "Вальсовый прыжок",
    "toe_loop": "Перекидной",
    "flip": "Флип",
    "salchow": "Сальхов",
    "loop": "Петля",
    "lutz": "Лютц",
    "axel": "Аксель",
}

RECOMMENDATIONS = [
    "Увеличьте скорость разбега перед прыжком",
    "Приземление: сгибайте колени глубже, работайте над амортизацией",
    "Старайтесь удерживать корпус вертикально в полёте",
    "Руки пригибайте плотнее к корпусу для ускорения вращения",
    "Фокус на стабильность приземления — колени в одной плоскости",
    "Увеличьте время полёта за счёт более сильного отталкивания",
    "Вращение: уменьшите момент инерции, подтягивая руки быстрее",
    "Работайте над симметрией — левая сторона отстаёт",
]


def _make_datetime(days_ago: int) -> datetime:
    """Return a UTC datetime `days_ago` days back from now."""
    return datetime.now(UTC) - timedelta(days=days_ago)


def _generate_metric_value(metric_name: str, progress_factor: float) -> float:
    """Generate a metric value biased by progress_factor (0..1).

    Higher progress_factor = better values.
    """
    metric = METRIC_REGISTRY[metric_name]
    lo, hi = metric.ideal_range
    # Add noise, bias toward ideal range based on progress
    if metric.direction == "higher":
        base = lo + (hi - lo) * (0.4 + 0.5 * progress_factor)
    else:
        base = hi - (hi - lo) * (0.4 + 0.5 * progress_factor)
    noise = random.uniform(-0.15 * (hi - lo), 0.15 * (hi - lo))
    return max(lo * 0.5, min(hi * 1.3, base + noise))


async def _create_user(session: AsyncSession, email: str, password: str, name: str) -> User:
    user = User(
        email=email,
        hashed_password=hash_password(password),
        display_name=name,
        height_cm=random.choice([160, 165, 170, 175]),
        weight_kg=random.choice([50, 55, 58, 62, 65]),
        language="ru",
        timezone="Europe/Moscow",
        theme="system",
        is_active=True,
    )
    session.add(user)
    await session.flush()
    return user


async def _seed() -> None:
    async with async_session_factory() as session:
        # -- Clean old demo data ------------------------------------------------
        skater = await session.execute(sa.select(User).where(User.email == SKATER_EMAIL))
        skater = skater.scalar_one_or_none()
        if skater:
            # cascade deletes handle sessions/metrics/connections
            await session.delete(skater)
            await session.commit()

        coach = await session.execute(sa.select(User).where(User.email == COACH_EMAIL))
        coach = coach.scalar_one_or_none()
        if coach:
            await session.delete(coach)
            await session.commit()

        # -- Create users -------------------------------------------------------
        skater = await _create_user(session, SKATER_EMAIL, SKATER_PASSWORD, "Анна Скater")
        coach = await _create_user(session, COACH_EMAIL, COACH_PASSWORD, "Тренер Иван")
        await session.flush()

        # -- Create connection --------------------------------------------------
        conn = Connection(
            from_user_id=coach.id,
            to_user_id=skater.id,
            connection_type=ConnectionType.COACHING,
            status=ConnectionStatus.ACTIVE,
            initiated_by=coach.id,
        )
        session.add(conn)

        # -- Create sessions ----------------------------------------------------
        total_sessions = 20
        sessions: list[Session] = []
        for i in range(total_sessions):
            days_ago = int((i / total_sessions) * 90)  # spread over 90 days
            element = random.choice(ELEMENT_TYPES)
            progress = i / total_sessions  # 0..1 improving

            overall = min(9.8, 3.5 + progress * 5.5 + random.uniform(-0.5, 0.5))
            overall = round(overall, 2)

            recs = random.sample(RECOMMENDATIONS, k=random.randint(0, 3))

            sess = Session(
                user_id=skater.id,
                element_type=element,
                status="completed",
                overall_score=overall,
                recommendations=recs,
                processed_at=_make_datetime(days_ago),
                phases={
                    "takeoff_frame": 45,
                    "peak_frame": 62,
                    "landing_frame": 88,
                },
            )
            session.add(sess)
            sessions.append(sess)

        await session.flush()

        # -- Create session metrics ---------------------------------------------
        pr_bests: dict[str, float] = {}
        for idx, sess in enumerate(sessions):
            progress = idx / total_sessions
            applicable = [
                m for m in METRIC_REGISTRY.values() if sess.element_type in m.element_types
            ]
            for metric_def in applicable:
                val = _generate_metric_value(metric_def.name, progress)
                val = round(val, 3)

                is_pr = False
                prev_best = pr_bests.get(metric_def.name)
                if metric_def.direction == "higher":
                    if prev_best is None or val > prev_best:
                        pr_bests[metric_def.name] = val
                        is_pr = True
                elif prev_best is None or val < prev_best:
                    pr_bests[metric_def.name] = val
                    is_pr = True

                lo, hi = metric_def.ideal_range
                in_range = lo <= val <= hi

                sm = SessionMetric(
                    session_id=sess.id,
                    metric_name=metric_def.name,
                    metric_value=val,
                    is_pr=is_pr,
                    prev_best=prev_best,
                    reference_value=round((lo + hi) / 2, 3),
                    is_in_range=in_range,
                )
                session.add(sm)

        await session.commit()
        print(f"Seeded {total_sessions} sessions for {skater.display_name} ({skater.email})")
        print(f"Coach: {coach.display_name} ({coach.email})")
        print("Connection: ACTIVE coaching")
        print("Done.")


if __name__ == "__main__":
    asyncio.run(_seed())
