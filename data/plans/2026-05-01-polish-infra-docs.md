# Frontend Polish, Backend Infra & Documentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hip/trunk angle arcs to 2D skeleton overlay, build side-by-side session comparison, add rate limiting and response caching to backend, and improve documentation (backend README, OpenAPI, ROADMAP update).

**Architecture:** Feature 1 extends existing `drawAngleArc` in `skeleton-canvas.tsx` with hip and trunk angles. Feature 2 creates a new comparison page/route that renders two `VideoWithSkeleton` components side-by-side using `useSession` hooks. Feature 3 adds `slowapi` rate limiting to auth/upload endpoints and `fastapi-cache` for expensive GET endpoints (sessions list, metrics trends). Feature 4 writes a backend README and customizes FastAPI OpenAPI metadata.

**Tech Stack:** Next.js 16, React 19, Canvas 2D API, FastAPI, slowapi, fastapi-cache, Valkey (Redis)

---

## File Map

| File | Responsibility |
|------|---------------|
| `frontend/src/components/analysis/skeleton-canvas.tsx` | Add hip angle arcs and trunk lean arc drawing. |
| `frontend/src/app/(app)/compare/page.tsx` | NEW. Side-by-side session comparison page. |
| `frontend/src/components/session/session-comparison.tsx` | NEW. Two `VideoWithSkeleton` instances with session selector. |
| `backend/app/rate_limit.py` | NEW. slowapi Limiter setup and rate limit rules. |
| `backend/app/main.py` | Wire rate limiting and cache middleware. |
| `backend/app/routes/sessions.py` | Add `@cache` decorator to list endpoint. |
| `backend/app/routes/metrics.py` | Add `@cache` decorator to trend endpoint. |
| `backend/README.md` | NEW. Backend documentation. |
| `backend/app/main.py` | Update FastAPI title/description/OpenAPI tags. |
| `ROADMAP.md` | Update status and add new milestones. |

---

## Dependencies

Backend:
```bash
uv add slowapi fastapi-cache2
```

---

### Task 1: Joint Angle Arcs (Hip + Trunk)

**Files:**

- Modify: `frontend/src/components/analysis/skeleton-canvas.tsx`

- [ ] **Step 1: Add hip angle arcs**

After the existing knee angle arc blocks (lines ~158-185), add right and left hip angles:

```typescript
// Right hip angle: shoulder -> hip -> knee
const rShoulderHip = pose[14]
const rHipJoint = pose[1]
const rKneeJoint = pose[2]
if (rShoulderHip && rHipJoint && rKneeJoint) {
  drawAngleArc(
    ctx,
    [rShoulderHip[0], rShoulderHip[1]],
    [rHipJoint[0], rHipJoint[1]],
    [rKneeJoint[0], rKneeJoint[1]],
    width,
    height,
  )
}

// Left hip angle
const lShoulderHip = pose[11]
const lHipJoint = pose[4]
const lKneeJoint = pose[5]
if (lShoulderHip && lHipJoint && lKneeJoint) {
  drawAngleArc(
    ctx,
    [lShoulderHip[0], lShoulderHip[1]],
    [lHipJoint[0], lHipJoint[1]],
    [lKneeJoint[0], lKneeJoint[1]],
    width,
    height,
  )
}
```

- [ ] **Step 2: Add trunk lean arc**

After hip arcs, add trunk lean visualization:

```typescript
// Trunk lean: hip -> spine -> neck
const hipCenter = pose[0]
const spineCenter = pose[7]
const neckCenter = pose[9]
if (hipCenter && spineCenter && neckCenter) {
  drawAngleArc(
    ctx,
    [hipCenter[0], hipCenter[1]],
    [spineCenter[0], spineCenter[1]],
    [neckCenter[0], neckCenter[1]],
    width,
    height,
  )
}
```

- [ ] **Step 3: Verify**

```bash
cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check src/components/analysis/skeleton-canvas.tsx
```

- [ ] **Step 4: Commit**

```bash
git add src/components/analysis/skeleton-canvas.tsx
git commit -m "feat(frontend): draw hip and trunk angle arcs on skeleton overlay"
```

---

### Task 2: Side-by-Side Session Comparison

**Files:**

- Create: `frontend/src/app/(app)/compare/page.tsx`
- Create: `frontend/src/components/session/session-comparison.tsx`
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Create `SessionComparison` component**

```typescript
"use client"

import { useState } from "react"
import { useTranslations } from "@/i18n"
import { useSession, useSessions } from "@/lib/api/sessions"
import { VideoWithSkeleton } from "@/components/analysis/video-with-skeleton"

export function SessionComparison() {
  const t = useTranslations("compare")
  const { data: sessionsData } = useSessions()
  const [leftId, setLeftId] = useState("")
  const [rightId, setRightId] = useState("")
  const { data: leftSession } = useSession(leftId)
  const { data: rightSession } = useSession(rightId)

  const sessions = sessionsData?.sessions ?? []

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:gap-4">
        <select
          value={leftId}
          onChange={e => setLeftId(e.target.value)}
          className="rounded-lg border border-border bg-transparent px-3 py-2 text-sm flex-1"
        >
          <option value="">{t("selectLeft")}</option>
          {sessions.map(s => (
            <option key={s.id} value={s.id}>
              {s.element_type} — {new Date(s.created_at).toLocaleDateString("ru-RU")}
            </option>
          ))}
        </select>
        <select
          value={rightId}
          onChange={e => setRightId(e.target.value)}
          className="rounded-lg border border-border bg-transparent px-3 py-2 text-sm flex-1"
        >
          <option value="">{t("selectRight")}</option>
          {sessions.map(s => (
            <option key={s.id} value={s.id}>
              {s.element_type} — {new Date(s.created_at).toLocaleDateString("ru-RU")}
            </option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {leftSession?.pose_data && leftSession?.processed_video_url && (
          <VideoWithSkeleton
            videoUrl={leftSession.processed_video_url}
            poseData={leftSession.pose_data}
            totalFrames={Math.max(...leftSession.pose_data.frames)}
            fps={leftSession.pose_data.fps}
            className="rounded-xl"
          />
        )}
        {rightSession?.pose_data && rightSession?.processed_video_url && (
          <VideoWithSkeleton
            videoUrl={rightSession.processed_video_url}
            poseData={rightSession.pose_data}
            totalFrames={Math.max(...rightSession.pose_data.frames)}
            fps={rightSession.pose_data.fps}
            className="rounded-xl"
          />
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create comparison page**

```typescript
"use client"

import { SessionComparison } from "@/components/session/session-comparison"
import { useTranslations } from "@/i18n"

export default function ComparePage() {
  const t = useTranslations("compare")
  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-4">
      <h1 className="text-xl font-semibold">{t("title")}</h1>
      <SessionComparison />
    </div>
  )
}
```

- [ ] **Step 3: Add translations**

`frontend/messages/ru.json` under `"compare"`:
```json
"title": "Сравнение",
"selectLeft": "Выберите левую сессию",
"selectRight": "Выберите правую сессию"
```

`frontend/messages/en.json` under `"compare"`:
```json
"title": "Compare",
"selectLeft": "Select left session",
"selectRight": "Select right session"
```

- [ ] **Step 4: Verify**

```bash
cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check src/components/session/session-comparison.tsx 'src/app/(app)/compare/page.tsx'
```

- [ ] **Step 5: Commit**

```bash
git add src/components/session/session-comparison.tsx 'src/app/(app)/compare/page.tsx' frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(frontend): side-by-side session comparison page"
```

---

### Task 3: Rate Limiting & Caching

**Files:**

- Create: `backend/app/rate_limit.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/routes/sessions.py`
- Modify: `backend/app/routes/metrics.py`
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Install dependencies**

```bash
cd /home/michael/Github/skating-biomechanics-ml/backend && uv add slowapi fastapi-cache2
```

- [ ] **Step 2: Create rate limit setup**

`backend/app/rate_limit.py`:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, FastAPI

limiter = Limiter(key_func=get_remote_address)

def add_rate_limiting(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

- [ ] **Step 3: Wire into main.py**

In `backend/app/main.py`, after `app = FastAPI(...)`:
```python
from app.rate_limit import add_rate_limiting

add_rate_limiting(app)
```

Add import at top:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address
```

- [ ] **Step 4: Add rate limits to auth routes**

In `backend/app/routes/auth.py`, add decorators:
```python
from app.rate_limit import limiter
from fastapi import Request

@router.post("/login")
@limiter.limit("5/minute")
def login(request: Request, ...):
    ...

@router.post("/register")
@limiter.limit("3/minute")
def register(request: Request, ...):
    ...
```

- [ ] **Step 5: Add cache to expensive endpoints**

In `backend/app/routes/sessions.py`:
```python
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache

@router.get("/")
@cache(expire=60)
def list_sessions(...):
    ...
```

In `backend/app/routes/metrics.py`:
```python
@router.get("/trend")
@cache(expire=300)
def get_trend(...):
    ...
```

- [ ] **Step 6: Initialize cache on startup**

In `backend/app/main.py`, in the `lifespan` function:
```python
from fastapi_cache.backends.redis import RedisBackend

async with lifespan:
    # existing init
    redis = aioredis.from_url(f"redis://{settings.valkey.host}:{settings.valkey.port}/{settings.valkey.db}")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
```

- [ ] **Step 7: Verify**

```bash
cd /home/michael/Github/skating-biomechanics-ml/backend && uv run pytest tests/ -q
```

- [ ] **Step 8: Commit**

```bash
git add backend/app/rate_limit.py backend/app/main.py backend/app/routes/auth.py backend/app/routes/sessions.py backend/app/routes/metrics.py backend/pyproject.toml backend/uv.lock
git commit -m "feat(backend): rate limiting and response caching"
```

---

### Task 4: Documentation

**Files:**

- Create: `backend/README.md`
- Modify: `backend/app/main.py`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Write backend README**

`backend/README.md`:
```markdown
# AI Тренер Backend

FastAPI API server for figure skating biomechanical analysis.

## Quick Start

```bash
uv sync
uv run alembic upgrade head
uv run python -m app.main
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Architecture

| Layer | Technology |
|-------|-----------|
| API | FastAPI |
| Database | PostgreSQL + SQLAlchemy |
| Queue | Valkey (Redis-compatible) + arq |
| Storage | Cloudflare R2 (S3-compatible) |
| Auth | JWT access (15min) + refresh (7d) tokens |

## Key Routes

- `POST /api/v1/auth/login` — JWT login
- `POST /api/v1/auth/register` — User registration
- `GET /api/v1/sessions` — List sessions (paginated)
- `POST /api/v1/sessions` — Create session
- `GET /api/v1/sessions/{id}` — Session detail
- `DELETE /api/v1/sessions/{id}` — Delete session
- `POST /api/v1/uploads/presigned` — Get S3 presigned URL
- `GET /api/v1/metrics/trend` — Metric trends
- `GET /api/v1/connections` — Coach-student relationships

## Testing

```bash
uv run pytest tests/ -q
```

## Environment Variables

See `app/config.py` for all settings.
```

- [ ] **Step 2: Customize OpenAPI metadata**

In `backend/app/main.py`, update:
```python
app = FastAPI(
    title="AI Тренер — API",
    description="Biomechanical analysis API for figure skating.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)
```

- [ ] **Step 3: Update ROADMAP**

Add new milestone to `ROADMAP.md` under "Version History":
```markdown
| 1.1 | 2026-05-01 | Frontend polish | Joint angle arcs, session comparison, feed filters, inline editing |
```

Update status sections for completed frontend features.

- [ ] **Step 4: Commit**

```bash
git add backend/README.md backend/app/main.py ROADMAP.md
git commit -m "docs: backend README and OpenAPI metadata"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] Hip/trunk angle arcs on skeleton overlay (Task 1)
- [x] Side-by-side session comparison (Task 2)
- [x] Rate limiting on auth endpoints (Task 3)
- [x] Response caching on expensive GETs (Task 3)
- [x] Backend README (Task 4)
- [x] OpenAPI customization (Task 4)
- [x] ROADMAP update (Task 4)

**2. Placeholder scan:** None. All code provided.

**3. Type consistency:**
- `limiter.limit("5/minute")` — slowapi standard format
- `cache(expire=60)` — fastapi-cache2 decorator format
- `VideoWithSkeleton` props match existing usage (videoUrl, poseData, totalFrames, fps)

---

## Execution Handoff

**Plan complete and saved to `data/plans/2026-05-01-polish-infra-docs.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review

**Which approach?**
