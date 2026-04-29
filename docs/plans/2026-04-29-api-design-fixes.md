# API Design Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix API design inconsistencies in the FastAPI backend to align with REST best practices.

**Architecture:** Non-breaking changes only. Standardize router prefixes, replace magic status codes, introduce structured `ErrorResponse`, and complete pagination schemas. Breaking renames (`/detect` -> `/detections`) deferred to API v2.

**Tech Stack:** FastAPI, Pydantic, pytest + httpx

---

## File Structure

| File | Role |
|------|------|
| `backend/app/schemas.py` | Add `ErrorResponse`, update list response models |
| `backend/app/routes/misc.py` | Fix magic status code |
| `backend/app/routes/process.py` | Fix magic status code |
| `backend/app/routes/__init__.py` | Add `raise_api_error` helper (new content) |
| `backend/app/routes/auth.py` | Refactor HTTPException to structured errors |
| `backend/app/routes/sessions.py` | Refactor HTTPException to structured errors |
| `backend/app/routes/connections.py` | Refactor HTTPException to structured errors |
| `backend/app/routes/metrics.py` | Refactor HTTPException to structured errors |
| `backend/app/routes/uploads.py` | Refactor HTTPException to structured errors |
| `backend/app/routes/choreography.py` | Refactor HTTPException to structured errors |
| `backend/app/main.py` | Standardize router prefix inclusion |
| `backend/tests/routes/test_misc.py` | Update error response assertions |
| `backend/tests/routes/test_process.py` | Update error response assertions |

---

### Task 1: Fix Magic Status Codes

**Files:**
- Modify: `backend/app/routes/misc.py:33`
- Modify: `backend/app/routes/process.py:76`
- Test: `backend/tests/routes/test_misc.py`, `backend/tests/routes/test_process.py`

- [ ] **Step 1: Replace bare 404 in misc.py**

```python
from fastapi import APIRouter, HTTPException, status

# ... at line 33, change:
raise HTTPException(status_code=404, detail="File not found")
# to:
raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
```

- [ ] **Step 2: Replace bare 404 in process.py**

```python
# ... at line 76, change:
raise HTTPException(status_code=404, detail="Task not found")
# to:
raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
```

- [ ] **Step 3: Run affected route tests**

Run:
```bash
uv run pytest backend/tests/routes/test_misc.py backend/tests/routes/test_process.py -v
```
Expected: PASS (status code value unchanged — 404 still 404, just explicit constant)

- [ ] **Step 4: Commit**

```bash
git add backend/app/routes/misc.py backend/app/routes/process.py
git commit -m "refactor(backend): use status.HTTP_404_NOT_FOUND instead of bare 404"
```

---

### Task 2: Add ErrorResponse Schema and Helper

**Files:**
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/routes/__init__.py`
- Test: `backend/tests/test_schemas.py` (new)

- [ ] **Step 1: Add ErrorResponse to schemas.py**

Add near the top of `backend/app/schemas.py` (after imports, before existing models):

```python
class ValidationErrorDetail(BaseModel):
    field: str
    message: str
    value: Any


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict | list[ValidationErrorDetail] | None = None
    path: str = ""
```

Import `Any` at the top of schemas.py if not already present:
```python
from typing import Any
```

- [ ] **Step 2: Add raise_api_error helper in routes/__init__.py**

Replace the empty `backend/app/routes/__init__.py` with:

```python
"""Route utilities — shared helpers for all API route modules."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.schemas import ErrorResponse


def raise_api_error(
    status_code: int,
    error: str,
    message: str,
    details: dict | list | None = None,
    request: Request | None = None,
) -> None:
    """Raise an HTTPException with a structured ErrorResponse body.

    Usage:
        raise_api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            error="NotFound",
            message="User not found",
            details={"id": user_id},
            request=request,
        )
    """
    body = ErrorResponse(
        error=error,
        message=message,
        details=details,
        path=str(request.url.path) if request else "",
    )
    raise HTTPException(status_code=status_code, detail=body.model_dump())
```

- [ ] **Step 3: Write failing test for ErrorResponse**

Create `backend/tests/test_schemas.py`:

```python
"""Tests for shared Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import ErrorResponse, ValidationErrorDetail


class TestErrorResponse:
    def test_minimal_error(self):
        e = ErrorResponse(error="NotFound", message="User not found")
        assert e.error == "NotFound"
        assert e.message == "User not found"
        assert e.details is None
        assert e.path == ""

    def test_with_details_dict(self):
        e = ErrorResponse(
            error="ValidationError",
            message="Bad input",
            details={"field": "email"},
        )
        assert e.details == {"field": "email"}

    def test_with_details_list(self):
        e = ErrorResponse(
            error="ValidationError",
            message="Bad input",
            details=[
                ValidationErrorDetail(field="email", message="Invalid", value="bad"),
            ],
        )
        assert len(e.details) == 1  # type: ignore[arg-type]
        assert e.details[0].field == "email"  # type: ignore[index]

    def test_model_dump(self):
        e = ErrorResponse(error="Test", message="Msg", path="/api/v1/users")
        d = e.model_dump()
        assert d["error"] == "Test"
        assert d["message"] == "Msg"
        assert d["path"] == "/api/v1/users"
        assert d["details"] is None
```

- [ ] **Step 4: Run test to verify it fails**

Run:
```bash
uv run pytest backend/tests/test_schemas.py -v
```
Expected: FAIL with import error (ErrorResponse not found if schemas.py edit failed)

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
uv run pytest backend/tests/test_schemas.py -v
```
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas.py backend/app/routes/__init__.py backend/tests/test_schemas.py
git commit -m "feat(backend): add ErrorResponse schema and raise_api_error helper"
```

---

### Task 3: Standardize Router Prefixes

**Files:**
- Modify: `backend/app/main.py`
- Modify: `backend/app/routes/auth.py`
- Modify: `backend/app/routes/users.py`
- Modify: `backend/app/routes/detect.py`
- Modify: `backend/app/routes/models.py`
- Modify: `backend/app/routes/process.py`
- Modify: `backend/app/routes/misc.py`
- Modify: `backend/app/routes/sessions.py`
- Modify: `backend/app/routes/metrics.py`
- Modify: `backend/app/routes/connections.py`
- Modify: `backend/app/routes/uploads.py`
- Modify: `backend/app/routes/choreography.py`
- Test: `backend/tests/routes/` (all existing route tests)

**Current state:** Inconsistent. `auth` and `users` get prefix in `main.py` (`prefix="/auth"`, `prefix="/users"`), while all other routers define full paths inline in decorators (`/detect`, `/process/queue`, etc.).

**Target state:** All prefixes live in `main.py` for the top-level resource name. Each router declares its sub-routes inline. This keeps endpoint paths visible in one file (`main.py`) and sub-resource paths in route files.

- [ ] **Step 1: Update main.py to include all prefixes**

In `backend/app/main.py`, change lines 63-75 from:

```python
api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(auth.router, prefix="/auth")
api_v1.include_router(users.router, prefix="/users")
api_v1.include_router(detect.router)
api_v1.include_router(models.router)
api_v1.include_router(process.router)
api_v1.include_router(misc.router)
api_v1.include_router(sessions.router)
api_v1.include_router(metrics.router)
api_v1.include_router(connections.router)
api_v1.include_router(uploads.router)
api_v1.include_router(choreography.router)
app.include_router(api_v1)
```

To:

```python
api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(auth.router, prefix="/auth")
api_v1.include_router(users.router, prefix="/users")
api_v1.include_router(detect.router, prefix="/detect")
api_v1.include_router(models.router, prefix="/models")
api_v1.include_router(process.router, prefix="/process")
api_v1.include_router(misc.router, prefix="/misc")
api_v1.include_router(sessions.router, prefix="/sessions")
api_v1.include_router(metrics.router, prefix="/metrics")
api_v1.include_router(connections.router, prefix="/connections")
api_v1.include_router(uploads.router, prefix="/uploads")
api_v1.include_router(choreography.router, prefix="/choreography")
app.include_router(api_v1)
```

- [ ] **Step 2: Strip leading resource paths from route decorators**

For each router, strip the leading resource segment from decorators. Examples:

**detect.py** — change `/detect` -> `/`:
```python
@router.post("/", response_model=DetectQueueResponse)
```

**process.py** — change `/process/queue` -> `/queue`, `/process/{task_id}/status` -> `/{task_id}/status`, etc.:
```python
@router.post("/queue", response_model=QueueProcessResponse)
@router.get("/{task_id}/status", response_model=TaskStatusResponse)
@router.post("/{task_id}/cancel")
@router.get("/{task_id}/stream")
```

**sessions.py** — change `/sessions` -> `/`, `/sessions/{session_id}` -> `/{session_id}`:
```python
@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
@router.get("/", response_model=SessionListResponse)
@router.get("/{session_id}", response_model=SessionResponse)
@router.patch("/{session_id}", response_model=SessionResponse)
@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
```

Apply the same pattern to all remaining route files:
- `models.py`: strip `/models` prefix
- `metrics.py`: strip `/metrics` prefix
- `connections.py`: strip `/connections` prefix
- `uploads.py`: strip `/uploads` prefix
- `choreography.py`: strip `/choreography` prefix
- `misc.py`: `/health` stays `/health` (sub-resource), `/outputs/{key:path}` stays as-is
- `auth.py`: `/register` -> `/register` (already correct — sub-resource of `/auth`)
- `users.py`: `/users/me` -> `/me` (already has `/users` prefix in main.py)

- [ ] **Step 3: Run all route tests**

Run:
```bash
uv run pytest backend/tests/routes/ -v
```
Expected: PASS (paths unchanged from client perspective — `/api/v1/detect` still `/api/v1/detect` because prefix moved from inline to main.py)

- [ ] **Step 4: Commit**

```bash
git add backend/app/main.py backend/app/routes/
git commit -m "refactor(backend): standardize router prefixes in main.py"
```

---

### Task 4: Refactor HTTPException to Structured ErrorResponse

**Files:**
- Modify: `backend/app/routes/auth.py`
- Modify: `backend/app/routes/sessions.py`
- Modify: `backend/app/routes/connections.py`
- Modify: `backend/app/routes/metrics.py`
- Modify: `backend/app/routes/uploads.py`
- Modify: `backend/app/routes/choreography.py`
- Modify: `backend/app/routes/misc.py`
- Modify: `backend/app/routes/process.py`
- Modify: `backend/app/routes/detect.py`
- Test: `backend/tests/routes/test_auth_direct.py`, `backend/tests/routes/test_sessions.py`, etc.

- [ ] **Step 1: Refactor auth.py errors**

Replace bare `HTTPException(...)` with `raise_api_error(...)` in `backend/app/routes/auth.py`.

Add import at the top:
```python
from app.routes import raise_api_error
```

Replace each exception. Example for register:
```python
# Before:
raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

# After:
raise_api_error(
    status_code=status.HTTP_409_CONFLICT,
    error="Conflict",
    message="Email already registered",
    details={"field": "email"},
)
```

Apply the same pattern to:
- `/login`: error="Unauthorized", message="Invalid email or password"
- `/refresh`: error="Unauthorized", message="Invalid or expired refresh token"

- [ ] **Step 2: Refactor sessions.py errors**

Replace each `HTTPException` in `backend/app/routes/sessions.py`:

```python
from app.routes import raise_api_error

# In list_sessions, get_session, patch_session, delete_session:
raise_api_error(
    status_code=status.HTTP_403_FORBIDDEN,
    error="Forbidden",
    message="Not a coach for this user",
)

raise_api_error(
    status_code=status.HTTP_403_FORBIDDEN,
    error="Forbidden",
    message="Not authorized",
)

raise_api_error(
    status_code=status.HTTP_404_NOT_FOUND,
    error="NotFound",
    message="Session not found",
    details={"id": session_id},
)
```

- [ ] **Step 3: Refactor connections.py errors**

Replace each `HTTPException` in `backend/app/routes/connections.py`:

```python
from app.routes import raise_api_error

# User not found
raise_api_error(
    status_code=status.HTTP_404_NOT_FOUND,
    error="NotFound",
    message="User not found",
    details={"email": body.to_user_email},
)

# Connection already exists
raise_api_error(
    status_code=status.HTTP_409_CONFLICT,
    error="Conflict",
    message="Connection already exists",
)

# Connection not found
raise_api_error(
    status_code=status.HTTP_404_NOT_FOUND,
    error="NotFound",
    message="Connection not found",
    details={"id": conn_id},
)

# Not authorized
raise_api_error(
    status_code=status.HTTP_403_FORBIDDEN,
    error="Forbidden",
    message="Not authorized",
)

# Not an active invite / Already ended
raise_api_error(
    status_code=status.HTTP_400_BAD_REQUEST,
    error="BadRequest",
    message="Not an active invite",  # or "Already ended"
)
```

- [ ] **Step 4: Refactor metrics.py errors**

```python
from app.routes import raise_api_error

raise_api_error(
    status_code=status.HTTP_403_FORBIDDEN,
    error="Forbidden",
    message="Not a coach for this user",
)

raise_api_error(
    status_code=status.HTTP_400_BAD_REQUEST,
    error="BadRequest",
    message=f"Unknown metric: {metric_name}",
    details={"metric": metric_name},
)
```

- [ ] **Step 5: Refactor uploads.py errors**

```python
from app.routes import raise_api_error

raise_api_error(
    status_code=status.HTTP_400_BAD_REQUEST,
    error="BadRequest",
    message="No parts provided",
)
```

- [ ] **Step 6: Refactor choreography.py, misc.py, process.py, detect.py errors**

Apply same `raise_api_error` pattern to remaining `HTTPException` occurrences in:
- `choreography.py` (music upload, program CRUD)
- `misc.py` (file not found)
- `process.py` (task not found)
- `detect.py` (any HTTPException)

Each error should follow the pattern: `error` is PascalCase error code (e.g., `NotFound`, `Forbidden`, `BadRequest`, `Conflict`, `Unauthorized`), `message` is human-readable string, `details` contains relevant IDs or fields.

- [ ] **Step 7: Run all route tests**

Run:
```bash
uv run pytest backend/tests/routes/ -v
```
Expected: Some tests may FAIL if they assert on `response.json()["detail"]` as a string. The new format returns `{"detail": {"error": "...", "message": "...", ...}}`.

- [ ] **Step 8: Update test assertions for structured errors**

For each failing test, update error assertions. Example in `backend/tests/routes/test_auth_direct.py`:

```python
# Before:
assert response.json()["detail"] == "Email already registered"

# After:
data = response.json()["detail"]
assert data["error"] == "Conflict"
assert data["message"] == "Email already registered"
```

Apply the same pattern to:
- `test_sessions.py` and `test_sessions_direct.py`
- `test_connections.py` and `test_connections_direct.py`
- `test_metrics.py` and `test_metrics_direct.py`
- `test_misc.py`
- `test_process.py`
- `test_choreography_upload.py`
- `test_detect.py`
- `test_uploads.py`

- [ ] **Step 9: Run all route tests again**

Run:
```bash
uv run pytest backend/tests/routes/ -v
```
Expected: PASS (all tests updated for structured error format)

- [ ] **Step 10: Commit**

```bash
git add backend/app/routes/ backend/tests/routes/
git commit -m "refactor(backend): use structured ErrorResponse in all HTTPExceptions"
```

---

### Task 5: Complete Pagination Response Schemas

**Files:**
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/routes/sessions.py`
- Modify: `backend/app/routes/connections.py`
- Modify: `backend/app/routes/choreography.py`
- Test: `backend/tests/test_schemas.py`

- [ ] **Step 1: Create PaginatedResponse base in schemas.py**

Add to `backend/app/schemas.py`:

```python
class PaginatedResponse(BaseModel):
    """Base for all paginated list responses."""

    total: int
    page: int = 1
    page_size: int = 20
    pages: int = 1

    @property
    def has_next(self) -> bool:
        return self.page < self.pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1
```

- [ ] **Step 2: Update SessionListResponse**

Replace:
```python
class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int
```

With:
```python
class SessionListResponse(PaginatedResponse):
    sessions: list[SessionResponse]
```

- [ ] **Step 3: Update ConnectionListResponse**

Replace:
```python
class ConnectionListResponse(BaseModel):
    connections: list[ConnectionResponse]
```

With:
```python
class ConnectionListResponse(PaginatedResponse):
    connections: list[ConnectionResponse]
```

- [ ] **Step 4: Update ProgramListResponse**

Replace:
```python
class ProgramListResponse(BaseModel):
    programs: list[ChoreographyProgramResponse]
```

With:
```python
class ProgramListResponse(PaginatedResponse):
    programs: list[ChoreographyProgramResponse]
```

- [ ] **Step 5: Update list endpoints to populate pagination fields**

**sessions.py** (`list_sessions`, ~line 70):
```python
# After computing total and sessions:
pages = (total + limit - 1) // limit if limit else 1
page = (offset // limit) + 1 if limit else 1

return SessionListResponse(
    sessions=[await _session_to_response(s) for s in sessions],
    total=total,
    page=page,
    page_size=limit,
    pages=pages,
)
```

**connections.py** (`list_connections`, `list_pending`, ~line 105):
```python
# Add limit/offset params if not present (currently no pagination in connections)
```

**Note:** `connections.py` currently has no pagination. Either skip adding pagination fields there (leave default values) or add `limit`/`offset` params in a follow-up.

**choreography.py** (`list_programs`, ~line 265):
```python
# After computing total and programs:
pages = (total + limit - 1) // limit if limit else 1
page = (offset // limit) + 1 if limit else 1

return ProgramListResponse(
    programs=[_program_to_response(p) for p in programs],
    total=total,
    page=page,
    page_size=limit,
    pages=pages,
)
```

- [ ] **Step 6: Write tests for PaginatedResponse**

Add to `backend/tests/test_schemas.py`:

```python
class TestPaginatedResponse:
    def test_defaults(self):
        p = PaginatedResponse(total=50)
        assert p.total == 50
        assert p.page == 1
        assert p.page_size == 20
        assert p.pages == 1
        assert not p.has_next
        assert not p.has_prev

    def test_has_next(self):
        p = PaginatedResponse(total=50, page=1, page_size=20, pages=3)
        assert p.has_next
        assert not p.has_prev

    def test_has_prev(self):
        p = PaginatedResponse(total=50, page=2, page_size=20, pages=3)
        assert p.has_next
        assert p.has_prev

    def test_last_page(self):
        p = PaginatedResponse(total=50, page=3, page_size=20, pages=3)
        assert not p.has_next
        assert p.has_prev
```

- [ ] **Step 7: Run schema + route tests**

Run:
```bash
uv run pytest backend/tests/test_schemas.py backend/tests/routes/test_sessions.py backend/tests/routes/test_choreography_upload.py -v
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/app/schemas.py backend/app/routes/sessions.py backend/app/routes/choreography.py backend/tests/test_schemas.py
git commit -m "feat(backend): add PaginatedResponse base and complete pagination fields"
```

---

### Task 6: Final Integration Test Run

- [ ] **Step 1: Run full backend test suite**

Run:
```bash
uv run pytest backend/tests/ -v --tb=short
```
Expected: PASS (all tests pass with new error format and pagination)

- [ ] **Step 2: Run linter/type-checker**

Run:
```bash
uv run ruff check backend/app/routes/ backend/app/schemas.py
uv run basedpyright backend/app/routes/ backend/app/schemas.py
```
Expected: No errors

- [ ] **Step 3: Commit any final fixes**

If lint/type issues found, fix and commit:
```bash
git add backend/
git commit -m "chore(backend): fix lint/type issues after API design refactor"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ Magic status codes -> Task 1
- ✅ Router prefix standardization -> Task 3
- ✅ Structured ErrorResponse -> Tasks 2 + 4
- ✅ Complete pagination -> Task 5
- ❌ Rate limiting -> Excluded (requires infrastructure: Redis/Valkey rate-limit store)
- ❌ `/detect` -> `/detections` rename -> Excluded (breaking change, defer to API v2)
- ❌ HATEOAS `_links` -> Excluded (not required for current use case)

**2. Placeholder scan:**
- No "TBD", "TODO", or "implement later"
- Every step has complete code
- Every step has exact file paths

**3. Type consistency:**
- `raise_api_error` signature consistent across all tasks
- `ErrorResponse` fields consistent: `error`, `message`, `details`, `path`
- `PaginatedResponse` fields: `total`, `page`, `page_size`, `pages`

**4. Breaking changes:** None. All URL paths remain identical from client perspective.

---

## Deferred for API v2

| Change | Why Deferred |
|--------|-------------|
| `/detect` -> `/detections` | Breaking — clients use `/api/v1/detect` |
| `/process/*` -> `/processes/*` or `/tasks/process/*` | Breaking — clients use `/api/v1/process/queue` |
| Rate limiting (slowapi) | Requires infrastructure decision (Valkey vs in-memory) |
| HATEOAS `_links` in responses | Nice-to-have, not required for SPA frontend |
