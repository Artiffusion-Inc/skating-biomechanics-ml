# Upload Page Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace camera-recording upload page with file-upload-only page that accepts ZIP (from mobile app) or standalone video (MP4/MOV).

**Architecture:** Frontend unpacks ZIP via fflate (per-file, memory-safe), uploads video via existing ChunkedUploader, IMU/manifest via presigned PUT URLs (direct to R2), then creates session LAST with all keys (atomic, no race). Backend adds 3 nullable columns to Session model + 1 new presign endpoint + extends CreateSessionRequest.

**Tech Stack:** fflate, ChunkedUploader (existing), Litestar, SQLAlchemy, Alembic

---

## Task 1: Backend — Add IMU/manifest columns to Session model + schemas

**Files:**

- Modify: `backend/app/models/session.py:52`
- Modify: `backend/app/schemas.py:296-298` (CreateSessionRequest)
- Modify: `backend/app/schemas.py:301-304` (PatchSessionRequest — no change needed, intentionally exclude)
- Modify: `backend/app/schemas.py:360-383` (SessionResponse)
- Modify: `backend/app/routes/sessions.py:41-64` (_session_to_response)
- Modify: `backend/app/routes/sessions.py:71-82` (create_session handler)

- [ ] **Step 1: Add columns to Session model**

Add after `process_task_id` line (line 52) in `backend/app/models/session.py`:

```python
    imu_left_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    imu_right_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    manifest_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
```

- [ ] **Step 2: Add optional fields to CreateSessionRequest**

In `backend/app/schemas.py`, replace `CreateSessionRequest` (line 296-298):

```python
class CreateSessionRequest(BaseModel):
    element_type: str = Field(..., min_length=1, max_length=50)
    video_key: str | None = Field(default=None, max_length=500)
    imu_left_key: str | None = Field(default=None, max_length=500)
    imu_right_key: str | None = Field(default=None, max_length=500)
    manifest_key: str | None = Field(default=None, max_length=500)
```

Note: Do NOT add these fields to `PatchSessionRequest` (per review finding H6 — prevents path traversal via arbitrary key strings).

- [ ] **Step 3: Add fields to SessionResponse**

Add after `process_task_id` in `SessionResponse` (line 378):

```python
    imu_left_key: str | None = None
    imu_right_key: str | None = None
    manifest_key: str | None = None
```

- [ ] **Step 4: Update `_session_to_response` in sessions route**

Add to the dict in `backend/app/routes/sessions.py:42-63`:

```python
            "imu_left_key": session.imu_left_key,
            "imu_right_key": session.imu_right_key,
            "manifest_key": session.manifest_key,
```

- [ ] **Step 5: Update `create_session` handler to pass IMU keys**

Replace `create_session` handler in `backend/app/routes/sessions.py:71-82`:

```python
    @post("", status_code=HTTP_201_CREATED)
    async def create_session(
        self, data: CreateSessionRequest, verified_user: VerifiedUser, db: DbDep
    ) -> SessionResponse:
        session = await create(
            db,
            user_id=verified_user.id,
            element_type=data.element_type,
            video_key=data.video_key,
            imu_left_key=data.imu_left_key,
            imu_right_key=data.imu_right_key,
            manifest_key=data.manifest_key,
            status="queued" if data.video_key else "uploading",
        )
        return await _session_to_response(session)
```

This works because `crud/session.py:create()` passes `**kwargs` to `Session()` constructor (line 18).

- [ ] **Step 6: Run type check**

Run: `cd backend && uv run basedpyright app/models/session.py app/schemas.py app/routes/sessions.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/session.py backend/app/schemas.py backend/app/routes/sessions.py
git commit -m "feat(backend): add imu/manifest keys to Session model and CreateSessionRequest"
```

---

## Task 2: Backend — Generate Alembic migration

**Files:**

- Create: `backend/alembic/versions/<auto>_add_session_imu_keys.py`

- [ ] **Step 1: Generate migration**

Run: `cd backend && uv run alembic revision --autogenerate -m "add_session_imu_keys"`
Expected: New migration file created with 3 `add_column` operations

- [ ] **Step 2: Inspect migration file**

Read the generated migration. Verify it has:
- `sa.Column('imu_left_key', sa.String(500), nullable=True)`
- `sa.Column('imu_right_key', sa.String(500), nullable=True)`
- `sa.Column('manifest_key', sa.String(500), nullable=True)`

- [ ] **Step 3: Commit**

```bash
git add backend/alembic/versions/
git commit -m "chore(backend): alembic migration add_session_imu_keys"
```

---

## Task 3: Backend — Add POST /uploads/presign endpoint

**Files:**

- Modify: `backend/app/routes/uploads.py`
- Test: `backend/tests/routes/test_uploads_presign.py`

This endpoint generates a single presigned PUT URL for direct R2 upload. Used for IMU protobuf and manifest files (small files <10MB, no need for multipart).

- [ ] **Step 1: Write failing test for POST /uploads/presign**

Create `backend/tests/routes/test_uploads_presign.py`:

```python
"""Tests for the presign upload endpoint."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


async def test_presign_returns_url_and_key(client, auth_headers, authed_user):
    resp = await client.post(
        "/presign",
        headers=auth_headers,
        params={"file_name": "imu_left.pb", "content_type": "application/x-protobuf"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "url" in body
    assert "key" in body
    assert f"uploads/{authed_user.id}" in body["key"]
    assert body["key"].endswith("imu_left.pb")


async def test_presign_missing_file_name(client, auth_headers):
    resp = await client.post(
        "/presign",
        headers=auth_headers,
        params={"content_type": "application/x-protobuf"},
    )
    assert resp.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/routes/test_uploads_presign.py -v`
Expected: FAIL (endpoint not defined)

- [ ] **Step 3: Add presign endpoint to UploadsController**

Add to `backend/app/routes/uploads.py` after `complete_upload`:

```python
    @post("/presign")
    async def presign_upload(
        self,
        verified_user: VerifiedUser,
        file_name: str = Parameter(min_length=1),
        content_type: str = Parameter(default="application/octet-stream"),
    ) -> dict:
        """Generate a presigned PUT URL for direct R2 upload (small files)."""
        r2 = _client()
        bucket = get_settings().r2.bucket
        key = f"uploads/{verified_user.id}/{uuid.uuid4()}/{file_name}"

        url = r2.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=3600,
        )

        return {"url": url, "key": key}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/routes/test_uploads_presign.py -v`
Expected: PASS

- [ ] **Step 5: Run full backend test suite**

Run: `cd backend && uv run pytest tests/ -v --timeout=30`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/routes/uploads.py backend/tests/routes/test_uploads_presign.py
git commit -m "feat(backend): add POST /uploads/presign for small-file direct R2 upload"
```

---

## Task 4: Frontend — Add fflate dependency

**Files:**

- Modify: `frontend/package.json`

- [ ] **Step 1: Install fflate**

Run: `cd frontend && bun add fflate`

- [ ] **Step 2: Verify installation**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS (no type errors from new dependency)

- [ ] **Step 3: Commit**

```bash
cd frontend && git add package.json bun.lockb
git commit -m "chore(frontend): add fflate dependency for ZIP unpacking"
```

---

## Task 5: Frontend — Create zip-parser.ts with fflate

**Files:**

- Create: `frontend/src/lib/zip-parser.ts`
- Test: `frontend/src/lib/__tests__/zip-parser.test.ts`

- [ ] **Step 1: Write zip-parser.ts**

Create `frontend/src/lib/zip-parser.ts`:

```typescript
import { unzip, type Unzipped } from "fflate"

export interface ZipContents {
  video: File | null
  imuLeft: ArrayBuffer | null
  imuRight: ArrayBuffer | null
  manifest: { [key: string]: unknown } | null
  videoName: string | null
  manifestVersion: string | null
}

function unzipAsync(file: File): Promise<Unzipped> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      unzip(new Uint8Array(reader.result as ArrayBuffer), (err, data) => {
        if (err) reject(err)
        else resolve(data)
      })
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsArrayBuffer(file)
  })
}

function basename(path: string): string {
  return path.split("/").pop() ?? path
}

export async function parseZip(file: File): Promise<ZipContents> {
  const entries = await unzipAsync(file)

  let video: File | null = null
  let videoName: string | null = null
  let imuLeft: ArrayBuffer | null = null
  let imuRight: ArrayBuffer | null = null
  let manifest: { [key: string]: unknown } | null = null
  let manifestVersion: string | null = null

  for (const [path, data] of Object.entries(entries)) {
    const name = basename(path)

    // Skip macOS metadata
    if (path.startsWith("__MACOSX") || name.startsWith(".")) continue

    const ext = name.split(".").pop()?.toLowerCase() ?? ""
    if (ext === "mp4" || ext === "mov" || ext === "webm") {
      const blob = new Blob([data], { type: `video/${ext}` })
      video = new File([blob], name, { type: `video/${ext}` })
      videoName = name
    } else if (name.endsWith("_left.pb")) {
      imuLeft = data.buffer as ArrayBuffer
    } else if (name.endsWith("_right.pb")) {
      imuRight = data.buffer as ArrayBuffer
    } else if (ext === "json") {
      try {
        const parsed = JSON.parse(new TextDecoder().decode(data)) as { [key: string]: unknown }
        manifest = parsed
        manifestVersion = typeof parsed.version === "string" ? parsed.version : null
      } catch {
        // Not a valid JSON manifest, skip
      }
    }
  }

  return { video, imuLeft, imuRight, manifest, videoName, manifestVersion }
}

export function isZipFile(file: File): boolean {
  return file.name.toLowerCase().endsWith(".zip") || file.type === "application/zip"
}

export function isVideoFile(file: File): boolean {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? ""
  if (["mp4", "mov", "webm", "mkv"].includes(ext)) return true
  if (file.type.startsWith("video/")) return true
  return false
}
```

Key differences from JSZip version:
- Uses `fflate.unzip()` — per-file extraction, no full archive materialization
- `ArrayBuffer` for IMU (not Blob) — directly uploadable via presigned PUT
- Skips `__MACOSX` metadata (review finding M7)
- `isVideoFile` checks both extension AND MIME type (review finding M3)
- Case-insensitive extension matching

- [ ] **Step 2: Write unit test for zip-parser**

Create `frontend/src/lib/__tests__/zip-parser.test.ts`:

```typescript
import { describe, it, expect } from "vitest"
import { zipSync, strToU8 } from "fflate"
import { parseZip, isZipFile, isVideoFile } from "../zip-parser"

function createTestZip(files: { name: string; content: string | Uint8Array }[]): File {
  const entries: Record<string, Uint8Array> = {}
  for (const f of files) {
    entries[f.name] = typeof f.content === "string" ? strToU8(f.content) : f.content
  }
  const zipped = zipSync(entries)
  return new File([zipped], "test.zip", { type: "application/zip" })
}

describe("zip-parser", () => {
  it("detects ZIP file by extension", () => {
    expect(isZipFile(new File([], "test.zip", { type: "application/zip" }))).toBe(true)
    expect(isZipFile(new File([], "test.mp4"))).toBe(false)
  })

  it("detects video file by extension and MIME", () => {
    expect(isVideoFile(new File([], "test.mp4"))).toBe(true)
    expect(isVideoFile(new File([], "test.MOV"))).toBe(true)
    expect(isVideoFile(new File([], "test.zip"))).toBe(false)
    expect(isVideoFile(new File([], "test", { type: "video/mp4" }))).toBe(true)
  })

  it("extracts video, IMU, and manifest from ZIP", async () => {
    const zip = createTestZip([
      { name: "capture_20260507.mp4", content: "fake-video" },
      { name: "capture_20260507_left.pb", content: new Uint8Array([1, 2, 3]) },
      { name: "capture_20260507_right.pb", content: new Uint8Array([4, 5, 6]) },
      { name: "capture_20260507.json", content: JSON.stringify({ version: "1.0", videoFps: 60 }) },
    ])
    const result = await parseZip(zip)

    expect(result.video).not.toBeNull()
    expect(result.videoName).toBe("capture_20260507.mp4")
    expect(result.imuLeft).not.toBeNull()
    expect(result.imuRight).not.toBeNull()
    expect(result.manifest).toEqual({ version: "1.0", videoFps: 60 })
    expect(result.manifestVersion).toBe("1.0")
  })

  it("returns nulls for missing components", async () => {
    const zip = createTestZip([
      { name: "video.mp4", content: "only-video" },
    ])
    const result = await parseZip(zip)

    expect(result.video).not.toBeNull()
    expect(result.imuLeft).toBeNull()
    expect(result.imuRight).toBeNull()
    expect(result.manifest).toBeNull()
  })

  it("returns null video when ZIP has no video", async () => {
    const zip = createTestZip([
      { name: "data.json", content: "{}" },
    ])
    const result = await parseZip(zip)

    expect(result.video).toBeNull()
  })

  it("skips __MACOSX and dotfiles", async () => {
    const zip = createTestZip([
      { name: "__MACOSX/._capture.mp4", content: "junk" },
      { name: ".DS_Store", content: "junk" },
      { name: "capture.mp4", content: "real-video" },
    ])
    const result = await parseZip(zip)

    expect(result.videoName).toBe("capture.mp4")
  })
})
```

- [ ] **Step 3: Run tests**

Run: `cd frontend && bunx vitest run src/lib/__tests__/zip-parser.test.ts`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/zip-parser.ts frontend/src/lib/__tests__/zip-parser.test.ts
git commit -m "feat(frontend): add zip-parser with fflate for memory-safe ZIP unpacking"
```

---

## Task 6: Frontend — Add presign upload helpers to uploads.ts

**Files:**

- Modify: `frontend/src/lib/api/uploads.ts`

- [ ] **Step 1: Add presignUpload and uploadToPresignedUrl helpers**

Add to end of `frontend/src/lib/api/uploads.ts`:

```typescript
const PresignResponseSchema = z.object({
  url: z.string(),
  key: z.string(),
})

/** Get a presigned PUT URL for direct R2 upload (small files like IMU/manifest). */
export async function presignUpload(
  fileName: string,
  contentType = "application/octet-stream",
): Promise<{ url: string; key: string }> {
  return apiFetch(
    `/uploads/presign?file_name=${encodeURIComponent(fileName)}&content_type=${encodeURIComponent(contentType)}`,
    PresignResponseSchema,
    { method: "POST" },
  )
}

/** Upload data directly to R2 via presigned PUT URL. */
export async function uploadToPresignedUrl(
  url: string,
  data: ArrayBuffer | Blob,
  contentType = "application/octet-stream",
): Promise<void> {
  const res = await fetch(url, {
    method: "PUT",
    body: data,
    headers: { "Content-Type": contentType },
  })
  if (!res.ok) throw new Error(`Presigned upload failed: ${res.status}`)
}
```

Note: `presignUpload` uses `apiFetch` (handles auth + refresh). `uploadToPresignedUrl` goes direct to R2 (no auth needed, presigned URL is the auth).

- [ ] **Step 2: Run type check**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api/uploads.ts
git commit -m "feat(frontend): add presignUpload and uploadToPresignedUrl helpers"
```

---

## Task 7: Frontend — Update sessions.ts for IMU keys in createSession

**Files:**

- Modify: `frontend/src/lib/api/sessions.ts`

- [ ] **Step 1: Update useCreateSession mutation type**

In `frontend/src/lib/api/sessions.ts`, change `useCreateSession` (line 91-98):

```typescript
export function useCreateSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: {
      element_type: string
      video_key?: string
      imu_left_key?: string
      imu_right_key?: string
      manifest_key?: string
    }) => apiPost("/sessions", SessionSchema, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  })
}
```

- [ ] **Step 2: Add IMU/manifest keys to SessionSchema**

Add after `process_task_id` in `SessionSchema` (line 62):

```typescript
  imu_left_key: z.string().nullable().optional(),
  imu_right_key: z.string().nullable().optional(),
  manifest_key: z.string().nullable().optional(),
```

- [ ] **Step 3: Remove usePatchSession if no longer needed**

Check if `usePatchSession` is used anywhere except the old upload page and retry flow. If only used by `useRetrySession` and old upload, keep it (retry still needs it). Do NOT remove.

- [ ] **Step 4: Run type check**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api/sessions.ts
git commit -m "feat(frontend): add imu/manifest keys to SessionSchema and useCreateSession"
```

---

## Task 8: Frontend — Create DropZone component

**Files:**

- Create: `frontend/src/components/upload/drop-zone.tsx`

- [ ] **Step 1: Write DropZone component**

Create `frontend/src/components/upload/drop-zone.tsx`:

```tsx
"use client"

import { useCallback, useRef, useState } from "react"
import { Upload } from "lucide-react"
import { useTranslations } from "@/i18n"
import { isZipFile, isVideoFile } from "@/lib/zip-parser"

const ACCEPTED_EXTENSIONS = ".zip,.mp4,.mov,.webm,.mkv"
const MAX_SIZE = 500 * 1024 * 1024 // 500MB

export function DropZone({
  onFile,
  invalidFile,
  fileTooLarge,
}: {
  onFile: (file: File) => void
  invalidFile: string
  fileTooLarge: string
}) {
  const t = useTranslations("upload")
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const validate = useCallback(
    (file: File): boolean => {
      if (!isZipFile(file) && !isVideoFile(file)) {
        toast.error(invalidFile)
        return false
      }
      if (file.size > MAX_SIZE) {
        toast.error(fileTooLarge)
        return false
      }
      return true
    },
    [invalidFile, fileTooLarge],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      const file = e.dataTransfer.files[0]
      if (file && validate(file)) onFile(file)
    },
    [onFile, validate],
  )

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file && validate(file)) onFile(file)
    },
    [onFile, validate],
  )

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`mx-auto flex max-w-lg cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-8 py-16 transition-colors ${
        dragOver
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/50 hover:bg-accent/30"
      }`}
    >
      <Upload className="h-10 w-10 text-muted-foreground" />
      <p className="nike-h3">{t("dropHint")}</p>
      <p className="text-sm text-muted-foreground">{t("dropOrClick")}</p>
      <p className="text-xs text-muted-foreground">{t("maxSize")}</p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS}
        className="hidden"
        onChange={handleChange}
      />
    </div>
  )
}
```

Key fixes from review:
- Validation messages passed as props from parent (fixes C4 — no `toast` import needed)
- `isVideoFile` now also checks MIME type (review M3)

- [ ] **Step 2: Run type check**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/upload/drop-zone.tsx
git commit -m "feat(frontend): add DropZone component for file upload"
```

---

## Task 9: Frontend — Create FilePreview component

**Files:**

- Create: `frontend/src/components/upload/file-preview.tsx`

- [ ] **Step 1: Write FilePreview component**

Create `frontend/src/components/upload/file-preview.tsx`:

```tsx
"use client"

import { FileVideo, FileArchive, Activity, ClipboardCheck } from "lucide-react"
import { useTranslations } from "@/i18n"
import type { ZipContents } from "@/lib/zip-parser"

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${bytes} B`
}

export function FilePreview({
  file,
  zipContents,
  previewUrl,
  onRemove,
  onUpload,
}: {
  file: File
  zipContents: ZipContents | null
  previewUrl: string | null
  onRemove: () => void
  onUpload: () => void
}) {
  const t = useTranslations("upload")
  const isZip = zipContents !== null

  return (
    <div className="mx-auto max-w-lg space-y-5 px-4 py-4">
      {/* File header */}
      <div className="flex items-center gap-3">
        {isZip ? (
          <FileArchive className="h-6 w-6 text-primary" />
        ) : (
          <FileVideo className="h-6 w-6 text-primary" />
        )}
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{file.name}</p>
          <p className="text-xs text-muted-foreground">{formatSize(file.size)}</p>
        </div>
      </div>

      {/* ZIP contents summary */}
      {isZip && (
        <div className="space-y-2 rounded-xl border border-border p-4">
          {zipContents.video && (
            <div className="flex items-center gap-2 text-sm">
              <FileVideo className="h-4 w-4 text-muted-foreground" />
              <span>{t("videoFound")}</span>
            </div>
          )}
          {(zipContents.imuLeft || zipContents.imuRight) && (
            <div className="flex items-center gap-2 text-sm">
              <Activity className="h-4 w-4 text-muted-foreground" />
              <span>{t("imuInfo", { sensorCount: (zipContents.imuLeft && zipContents.imuRight ? 2 : 1) })}</span>
            </div>
          )}
          <div className="flex items-center gap-2 text-sm">
            <ClipboardCheck className="h-4 w-4 text-muted-foreground" />
            <span>{zipContents.manifest ? t("manifestOk") : t("manifestMissing")}</span>
          </div>
        </div>
      )}

      {/* Video preview for standalone MP4 */}
      {!isZip && previewUrl && (
        <div className="overflow-hidden rounded-2xl" style={{ backgroundColor: "oklch(var(--background))" }}>
          {/* biome-ignore lint/a11y/useMediaCaption: user upload, no captions */}
          <video
            src={previewUrl}
            controls
            playsInline
            className="aspect-video w-full object-contain"
          />
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={onRemove}
          className="flex flex-1 items-center justify-center gap-2 rounded-2xl border border-border px-4 py-3 font-medium text-muted-foreground transition-colors hover:bg-accent"
        >
          {t("remove")}
        </button>
        <button
          type="button"
          onClick={onUpload}
          className="flex flex-[2] items-center justify-center gap-2 rounded-2xl bg-primary px-4 py-3 font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        >
          {t("startUpload")}
        </button>
      </div>
    </div>
  )
}
```

Key fixes from review:
- `previewUrl` passed as prop (fixes H4 — parent manages URL lifecycle, revokes on remove)
- Simplified IMU info (no sampleCount placeholder — not extractable from protobuf without parsing)

- [ ] **Step 2: Run type check**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/upload/file-preview.tsx
git commit -m "feat(frontend): add FilePreview component for ZIP and video files"
```

---

## Task 10: Frontend — Rewrite UploadPage

**Files:**

- Modify: `frontend/src/app/(app)/upload/page.tsx`

- [ ] **Step 1: Rewrite UploadPage**

Replace entire content of `frontend/src/app/(app)/upload/page.tsx`:

```tsx
"use client"

import { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import { Loader2, CheckCircle2, X } from "lucide-react"
import { toast } from "sonner"
import { useTranslations } from "@/i18n"
import { useMountEffect } from "@/lib/useMountEffect"
import { ChunkedUploader, presignUpload, uploadToPresignedUrl } from "@/lib/api/uploads"
import { useCreateSession, usePatchSession } from "@/lib/api/sessions"
import { enqueueProcess } from "@/lib/api/process"
import { parseZip, isZipFile, type ZipContents } from "@/lib/zip-parser"
import { DropZone } from "@/components/upload/drop-zone"
import { FilePreview } from "@/components/upload/file-preview"

type Step = "idle" | "parsing" | "picked" | "uploading" | "done"

export default function UploadPage() {
  const router = useRouter()
  const createSession = useCreateSession()
  const patchSession = usePatchSession()
  const t = useTranslations("upload")

  const [step, setStep] = useState<Step>("idle")
  const [file, setFile] = useState<File | null>(null)
  const [zipContents, setZipContents] = useState<ZipContents | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [uploadPhase, setUploadPhase] = useState("")
  const uploaderRef = useRef<ChunkedUploader | null>(null)

  useMountEffect(() => {
    return () => {
      uploaderRef.current?.abort()
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  })

  async function handleFile(f: File) {
    if (isZipFile(f)) {
      setStep("parsing")
      try {
        const contents = await parseZip(f)
        if (!contents.video) {
          toast.error(t("noVideoInZip"))
          setStep("idle")
          return
        }
        setFile(f)
        setZipContents(contents)
        setStep("picked")
      } catch {
        toast.error(t("zipReadError"))
        setStep("idle")
      }
    } else {
      setFile(f)
      setZipContents(null)
      setPreviewUrl(URL.createObjectURL(f))
      setStep("picked")
    }
  }

  function handleRemove() {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setFile(null)
    setZipContents(null)
    setPreviewUrl(null)
    setProgress(0)
    setStep("idle")
  }

  async function uploadImuToR2(
    data: ArrayBuffer,
    fileName: string,
    contentType: string,
  ): Promise<string | null> {
    try {
      const { url, key } = await presignUpload(fileName, contentType)
      await uploadToPresignedUrl(url, data, contentType)
      return key
    } catch {
      return null
    }
  }

  async function handleUpload() {
    if (!file) return
    setStep("uploading")
    setProgress(0)

    try {
      const videoFile = zipContents?.video ?? file
      let imuLeftKey: string | null = null
      let imuRightKey: string | null = null
      let manifestKey: string | null = null

      // Phase 1: Upload IMU/manifest to R2 via presigned URLs (if ZIP)
      if (zipContents) {
        setUploadPhase(t("uploadingImu"))

        if (zipContents.imuLeft) {
          imuLeftKey = await uploadImuToR2(zipContents.imuLeft, "imu_left.pb", "application/x-protobuf")
        }
        if (zipContents.imuRight) {
          imuRightKey = await uploadImuToR2(zipContents.imuRight, "imu_right.pb", "application/x-protobuf")
        }
        if (zipContents.manifest) {
          const manifestData = new TextEncoder().encode(JSON.stringify(zipContents.manifest))
          manifestKey = await uploadImuToR2(manifestData.buffer as ArrayBuffer, "manifest.json", "application/json")
        }

        // Warn if any IMU upload failed, proceed with video-only
        if ((zipContents.imuLeft && !imuLeftKey) || (zipContents.imuRight && !imuRightKey)) {
          toast.warning(t("imuUploadWarning"))
        }
      }

      // Phase 2: Upload video via ChunkedUploader
      setUploadPhase(t("uploadingVideo"))
      const uploader = new ChunkedUploader(videoFile, (loaded, total) => {
        setProgress(Math.round((loaded / total) * 100))
      })
      uploaderRef.current = uploader
      const videoKey = await uploader.upload()

      // Phase 3: Create session with ALL keys (atomic, no race)
      setUploadPhase(t("startingAnalysis"))
      setProgress(100)
      const session = await createSession.mutateAsync({
        element_type: "auto",
        video_key: videoKey,
        ...(imuLeftKey ? { imu_left_key: imuLeftKey } : {}),
        ...(imuRightKey ? { imu_right_key: imuRightKey } : {}),
        ...(manifestKey ? { manifest_key: manifestKey } : {}),
      })

      // Phase 4: Enqueue processing
      const processRes = await enqueueProcess({
        video_key: videoKey,
        person_click: { x: -1, y: -1 },
        session_id: session.id,
      })
      await patchSession.mutateAsync({
        id: session.id,
        body: { process_task_id: processRes.task_id },
      })

      setStep("done")
      toast.success(t("videoUploaded"))

      if (session?.id) {
        router.push(`/sessions/${session.id}`)
      }
    } catch {
      toast.error(t("uploadError"))
      setProgress(0)
      setStep("picked")
    }
  }

  function handleCancel() {
    uploaderRef.current?.abort()
    setProgress(0)
    setStep("picked")
  }

  // Done state
  if (step === "done") {
    return (
      <div className="flex flex-col items-center justify-center gap-4 px-4 py-20">
        <CheckCircle2 className="h-12 w-12 text-primary" />
        <p className="nike-h3">{t("videoUploaded")}</p>
        <p className="text-sm text-muted-foreground">{t("analyzingHint")}</p>
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  // Parsing state
  if (step === "parsing") {
    return (
      <div className="mx-auto max-w-lg space-y-5 px-4 py-20">
        <div className="text-center">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary" />
          <p className="mt-3 nike-h3">{t("parsingZip")}</p>
        </div>
      </div>
    )
  }

  // Uploading state
  if (step === "uploading") {
    return (
      <div className="mx-auto max-w-lg space-y-5 px-4 py-20">
        <div className="text-center">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary" />
          <p className="mt-3 nike-h3">{uploadPhase}</p>
        </div>
        <div className="space-y-2">
          <div className="h-2 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-center text-xs text-muted-foreground">{progress}%</p>
        </div>
        <div className="flex justify-center">
          <button
            type="button"
            onClick={handleCancel}
            className="flex items-center gap-2 rounded-2xl border border-border px-4 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent"
          >
            <X className="h-4 w-4" />
            {t("cancelUpload")}
          </button>
        </div>
      </div>
    )
  }

  // Picked state
  if (step === "picked" && file) {
    return (
      <FilePreview
        file={file}
        zipContents={zipContents}
        previewUrl={previewUrl}
        onRemove={handleRemove}
        onUpload={handleUpload}
      />
    )
  }

  // Idle state
  return (
    <div className="flex flex-col items-center justify-center gap-4 px-4 py-8">
      <DropZone
        onFile={handleFile}
        invalidFile={t("invalidFile")}
        fileTooLarge={t("fileTooLarge")}
      />
    </div>
  )
}
```

Key changes from original plan:
- **`parsing` state** added (review H5)
- **Multi-phase upload**: IMU first, then video, then session creation (review C5)
- **Session created LAST** with all keys — single atomic call (review C5)
- **No `attachImu`/`attachManifest`** — presigned URLs to R2 directly (review C1, C6)
- **Cancel button** during upload (review M5)
- **Progress reset** to 0 on failure (review H5)
- **`previewUrl` managed here**, revoked on remove (review H4)
- **i18n keys** for all user-facing strings (review H5)

- [ ] **Step 2: Run type check**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Run lint**

Run: `cd frontend && bunx next lint`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/\(app\)/upload/page.tsx
git commit -m "feat(frontend): rewrite upload page — file upload only, ZIP support, session created last"
```

---

## Task 11: Frontend — Update i18n messages

**Files:**

- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Update ru.json upload section**

Replace the `"upload"` key (line 225-238) in `frontend/messages/ru.json`:

```json
  "upload": {
    "uploadingVideo": "Загрузка видео...",
    "parsingZip": "Распаковка архива...",
    "dropHint": "Перетащите ZIP или видео",
    "dropOrClick": "или нажмите для выбора файла",
    "maxSize": "до 500 MB",
    "videoFound": "Видео: найдено",
    "imuInfo": "IMU: {sensorCount} датчиков",
    "manifestOk": "Manifest: ✓",
    "manifestMissing": "Manifest: отсутствует",
    "remove": "Удалить",
    "startUpload": "Загрузить и анализировать",
    "uploadingImu": "Загрузка датчиков...",
    "startingAnalysis": "Запуск анализа...",
    "invalidFile": "Неподдерживаемый формат файла",
    "fileTooLarge": "Файл слишком большой (максимум 500 MB)",
    "zipReadError": "Не удалось прочитать ZIP-архив",
    "noVideoInZip": "В ZIP не найдено видео",
    "imuUploadWarning": "Данные датчиков не загружены, анализ только по видео",
    "uploadError": "Ошибка загрузки",
    "videoUploaded": "Видео загружено!",
    "analyzingHint": "Анализ начнётся автоматически...",
    "cancelUpload": "Отмена"
  }
```

- [ ] **Step 2: Update en.json upload section**

Replace the `"upload"` key in `frontend/messages/en.json`:

```json
  "upload": {
    "uploadingVideo": "Uploading video...",
    "parsingZip": "Unpacking archive...",
    "dropHint": "Drop ZIP or video",
    "dropOrClick": "or click to select file",
    "maxSize": "up to 500 MB",
    "videoFound": "Video: found",
    "imuInfo": "IMU: {sensorCount} sensor(s)",
    "manifestOk": "Manifest: ✓",
    "manifestMissing": "Manifest: missing",
    "remove": "Remove",
    "startUpload": "Upload and analyze",
    "uploadingImu": "Uploading sensors...",
    "startingAnalysis": "Starting analysis...",
    "invalidFile": "Unsupported file format",
    "fileTooLarge": "File too large (max 500 MB)",
    "zipReadError": "Could not read ZIP archive",
    "noVideoInZip": "No video found in ZIP",
    "imuUploadWarning": "Sensor data not uploaded, video-only analysis",
    "uploadError": "Upload error",
    "videoUploaded": "Video uploaded!",
    "analyzingHint": "Analysis will start automatically...",
    "cancelUpload": "Cancel"
  }
```

- [ ] **Step 3: Update nav.upload label**

In `ru.json`, change `"upload": "Запись"` to `"upload": "Загрузка"` (line 66).

In `en.json`, change `"upload": "Record"` to `"upload": "Upload"` (if exists).

- [ ] **Step 4: Commit**

```bash
git add frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(i18n): update upload page keys — remove camera, add drop/ZIP/parsing"
```

---

## Task 12: Frontend — Remove old camera/element-picker/chunked-uploader components

**Files:**

- Delete: `frontend/src/components/upload/camera-recorder.tsx`
- Delete: `frontend/src/components/upload/element-picker.tsx`
- Delete: `frontend/src/components/upload/chunked-uploader.tsx`

- [ ] **Step 1: Verify no imports reference these files**

Run: `cd frontend && grep -r "camera-recorder\|element-picker\|chunked-uploader" src/ --include="*.ts" --include="*.tsx" -l`
Expected: Only the files themselves (and possibly the old upload page which is already rewritten)

- [ ] **Step 2: Delete files**

```bash
rm frontend/src/components/upload/camera-recorder.tsx
rm frontend/src/components/upload/element-picker.tsx
rm frontend/src/components/upload/chunked-uploader.tsx
```

- [ ] **Step 3: Run type check**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS (no broken imports)

- [ ] **Step 4: Commit**

```bash
git add -u frontend/src/components/upload/
git commit -m "chore(frontend): remove camera-recorder, element-picker, chunked-uploader components"
```

---

## Task 13: Backend — Add TODO for IMU integration in worker

**Files:**

- Modify: `backend/app/worker.py`

- [ ] **Step 1: Add TODO comment for IMU integration**

Find the section in `backend/app/worker.py` where session data is loaded (near `session.video_key` usage). Add:

```python
    # TODO: integrate IMU into ML pipeline — session.imu_left_key, session.imu_right_key, session.manifest_key
    # Currently IMU data is uploaded to R2 but not consumed by the pipeline.
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/worker.py
git commit -m "chore(backend): add TODO for IMU integration in ML pipeline"
```

---

## Task 14: Verify full stack

- [ ] **Step 1: Run backend tests**

Run: `cd backend && uv run pytest tests/ -v --timeout=30`
Expected: PASS

- [ ] **Step 2: Run backend type check**

Run: `cd backend && uv run basedpyright app/`
Expected: PASS

- [ ] **Step 3: Run frontend type check**

Run: `cd frontend && bunx tsc --noEmit`
Expected: PASS

- [ ] **Step 4: Run frontend lint**

Run: `cd frontend && bunx next lint`
Expected: PASS

- [ ] **Step 5: Run frontend tests**

Run: `cd frontend && bunx vitest run`
Expected: PASS
