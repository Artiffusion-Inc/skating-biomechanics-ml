# Upload Page Rework — Expert Review Synthesis

**Date:** 2026-05-07
**Reviewers:** Backend Architect, Frontend/UX Architect, Systems Architect

---

## Critical Findings (must fix before implementation)

### C1. `data: bytes` invalid in Litestar — use `body: bytes`

Plan's endpoint uses `data: bytes` as parameter name. Litestar reserves `data` for Pydantic model extraction (JSON decode pipeline). Raw binary body requires reserved name `body: bytes`.

**Fix:** Replace `data: bytes` with `body: bytes` in both endpoints. Test client must use `content=b"..."` not `data=b"..."`.

### C2. JSZip loads entire ZIP into memory (~1.5GB peak for 500MB ZIP)

JSZip materializes full ZIP + decompressed entries in JS heap. Mobile devices (1-2GB browser limit) will OOM. Desktop Chrome shows "Out of Memory" on 500MB ZIPs.

**Fix:** Replace `jszip` with `fflate` (~40% smaller, per-file extraction without full archive materialization). Add `"parsing"` state to show spinner during ZIP extraction. Null out ZIP reference after extraction to free memory.

### C3. IMU/manifest data is uploaded but never consumed

The ML pipeline (`worker.py`, Vast.ai payload, GPU server) has zero awareness of IMU keys. Data goes to R2 → sits there unused. Users believe sensor data is analyzed when it's ignored.

**Fix:** Store IMU/manifest keys on Session as forward-compatibility fields. Upload to R2 for future ML integration. Add `// TODO: integrate IMU into ML pipeline` in worker.py. Do NOT create IMU upload endpoints yet — just store keys via `createSession` (see C5).

### C4. `toast` imported but missing in DropZone

`drop-zone.tsx` calls `toast.error()` without `import { toast } from "sonner"`.

**Fix:** Add import, or move validation to parent UploadPage (preferred — follows existing pattern).

### C5. Race condition: session created before IMU/manifest attached

Current flow: createSession → attachImu → attachManifest → patchSession → enqueueProcess. Session gets `status: "queued"` at createSession. Worker can start processing before IMU keys are set.

**Fix:** Restructure to create session LAST:
1. Upload video to R2 → video_key
2. Upload IMU/manifest to R2 → keys (if present)
3. `createSession(video_key, imu_left_key?, imu_right_key?, manifest_key?)` — single atomic call
4. `enqueueProcess(video_key, session_id)`

This eliminates: race condition, redundant PATCH, double-write. Add optional keys to `CreateSessionRequest`.

### C6. `apiFetch` binary body — missing Content-Type

`attachImu`/`attachManifest` pass raw Blob without `Content-Type: application/octet-stream`. Browser may set wrong type.

**Fix:** Add explicit `headers: { "Content-Type": "application/octet-stream" }` to both functions. Remove dead `FormData` code.

---

## High Findings (should fix)

### H1. Sync `upload_bytes()` blocks async event loop

All 3 reviewers flagged this. Use `upload_bytes_async()` from `app/storage.py`.

### H2. No request body size limit for IMU/manifest endpoints

IMU at 200Hz for 3min = ~3.4MB per sensor. No limit = potential DoS.

**Fix:** Add `request_max_body_size=10_485_760` (10MB) to IMU endpoint, 1MB to manifest.

### H3. Test fixtures don't match conftest.py

`verified_user` fixture does not exist. Use `authed_user` (has `is_verified=True`). Mock `upload_bytes_async` not `upload_bytes`. Tests need `content=` not `data=` for raw bytes.

### H4. `URL.createObjectURL` memory leak in FilePreview

`<video src={URL.createObjectURL(file)}>` never revoked.

**Fix:** Manage preview URL in UploadPage, revoke on `handleRemove`. Pass `previewUrl` as prop to FilePreview.

### H5. State machine gaps

- No `progress` reset on upload failure
- Hardcoded English strings in IMU failure toasts
- ZIP parsing has no cancellation mechanism

**Fix:** Reset `progress` to 0 in catch. Add i18n keys for IMU warnings. Add `"parsing"` step with spinner.

### H6. PatchSessionRequest should NOT expose R2 key fields

Allowing PATCH to set `imu_left_key`/`manifest_key` to arbitrary strings = path traversal / data integrity risk.

**Fix:** Don't add these fields to `PatchSessionRequest`. Keys set only via `CreateSessionRequest` or dedicated upload endpoints.

### H7. Double-write: endpoint stores key AND frontend PATCHes same key

Plan has IMU endpoint calling `update(db, session, **{field: key})` AND frontend calling `patchSession` with same keys.

**Fix:** With C5 restructure, this is eliminated — keys set at session creation, no PATCH needed.

---

## Medium Findings (nice to fix)

| # | Finding | Fix |
|---|---------|-----|
| M1 | Replace JSZip with `fflate` | `bun remove jszip && bun add fflate` |
| M2 | Multi-phase progress for ZIP uploads | Add `uploadPhase` state: "parsing" → "video" → "processing" |
| M3 | MIME type checks insufficient | Add `file.type.startsWith("video/")` fallback, case-insensitive ext |
| M4 | i18n Russian pluralization | Use `"сенсоров"` (works for all counts) or ICU plural format |
| M5 | No cancel button during upload | Add cancel button calling `uploaderRef.current?.abort()` |
| M6 | ZIP parsing blocks main thread | `fflate` yields between entries; add "parsing" state |
| M7 | `.json` matching too broad in zip-parser | Match manifest by basename prefix matching video; skip `__MACOSX` |
| M8 | No manifest version validation | Check `manifest.version === "1.0"`, warn on mismatch |
| M9 | Path traversal risk in session_id | Validate UUID format before constructing R2 key |
| M10 | No idempotency protection | Accept for MVP, add TODO |
| M11 | Dead `uploadBlob` in uploads.ts | Remove if not used by UploadPage |
| M12 | Choreography element-picker deletion risk | Use fully qualified path, verify with targeted grep |

---

## Revised Architecture

Based on findings, the recommended architecture changes:

### Before (plan)
```
Drop ZIP → JSZip unpack → chunk-upload video → createSession → attachImu → attachManifest → PATCH → enqueueProcess → redirect
```

### After (revised)
```
Drop ZIP → fflate unpack (show "parsing") → chunk-upload video → upload IMU/manifest to R2 → createSession(all keys) → enqueueProcess → redirect
```

**Key changes:**
1. **fflate** instead of JSZip (memory-safe)
2. **Session created last** with all keys (atomic, no race)
3. **No IMU/manifest endpoints** — keys passed directly to `createSession`
4. **IMU upload to R2** via presigned URLs (same pattern as video chunks)
5. **Forward-compatibility only** — IMU stored but not consumed by ML pipeline yet

### IMU Upload Flow (revised)

IMU files are small (<10MB each). Don't need chunked upload. Upload directly via presigned URL:

1. `POST /uploads/presign?file_name=imu_left.pb&content_type=application/x-protobuf` → presigned PUT URL + key
2. `PUT <presigned_url>` (direct to R2, bypasses backend body size limits)
3. Return key to frontend
4. Frontend passes key to `createSession`

This reuses the existing upload pattern (presigned URLs for direct R2 upload) and avoids the Litestar `body: bytes` issue entirely.

---

## Spec/Plan Updates Required

| Document | Updates |
|----------|---------|
| Spec | Replace JSZip with fflate, restructure data flow, add presign-based IMU upload, remove IMU/manifest endpoints, add `parsing` state |
| Plan Task 1 | Add `imu_left_key`/`imu_right_key`/`manifest_key` to `CreateSessionRequest` (not PatchSessionRequest) |
| Plan Task 3 | DELETE (no dedicated IMU/manifest endpoints) |
| Plan Task 4 | `bun add fflate` instead of `jszip` |
| Plan Task 5 | Rewrite zip-parser with fflate, type Manifest, add version check, basename matching |
| Plan Task 6 | Rewrite API helpers: presign + direct upload, remove attachImu/attachManifest, remove uploadBlob |
| Plan Task 7 | Add toast import, move validation to parent |
| Plan Task 8 | Fix objectURL leak, type manifest for preview |
| Plan Task 9 | Add `parsing` state, uploadPhase, restructure flow (session created last) |
| Plan Task 10 | Add new i18n keys: parsingZip, uploadingImu, imuUploadWarning, startingAnalysis |
