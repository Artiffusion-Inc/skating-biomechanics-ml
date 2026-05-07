# Upload Page Rework — Design Spec

**Date:** 2026-05-07
**Status:** Draft

## Problem

Mobile app (Flutter) now handles all recording: camera, IMU sensors (WT901), edge overlay, calibration, export. Web upload page has redundant camera recording UI (`CameraRecorder`, `MediaRecorder`) that duplicates ~20% of mobile functionality without IMU support.

## Solution

Replace camera-recording upload page with file-upload-only page. Accept ZIP (from mobile app) or standalone video (MP4/MOV). Frontend unpacks ZIP in browser, uploads components separately via existing multipart + new IMU/manifest endpoints.

## Architecture

### State Machine

```
idle → picked → uploading → done → redirect /sessions/{id}
```

- **idle** — drop zone + click to browse
- **picked** — file preview: video thumbnail (if MP4), IMU summary (if ZIP), manifest status
- **uploading** — progress bar (ChunkedUploader)
- **done** — toast + redirect

### Data Flow

```
File dropped/selected
  ├─ ZIP → JSZip unpack in browser
  │    ├─ *.mp4 → ChunkedUploader → R2 (video_key)
  │    ├─ *_left.pb → POST /sessions/{id}/imu → R2 (imu_left_key)
  │    ├─ *_right.pb → POST /sessions/{id}/imu → R2 (imu_right_key)
  │    └─ *.json (manifest) → POST /sessions/{id}/manifest → R2 (manifest_key)
  │
  └─ MP4/MOV → ChunkedUploader → R2 (video_key)

After upload:
  → createSession(element_type: "auto", video_key)
  → PATCH session (imu_left_key, imu_right_key, manifest_key — if present)
  → enqueueProcess(video_key, session_id)
  → redirect /sessions/{id}
```

### Frontend Unpacking (JSZip)

ZIP structure from mobile app:
```
capture_YYYYMMDD_HHMMSS.zip
  ├── capture_YYYYMMDD_HHMMSS.mp4
  ├── capture_YYYYMMDD_HHMMSS_left.pb
  ├── capture_YYYYMMDD_HHMMSS_right.pb
  └── capture_YYYYMMDD_HHMMSS.json   (manifest)
```

JSZip reads ZIP, extracts:
1. Video file → `File` blob for ChunkedUploader
2. IMU protobuf files → `Blob` for upload endpoints
3. Manifest JSON → parsed for preview display (sensor count, duration, calibration status)

### Backend Changes

**New endpoints** (attach IMU/manifest to session):

| Method | Path | Body | Response |
|--------|------|------|----------|
| POST | `/sessions/{id}/imu` | `{ side: "left"\|"right", data: bytes }` | `{ key: string }` |
| POST | `/sessions/{id}/manifest` | `{ data: bytes }` | `{ key: string }` |

These upload binary data to R2 and store the key on the session model.

**Session model additions:**

| Column | Type | Nullable |
|--------|------|----------|
| `imu_left_key` | string | yes |
| `imu_right_key` | string | yes |
| `manifest_key` | string | yes |

**Element picker removed** — always `element_type: "auto"`. No user selection.

## Components

### Files to Remove

| File | Reason |
|------|--------|
| `components/upload/camera-recorder.tsx` | Camera recording — mobile handles this |
| `components/upload/element-picker.tsx` | Element picker removed — auto-detect only |
| `components/upload/chunked-uploader.tsx` | Replaced by inline progress in UploadPage |

### Files to Add

| File | Purpose |
|------|---------|
| `components/upload/drop-zone.tsx` | Drag-and-drop + click file selector |
| `components/upload/file-preview.tsx` | File info: name, size, type detection, IMU summary |
| `lib/zip-parser.ts` | JSZip wrapper: unpack, extract video/IMU/manifest |

### Files to Modify

| File | Change |
|------|--------|
| `app/(app)/upload/page.tsx` | Full rewrite: idle → picked → uploading → done |
| `lib/api/uploads.ts` | Add `uploadBlob()` for IMU/manifest binary upload |
| `lib/api/sessions.ts` | Add `useAttachImu()`, `useAttachManifest()` mutations |
| `messages/ru.json` | Update upload i18n keys (remove camera/record, add drop/ZIP) |
| `messages/en.json` | Same |
| `backend/app/models/session.py` | Add `imu_left_key`, `imu_right_key`, `manifest_key` columns |
| `backend/app/routes/sessions.py` | Add POST `/sessions/{id}/imu`, POST `/sessions/{id}/manifest` |
| `backend/app/schemas.py` | Add IMU/manifest request/response schemas |

### Dependencies

| Package | Why |
|---------|-----|
| `jszip` | Browser-side ZIP unpacking (~45KB gzipped) |

## UI Design

### Idle State

```
┌─────────────────────────────────────┐
│                                     │
│      ┌───────────────────┐         │
│      │                   │         │
│      │   📁  Drop zone    │         │
│      │                   │         │
│      │  ZIP или MP4      │         │
│      │  до 500 MB        │         │
│      │                   │         │
│      └───────────────────┘         │
│                                     │
│   или нажмите для выбора файла      │
│                                     │
└─────────────────────────────────────┘
```

### Picked State (ZIP)

```
┌─────────────────────────────────────┐
│  📄 capture_20260507.zip    156 MB  │
│                                     │
│  🎬 Видео: 1920×1080, 60fps        │
│  📊 IMU: 2 датчика, 48K сэмплов    │
│  📋 Manifest: ✓                     │
│                                     │
│  [Удалить]    [Загрузить и анализир.]│
└─────────────────────────────────────┘
```

### Picked State (MP4)

```
┌─────────────────────────────────────┐
│  📄 skating_clip.mp4        89 MB   │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  ▶ video preview               │ │
│  └───────────────────────────────┘ │
│                                     │
│  [Удалить]    [Загрузить и анализир.]│
└─────────────────────────────────────┘
```

### Uploading State

```
┌─────────────────────────────────────┐
│                                     │
│     Загрузка видео...  67%          │
│     ████████████░░░░░░░░░░░         │
│                                     │
└─────────────────────────────────────┘
```

## i18n Keys

Remove: `cameraUnavailable`, `recordHint`, `retake`, `selectElement`, `autoDetectHint`

Add:
```json
{
  "upload": {
    "title": "Загрузка",
    "dropHint": "Перетащите ZIP или видео",
    "dropOrClick": "или нажмите для выбора файла",
    "maxSize": "до 500 MB",
    "videoInfo": "Видео: {width}×{height}, {fps}fps",
    "imuInfo": "IMU: {sensorCount} датчика, {sampleCount}K сэмплов",
    "manifestOk": "Manifest: ✓",
    "manifestMissing": "Manifest: отсутствует",
    "remove": "Удалить",
    "startUpload": "Загрузить и анализировать",
    "invalidFile": "Неподдерживаемый формат файла",
    "zipReadError": "Не удалось прочитать ZIP-архив",
    "noVideoInZip": "В ZIP не найдено видео"
  }
}
```

## Error Handling

| Case | Behavior |
|------|----------|
| Non-ZIP/MP4 file dropped | Show `invalidFile` toast, stay in idle |
| ZIP corrupted / JSZip fails | Show `zipReadError` toast, stay in idle |
| ZIP has no video file | Show `noVideoInZip` toast, stay in idle |
| Upload fails mid-chunk | Show `uploadError` toast, back to picked |
| IMU/manifest upload fails | Show warning toast, proceed with video-only |

## Migration

1. Add DB migration for `imu_left_key`, `imu_right_key`, `manifest_key` columns (nullable)
2. Deploy backend changes first (new endpoints + columns)
3. Deploy frontend changes
4. Remove `camera-recorder.tsx`, `element-picker.tsx`, `chunked-uploader.tsx`

## Testing

- **Unit**: `zip-parser.ts` — mock ZIP with video + IMU + manifest, verify extraction
- **Unit**: `drop-zone.tsx` — drag events, file type validation
- **Integration**: full flow — ZIP drop → unpack → upload → session created → redirected
- **Integration**: MP4 drop → upload → session created → redirected
- **Backend**: POST `/sessions/{id}/imu` — upload + R2 key stored
- **Backend**: POST `/sessions/{id}/manifest` — upload + R2 key stored