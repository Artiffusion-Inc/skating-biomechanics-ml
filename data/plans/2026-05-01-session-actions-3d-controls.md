# Session Actions + 3D Viewer Controls

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add share/delete actions to the session detail page and camera preset + render mode controls to the 3D skeleton viewer.

**Architecture:** `SessionActions` component handles share (clipboard API) and delete (confirm dialog + mutation + redirect). `useAnalysisStore` gets `cameraPreset` and `renderMode` fields. `ThreeJSkeletonViewer` adds a top control bar with preset buttons and a wireframe/solid toggle. `SkeletalMesh` switches between thin `Line` (wireframe) and thick `Line` + `Sphere` joints (solid).

**Tech Stack:** Next.js 16, React 19, Zustand, three.js, @react-three/drei, @react-three/fiber, Tailwind CSS v4

---

## File Map

| File | Responsibility |
|------|---------------|
| `frontend/src/components/session/session-actions.tsx` | NEW. Share (clipboard) and Delete (confirm + mutate) buttons. |
| `frontend/src/stores/analysis.ts` | Add `cameraPreset` and `renderMode` state + actions. |
| `frontend/src/components/analysis/threejs-skeleton-viewer.tsx` | Add top control bar: camera presets + render mode toggle. |
| `frontend/src/components/analysis/skeletal-mesh.tsx` | Switch geometry based on `renderMode` from store. |
| `frontend/src/app/(app)/sessions/[id]/page.tsx` | Import `SessionActions`, place in right column. |

---

## Dependencies

No new packages. `Sphere` already available in `@react-three/drei`.

---

### Task 1: Session Actions (Share + Delete)

**Files:**

- Create: `frontend/src/components/session/session-actions.tsx`
- Modify: `frontend/src/app/(app)/sessions/[id]/page.tsx`
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Create `SessionActions` component**

```typescript
"use client"

import { useRouter } from "next/navigation"
import { Share2, Trash2 } from "lucide-react"
import { useTranslations } from "@/i18n"
import { useDeleteSession } from "@/lib/api/sessions"

interface Props {
  sessionId: string
}

export function SessionActions({ sessionId }: Props) {
  const t = useTranslations("session")
  const router = useRouter()
  const deleteMutation = useDeleteSession()

  const handleShare = async () => {
    const url = window.location.href
    await navigator.clipboard.writeText(url)
    // Toast could be added here; for now rely on browser UI
  }

  const handleDelete = () => {
    if (!window.confirm(t("deleteConfirm"))) return
    deleteMutation.mutate(sessionId, {
      onSuccess: () => router.push("/feed"),
    })
  }

  return (
    <div className="flex gap-2">
      <button
        type="button"
        onClick={handleShare}
        className="flex items-center gap-1.5 rounded-xl border border-border px-3 py-1.5 text-sm hover:bg-muted"
      >
        <Share2 className="h-4 w-4" />
        {t("share")}
      </button>
      <button
        type="button"
        onClick={handleDelete}
        disabled={deleteMutation.isPending}
        className="flex items-center gap-1.5 rounded-xl border border-border px-3 py-1.5 text-sm text-destructive hover:bg-destructive/10"
      >
        <Trash2 className="h-4 w-4" />
        {deleteMutation.isPending ? t("deleting") : t("delete")}
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Add translation keys**

`frontend/messages/ru.json` under `"session"`:
```json
"share": "Поделиться",
"delete": "Удалить",
"deleteConfirm": "Удалить эту сессию? Это действие необратимо.",
"deleting": "Удаление..."
```

`frontend/messages/en.json` under `"session"`:
```json
"share": "Share",
"delete": "Delete",
"deleteConfirm": "Delete this session? This action cannot be undone.",
"deleting": "Deleting..."
```

- [ ] **Step 3: Wire into page.tsx**

Import:
```typescript
import { SessionActions } from "@/components/session/session-actions"
```

Insert near the top of the **right column**, before `ThreeJSkeletonViewer`:
```tsx
<SessionActions sessionId={session.id} />
```

- [ ] **Step 4: Verify**

```bash
cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check src/components/session/session-actions.tsx 'src/app/(app)/sessions/[id]/page.tsx'
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/session/session-actions.tsx frontend/src/app/(app)/sessions/[id]/page.tsx frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(frontend): share and delete actions on session detail"
```

---

### Task 2: Camera Presets in Analysis Store

**Files:**

- Modify: `frontend/src/stores/analysis.ts`

- [ ] **Step 1: Extend store with camera preset**

```typescript
export interface AnalysisState {
  currentFrame: number
  isPlaying: boolean
  playbackSpeed: number
  selectedJoint: number | null
  cameraPreset: "front" | "side" | "top"

  setCurrentFrame: (frame: number) => void
  setIsPlaying: (playing: boolean) => void
  setPlaybackSpeed: (speed: number) => void
  setSelectedJoint: (joint: number | null) => void
  setCameraPreset: (preset: "front" | "side" | "top") => void
  reset: () => void
}

export const useAnalysisStore = create<AnalysisState>(set => ({
  currentFrame: 0,
  isPlaying: false,
  playbackSpeed: 1.0,
  selectedJoint: null,
  cameraPreset: "front",

  setCurrentFrame: frame => set({ currentFrame: frame }),
  setIsPlaying: playing => set({ isPlaying: playing }),
  setPlaybackSpeed: speed => set({ playbackSpeed: speed }),
  setSelectedJoint: joint => set({ selectedJoint: joint }),
  setCameraPreset: preset => set({ cameraPreset: preset }),
  reset: () =>
    set({
      currentFrame: 0,
      isPlaying: false,
      playbackSpeed: 1.0,
      selectedJoint: null,
      cameraPreset: "front",
    }),
}))
```

- [ ] **Step 2: Verify**

```bash
cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check src/stores/analysis.ts
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/stores/analysis.ts
git commit -m "feat(frontend): add camera preset to analysis store"
```

---

### Task 3: Camera Preset Buttons in 3D Viewer

**Files:**

- Modify: `frontend/src/components/analysis/threejs-skeleton-viewer.tsx`
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Add CameraController inside Scene**

In `frontend/src/components/analysis/threejs-skeleton-viewer.tsx`, add `CameraController` component before `Scene`:

```typescript
import { useThree } from "@react-three/fiber"

const CAMERA_PRESETS = {
  front: { position: [0, 0, 1.5] as [number, number, number], target: [0, 0, 0] as [number, number, number] },
  side: { position: [1.5, 0, 0] as [number, number, number], target: [0, 0, 0] as [number, number, number] },
  top: { position: [0, 1.5, 0] as [number, number, number], target: [0, 0, 0] as [number, number, number] },
}

function CameraController() {
  const { camera } = useThree()
  const { cameraPreset } = useAnalysisStore()

  useEffect(() => {
    const preset = CAMERA_PRESETS[cameraPreset]
    if (preset) {
      camera.position.set(...preset.position)
      camera.lookAt(...preset.target)
      camera.updateProjectionMatrix()
    }
  }, [camera, cameraPreset])

  return null
}
```

And import `useEffect` from "react" and `useThree` from "@react-three/fiber".

Insert `<CameraController />` inside `Scene` return, before `PerspectiveCamera`.

- [ ] **Step 2: Add preset buttons in viewer container**

Add `CameraPresets` component after `PlaybackControls` inside `ThreeJSkeletonViewer` return:

```typescript
function CameraPresets() {
  const { cameraPreset, setCameraPreset } = useAnalysisStore()
  const t = useTranslations("analysis")
  const presets: Array<{ key: "front" | "side" | "top"; label: string }> = [
    { key: "front", label: t("viewFront") },
    { key: "side", label: t("viewSide") },
    { key: "top", label: t("viewTop") },
  ]
  return (
    <div className="absolute top-2 left-2 flex gap-1 rounded-lg p-1"
      style={{ backgroundColor: "oklch(var(--background) / 0.7)" }}
    >
      {presets.map(p => (
        <button
          key={p.key}
          type="button"
          onClick={() => setCameraPreset(p.key)}
          className={`rounded-md px-2 py-1 text-xs ${cameraPreset === p.key ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
        >
          {p.label}
        </button>
      ))}
    </div>
  )
}
```

Render `<CameraPresets />` inside the same relative container (after `PlaybackControls`).

- [ ] **Step 3: Add translations**

`frontend/messages/ru.json` under `"analysis"`:
```json
"viewFront": "Спереди",
"viewSide": "Сбоку",
"viewTop": "Сверху"
```

`frontend/messages/en.json` under `"analysis"`:
```json
"viewFront": "Front",
"viewSide": "Side",
"viewTop": "Top"
```

- [ ] **Step 4: Verify**

```bash
cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check src/components/analysis/threejs-skeleton-viewer.tsx
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/analysis/threejs-skeleton-viewer.tsx frontend/src/stores/analysis.ts frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(frontend): camera preset buttons on 3D skeleton viewer"
```

---

### Task 4: Wireframe / Solid Toggle

**Files:**

- Modify: `frontend/src/stores/analysis.ts`
- Modify: `frontend/src/components/analysis/skeletal-mesh.tsx`
- Modify: `frontend/src/components/analysis/threejs-skeleton-viewer.tsx`
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Extend store with renderMode**

Add to `AnalysisState`:
```typescript
renderMode: "wireframe" | "solid"
setRenderMode: (mode: "wireframe" | "solid") => void
```

And to store defaults:
```typescript
renderMode: "wireframe",
setRenderMode: mode => set({ renderMode: mode }),
```

- [ ] **Step 2: Update `SkeletalMesh` to support solid mode**

Import `Sphere` from `@react-three/drei`:
```typescript
import { Line, Sphere } from "@react-three/drei"
```

Add `SolidJoint` component:
```typescript
function SolidJoint({ position, color }: { position: [number, number, number]; color: string }) {
  return (
    <Sphere args={[0.02, 8, 8]} position={position}>
      <meshStandardMaterial color={color} />
    </Sphere>
  )
}
```

Change `SkeletalMesh` to accept `renderMode` prop:
```typescript
interface SkeletalMeshProps {
  poseData: PoseData
  frameMetrics: FrameMetrics | null
  currentFrame: number
  renderMode: "wireframe" | "solid"
}
```

In the return, conditionally render:
```tsx
{renderMode === "solid" &&
  joints.map(joint => (
    <SolidJoint
      key={`joint-${joint.index}`}
      position={joint.position as [number, number, number]}
      color={`#${joint.color.toString(16).padStart(6, "0")}`}
    />
  ))}
```

Keep `Joint` (point) for wireframe mode.

- [ ] **Step 3: Wire `renderMode` in `Scene` component**

`Scene` currently receives `poseData` and `frameMetrics`. Add `renderMode`:
```typescript
function Scene({ poseData, frameMetrics }: { poseData: PoseData; frameMetrics: FrameMetrics | null }) {
  const { currentFrame, renderMode } = useAnalysisStore()
  // ...
  return (
    <>
      <CameraController />
      <PerspectiveCamera ... />
      <OrbitControls ... />
      <Environment ... />
      <SkeletalMesh poseData={poseData} frameMetrics={frameMetrics} currentFrame={currentFrame} renderMode={renderMode} />
      <Grid ... />
    </>
  )
}
```

- [ ] **Step 4: Add render mode toggle button**

In `ThreeJSkeletonViewer`, add `RenderModeToggle` component (next to `CameraPresets`):

```typescript
function RenderModeToggle() {
  const { renderMode, setRenderMode } = useAnalysisStore()
  const t = useTranslations("analysis")
  return (
    <button
      type="button"
      onClick={() => setRenderMode(renderMode === "wireframe" ? "solid" : "wireframe")}
      className="absolute top-2 right-2 rounded-lg px-2 py-1 text-xs"
      style={{ backgroundColor: "oklch(var(--background) / 0.7)" }}
    >
      {renderMode === "wireframe" ? t("solid") : t("wireframe")}
    </button>
  )
}
```

Render it inside the container.

- [ ] **Step 5: Add translations**

`frontend/messages/ru.json` under `"analysis"`:
```json
"wireframe": "Каркас",
"solid": "Сплошной"
```

`frontend/messages/en.json` under `"analysis"`:
```json
"wireframe": "Wireframe",
"solid": "Solid"
```

- [ ] **Step 6: Verify**

```bash
cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check src/components/analysis/skeletal-mesh.tsx src/components/analysis/threejs-skeleton-viewer.tsx src/stores/analysis.ts
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/analysis/skeletal-mesh.tsx frontend/src/components/analysis/threejs-skeleton-viewer.tsx frontend/src/stores/analysis.ts frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(frontend): wireframe/solid toggle on 3D skeleton viewer"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] Share button with clipboard API (Task 1)
- [x] Delete button with confirm + redirect (Task 1)
- [x] Camera presets: front / side / top (Tasks 2-3)
- [x] Wireframe / solid render mode toggle (Task 4)

**2. Placeholder scan:** None. All code provided.

**3. Type consistency:**
- `useDeleteSession` mutation accepts `id: string` — confirmed in `frontend/src/lib/api/sessions.ts`
- `useAnalysisStore` extended with `cameraPreset` and `renderMode` — store file modified in Tasks 2 and 4
- `SkeletalMesh` receives `renderMode` prop — wired in Task 4
- `Sphere` from `@react-three/drei` used for solid joints — available in existing dependency

---

## Execution Handoff

**Plan complete and saved to `data/plans/2026-05-01-session-actions-3d-controls.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review

**Which approach?**
