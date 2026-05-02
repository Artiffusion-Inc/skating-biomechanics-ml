# Session Detail Deep Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Session Detail page production-ready with bidirectional video/graph/3D sync, lazy-loaded 3D viewer, desktop two-column layout, and download buttons.

**Architecture:** Extend the existing `useAnalysisStore` (Zustand) as the single source of truth for `currentFrame` and `isPlaying`. Wire `VideoWithSkeleton` to listen to store changes (seek + play/pause). Lazy-load `ThreeJSkeletonViewer` with `Suspense`. Split the page layout into two columns on desktop (`lg:grid-cols-2`). Reuse the existing `SessionDownloads` component.

**Tech Stack:** Next.js 16, React 19, Zustand, Recharts, three.js (lazy), Tailwind CSS v4, next-intl

---

## File Map

| File | Responsibility |
|------|---------------|
| `frontend/src/components/analysis/video-with-skeleton.tsx` | Video playback + skeleton overlay. Needs bidirectional frame sync and play/pause sync with store. |
| `frontend/src/components/analysis/threejs-skeleton-viewer.tsx` | 3D skeleton viewer. Needs `lazy()` import wrapper + play/pause controls overlay. |
| `frontend/src/stores/analysis.ts` | Zustand store. Already has `isPlaying` — no changes needed. |
| `frontend/src/app/(app)/sessions/[id]/page.tsx` | Session detail page. Needs two-column grid layout + `SessionDownloads` insertion. |
| `frontend/src/components/session/session-downloads.tsx` | Download buttons (video, poses, CSV). Already implemented — just import on page. |

---

### Task 1: Bidirectional Video ↔ Store Sync

**Files:**

- Modify: `frontend/src/components/analysis/video-with-skeleton.tsx`

- [ ] **Step 1: Add `fps` prop and `useEffect` for external frame seeking**

Add `fps: number` to props interface. Add a `useEffect` that reacts to `currentFrame` changes from the store and updates `video.currentTime` to `currentFrame / fps`. Add a second `useEffect` that reacts to `isPlaying` and calls `video.play()` or `video.pause()`.

```typescript
interface VideoWithSkeletonProps {
  videoUrl: string
  poseData: PoseData | null
  phases: PhasesData | null
  totalFrames: number
  fps?: number
  className?: string
}

export function VideoWithSkeleton({
  videoUrl,
  poseData,
  phases,
  totalFrames,
  fps = 30,
  className = "",
}: VideoWithSkeletonProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { currentFrame, setCurrentFrame, isPlaying, setIsPlaying } = useAnalysisStore()

  // Existing handleTimeUpdate
  const handleTimeUpdate = () => {
    if (!videoRef.current) return
    const video = videoRef.current
    const frame = Math.floor((video.currentTime / video.duration) * totalFrames)
    setCurrentFrame(frame)
  }

  // NEW: sync store → video time
  useEffect(() => {
    const video = videoRef.current
    if (!video || !video.duration) return
    const targetTime = currentFrame / fps
    if (Math.abs(video.currentTime - targetTime) > 1 / fps) {
      video.currentTime = targetTime
    }
  }, [currentFrame, fps])

  // NEW: sync store → video play/pause
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    if (isPlaying) {
      video.play().catch(() => {})
    } else {
      video.pause()
    }
  }, [isPlaying])

  const handleClick = () => {
    setIsPlaying(!isPlaying)
  }

  // ... rest unchanged, but add onClick={handleClick} to video wrapper when !poseData
```

- [ ] **Step 2: Wire `fps` prop from page**

Pass `fps={session.pose_data.fps}` to `<VideoWithSkeleton>` in `frontend/src/app/(app)/sessions/[id]/page.tsx`.

- [ ] **Step 3: Verify `bunx tsc --noEmit` passes**

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/analysis/video-with-skeleton.tsx frontend/src/app/(app)/sessions/[id]/page.tsx
git commit -m "feat(frontend): bidirectional video sync with analysis store"
```

---

### Task 2: Play/Pause Overlay on Skeleton Video

**Files:**

- Modify: `frontend/src/components/analysis/video-with-skeleton.tsx`

- [ ] **Step 1: Add overlay play/pause button**

When `poseData` is present, show a centered play/pause button overlay that toggles `isPlaying` in the store. Hide the overlay after 2 seconds of inactivity (use a local `showControls` state with `setTimeout`).

```typescript
import { useState, useCallback } from "react"
import { Play, Pause } from "lucide-react"

// Inside component:
const [showControls, setShowControls] = useState(true)
let hideTimeout = useRef<NodeJS.Timeout | null>(null)

const revealControls = useCallback(() => {
  setShowControls(true)
  if (hideTimeout.current) clearTimeout(hideTimeout.current)
  hideTimeout.current = setTimeout(() => setShowControls(false), 2000)
}, [])

// In the return (inside the relative div):
{poseData && (
  <div
    className={`absolute inset-0 flex items-center justify-center transition-opacity duration-300 ${showControls ? "opacity-100" : "opacity-0"}`}
    onClick={() => { setIsPlaying(!isPlaying); revealControls() }}
    onMouseMove={revealControls}
  >
    <button className="rounded-full bg-black/50 p-4 text-white hover:bg-black/70">
      {isPlaying ? <Pause className="h-8 w-8" /> : <Play className="h-8 w-8" />}
    </button>
  </div>
)}
```

- [ ] **Step 2: Run biome check**

```bash
cd frontend && bunx biome check src/components/analysis/video-with-skeleton.tsx
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/analysis/video-with-skeleton.tsx
git commit -m "feat(frontend): play/pause overlay on skeleton video"
```

---

### Task 3: Lazy-Load ThreeJSkeletonViewer

**Files:**

- Modify: `frontend/src/app/(app)/sessions/[id]/page.tsx`

- [ ] **Step 1: Replace direct import with `lazy()` + `Suspense`**

```typescript
import { lazy, Suspense } from "react"

const ThreeJSkeletonViewer = lazy(() => import("@/components/analysis/threejs-skeleton-viewer").then(m => ({ default: m.ThreeJSkeletonViewer })))
```

- [ ] **Step 2: Wrap the component in `Suspense` with a skeleton fallback**

Where `ThreeJSkeletonViewer` is rendered:

```tsx
<Suspense fallback={
  <div className="aspect-square rounded-xl bg-muted animate-pulse" />
}>
  <ThreeJSkeletonViewer
    poseData={session.pose_data}
    frameMetrics={session.frame_metrics}
    className="rounded-xl"
  />
</Suspense>
```

- [ ] **Step 3: Run biome + tsc**

```bash
cd frontend && bunx biome check src/app/(app)/sessions/[id]/page.tsx && bunx tsc --noEmit
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/(app)/sessions/[id]/page.tsx
git commit -m "feat(frontend): lazy-load 3D skeleton viewer with Suspense"
```

---

### Task 4: Desktop Two-Column Layout

**Files:**

- Modify: `frontend/src/app/(app)/sessions/[id]/page.tsx`

- [ ] **Step 1: Replace single-column wrapper with two-column grid**

Change the outer wrapper from:

```tsx
<div className="mx-auto max-w-2xl space-y-6 sm:max-w-3xl">
```

To:

```tsx
<div className="mx-auto max-w-2xl space-y-6 lg:max-w-none lg:grid lg:grid-cols-2 lg:gap-6">
  {/* Left column */}
  <div className="space-y-6">
    {/* header, VideoWithSkeleton, PhaseTimeline, FrameMetricsChart */}
  </div>
  {/* Right column */}
  <div className="space-y-6">
    {/* ThreeJSkeletonViewer, metrics, recommendations, downloads */}
  </div>
</div>
```

Move the following into the **left** column:
- Header (element type, date, score)
- `VideoWithSkeleton`
- `PhaseTimeline`
- `FrameMetricsChart`

Move the following into the **right** column:
- `ThreeJSkeletonViewer` (lazy + Suspense)
- Metrics block (`session.metrics.length > 0`)
- Recommendations block
- `SessionDownloads`

Keep the fallback raw `<video>` elements in the left column as-is.

- [ ] **Step 2: Run biome + tsc**

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/(app)/sessions/[id]/page.tsx
git commit -m "feat(frontend): two-column desktop layout for session detail"
```

---

### Task 5: Wire SessionDownloads Component

**Files:**

- Modify: `frontend/src/app/(app)/sessions/[id]/page.tsx`

- [ ] **Step 1: Import and insert `SessionDownloads`**

```typescript
import { SessionDownloads } from "@/components/session/session-downloads"
```

Insert in the **right column**, after recommendations:

```tsx
<SessionDownloads
  videoUrl={session.processed_video_url ?? session.video_url}
  posesUrl={session.poses_url}
  csvUrl={session.csv_url}
/>
```

- [ ] **Step 2: Run biome + tsc**

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/(app)/sessions/[id]/page.tsx
git commit -m "feat(frontend): wire session download buttons on detail page"
```

---

### Task 6: Play/Pause Button on ThreeJSkeletonViewer

**Files:**

- Modify: `frontend/src/components/analysis/threejs-skeleton-viewer.tsx`

- [ ] **Step 1: Add play/pause + frame advance controls overlay**

Add a bottom-center control bar inside the viewer container with:
- Rewind 10 frames button (`ChevronLeft`)
- Play/Pause button (`Play`/`Pause`)
- Forward 10 frames button (`ChevronRight`)
- Speed selector (0.5x, 1x, 2x)

```tsx
import { Play, Pause, ChevronLeft, ChevronRight } from "lucide-react"

function PlaybackControls() {
  const { isPlaying, setIsPlaying, currentFrame, setCurrentFrame, playbackSpeed, setPlaybackSpeed } = useAnalysisStore()
  return (
    <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex items-center gap-2 rounded-full px-3 py-1.5 text-xs"
      style={{ backgroundColor: "oklch(var(--background) / 0.7)" }}
    >
      <button onClick={() => setCurrentFrame(Math.max(0, currentFrame - 10))}><ChevronLeft className="h-4 w-4" /></button>
      <button onClick={() => setIsPlaying(!isPlaying)}>
        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      </button>
      <button onClick={() => setCurrentFrame(currentFrame + 10)}><ChevronRight className="h-4 w-4" /></button>
      <select
        value={playbackSpeed}
        onChange={e => setPlaybackSpeed(Number(e.target.value))}
        className="bg-transparent text-xs outline-none"
      >
        <option value={0.5}>0.5x</option>
        <option value={1}>1x</option>
        <option value={2}>2x</option>
      </select>
    </div>
  )
}
```

Render `<PlaybackControls />` inside the `ThreeJSkeletonViewer` return (after `Canvas`, inside the same relative container).

- [ ] **Step 2: Run biome + tsc**

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/analysis/threejs-skeleton-viewer.tsx
git commit -m "feat(frontend): playback controls on 3D skeleton viewer"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] Bidirectional video sync (Task 1)
- [x] Play/pause overlay on video (Task 2)
- [x] Lazy-load 3D viewer (Task 3)
- [x] Desktop two-column layout (Task 4)
- [x] Download buttons (Task 5)
- [x] 3D viewer playback controls (Task 6)

**2. Placeholder scan:** None found. Every step includes exact code.

**3. Type consistency:**
- `PoseData.fps` used for time calculation (defined in `frontend/src/types/index.ts:52`)
- `useAnalysisStore` fields: `currentFrame`, `isPlaying`, `playbackSpeed`, `setCurrentFrame`, `setIsPlaying`, `setPlaybackSpeed` — all confirmed in `frontend/src/stores/analysis.ts`
- `SessionDownloads` props match the component interface (`videoUrl`, `posesUrl`, `csvUrl`)

---

## Execution Handoff

**Plan complete and saved to `data/plans/2026-05-01-session-detail-deep-visualization.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review

**Which approach?**
