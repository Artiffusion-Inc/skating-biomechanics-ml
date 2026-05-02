# Session Detail — Diagnostics, Skeleton Overlay, PDF Export

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose diagnostics, enhanced skeleton overlay (labels + CoM + angles), and browser-native PDF export on the session detail page.

**Architecture:** Reuse existing `useDiagnostics` hook and `DiagnosticsList` component (already used on student page). Add a new `SessionDiagnostics` component that filters findings by session's `element_type`. Enhance `SkeletonCanvas` with mouse tracking for joint labels, CoM dot rendering, and knee-angle arcs. PDF export uses `window.print()` with `@media print` CSS to hide UI chrome.

**Tech Stack:** Next.js 16, React 19, Tailwind CSS v4, Canvas 2D API, Zustand

---

## File Map

| File | Responsibility |
|------|---------------|
| `frontend/src/app/(app)/sessions/[id]/page.tsx` | Import `SessionDiagnostics`, add print button, wire `useDiagnostics` |
| `frontend/src/components/analysis/session-diagnostics.tsx` | NEW. Wrap `DiagnosticsList` with element_type filter + loading state |
| `frontend/src/components/analysis/skeleton-canvas.tsx` | Add mouse tracking, joint labels, CoM dot, angle arcs |
| `frontend/src/app/globals.css` | Add `@media print` rules |

---

## Dependencies

No new npm packages needed. PDF via browser `window.print()`.

---

### Task 1: Session-Level Diagnostics Component

**Files:**

- Create: `frontend/src/components/analysis/session-diagnostics.tsx`
- Modify: `frontend/src/app/(app)/sessions/[id]/page.tsx`

- [ ] **Step 1: Create `SessionDiagnostics` component**

```typescript
"use client"

import { useTranslations } from "@/i18n"
import { useDiagnostics } from "@/lib/api/metrics"
import { DiagnosticsList } from "@/components/coach/diagnostics-list"

interface Props {
  elementType: string
}

export function SessionDiagnostics({ elementType }: Props) {
  const ts = useTranslations("sessions")
  const { data, isLoading } = useDiagnostics()

  if (isLoading) return <div className="h-20 animate-pulse rounded-xl bg-muted" />
  if (!data || !data.findings.length) return null

  const filtered = data.findings.filter(f => f.element === elementType)
  if (!filtered.length) return null

  return (
    <div className="rounded-2xl border border-border p-3 sm:p-4">
      <h2 className="text-sm font-medium mb-2">{ts("recommendations")}</h2>
      <DiagnosticsList findings={filtered} />
    </div>
  )
}
```

- [ ] **Step 2: Wire into page.tsx (right column, after metrics block)**

Add import:
```typescript
import { SessionDiagnostics } from "@/components/analysis/session-diagnostics"
```

Insert after the metrics block:
```tsx
{session.pose_data && (
  <SessionDiagnostics elementType={session.element_type} />
)}
```

- [ ] **Step 3: Verify**

```bash
cd frontend && bunx tsc --noEmit && bunx biome check src/components/analysis/session-diagnostics.tsx 'src/app/(app)/sessions/[id]/page.tsx'
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/analysis/session-diagnostics.tsx frontend/src/app/(app)/sessions/[id]/page.tsx
git commit -m "feat(frontend): show element-filtered diagnostics on session detail"
```

---

### Task 2: SkeletonCanvas — Joint Hover Labels

**Files:**

- Modify: `frontend/src/components/analysis/skeleton-canvas.tsx`

- [ ] **Step 1: Add mouse tracking state + handler**

Add imports:
```typescript
import { useLayoutEffect, useRef, useState, useCallback } from "react"
```

Add state inside component:
```typescript
const [hoverJoint, setHoverJoint] = useState<number | null>(null)
const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 })
```

Add handler:
```typescript
const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
  const canvas = canvasRef.current
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top
  setMousePos({ x: mx, y: my })

  const frameIndex = poseData.frames.indexOf(currentFrame)
  if (frameIndex === -1) return
  const pose = poseData.poses[frameIndex]
  if (!pose) return

  let closest = -1
  let closestDist = Infinity
  for (let i = 0; i < pose.length; i++) {
    const joint = pose[i]
    if (!joint) continue
    const [x, y, conf] = joint
    if (conf < 0.3) continue
    const dx = x * width - mx
    const dy = y * height - my
    const dist = Math.sqrt(dx * dx + dy * dy)
    if (dist < 20 && dist < closestDist) {
      closest = i
      closestDist = dist
    }
  }
  setHoverJoint(closest)
}, [currentFrame, poseData, width, height])

const handleMouseLeave = useCallback(() => setHoverJoint(null), [])
```

- [ ] **Step 2: Draw label in existing `useLayoutEffect`**

After the joint-drawing loop, add:
```typescript
// Draw hover label
if (hoverJoint !== null) {
  const joint = pose[hoverJoint]
  if (joint) {
    const [x, y] = joint
    const px = x * width
    const py = y * height
    ctx.font = "12px Inter, sans-serif"
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
    const text = `Joint ${hoverJoint}`
    const tw = ctx.measureText(text).width
    ctx.fillRect(px + 8, py - 16, tw + 8, 20)
    ctx.fillStyle = "#fff"
    ctx.fillText(text, px + 12, py - 2)
  }
}
```

- [ ] **Step 3: Attach handlers to canvas element**

```tsx
<canvas
  ref={canvasRef}
  width={width}
  height={height}
  className="absolute inset-0"
  onMouseMove={handleMouseMove}
  onMouseLeave={handleMouseLeave}
/>
```

- [ ] **Step 4: Verify**

```bash
cd frontend && bunx tsc --noEmit && bunx biome check src/components/analysis/skeleton-canvas.tsx
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/analysis/skeleton-canvas.tsx
git commit -m "feat(frontend): skeleton canvas joint hover labels"
```

---

### Task 3: SkeletonCanvas — CoM Dot

**Files:**

- Modify: `frontend/src/components/analysis/skeleton-canvas.tsx`

- [ ] **Step 1: Add CoM calculation + draw**

Inside `useLayoutEffect`, after joint drawing loop, add:

```typescript
// Draw CoM (center of mass)
// H3.6M Dempster weights: pelvis (0.0) is center, torso + head = upper body
const pelvis = pose[0]
const thorax = pose[8]
const neck = pose[9]
const head = pose[10]
if (pelvis && thorax && neck && head) {
  const [px, py] = pelvis
  const [tx, ty] = thorax
  const [nx, ny] = neck
  const [hx, hy] = head
  // Approximate CoM: weighted average of torso segments
  const comX = (px * 0.5 + tx * 0.3 + nx * 0.15 + hx * 0.05) * width
  const comY = (py * 0.5 + ty * 0.3 + ny * 0.15 + hy * 0.05) * height
  ctx.beginPath()
  ctx.arc(comX, comY, 6, 0, Math.PI * 2)
  ctx.fillStyle = "#ef4444"
  ctx.fill()
  ctx.strokeStyle = "#fff"
  ctx.lineWidth = 1
  ctx.stroke()
}
```

- [ ] **Step 2: Verify**

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/analysis/skeleton-canvas.tsx
git commit -m "feat(frontend): draw CoM dot on skeleton canvas"
```

---

### Task 4: SkeletonCanvas — Knee Angle Arcs

**Files:**

- Modify: `frontend/src/components/analysis/skeleton-canvas.tsx`

- [ ] **Step 1: Add knee angle drawing utility**

Add helper above component:
```typescript
function drawAngleArc(
  ctx: CanvasRenderingContext2D,
  a: [number, number],
  b: [number, number],
  c: [number, number],
  width: number,
  height: number,
) {
  const angle = Math.atan2(c[1] - b[1], c[0] - b[0]) - Math.atan2(a[1] - b[1], a[0] - b[0])
  const startAngle = Math.atan2(a[1] - b[1], a[0] - b[0])
  const radius = 25
  ctx.beginPath()
  ctx.arc(b[0] * width, b[1] * height, radius, startAngle, startAngle + angle, angle < 0)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.5)"
  ctx.lineWidth = 2
  ctx.stroke()
}
```

Inside `useLayoutEffect`, after connections, add:
```typescript
// Draw knee angles
const rHip = pose[1]
const rKnee = pose[2]
const rAnkle = pose[3]
if (rHip && rKnee && rAnkle) {
  drawAngleArc(ctx, [rHip[0], rHip[1]], [rKnee[0], rKnee[1]], [rAnkle[0], rAnkle[1]], width, height)
}

const lHip = pose[4]
const lKnee = pose[5]
const lAnkle = pose[6]
if (lHip && lKnee && lAnkle) {
  drawAngleArc(ctx, [lHip[0], lHip[1]], [lKnee[0], lKnee[1]], [lAnkle[0], lAnkle[1]], width, height)
}
```

- [ ] **Step 2: Verify**

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/analysis/skeleton-canvas.tsx
git commit -m "feat(frontend): draw knee angle arcs on skeleton canvas"
```

---

### Task 5: Browser-Native PDF Export

**Files:**

- Modify: `frontend/src/app/(app)/sessions/[id]/page.tsx`
- Modify: `frontend/src/app/globals.css`

- [ ] **Step 1: Add print button in page.tsx**

Import `Printer` icon:
```typescript
import { Printer } from "lucide-react"
```

Add print handler:
```typescript
const handlePrint = () => window.print()
```

Add button in right column (before `SessionDownloads` or after recommendations):
```tsx
<button
  type="button"
  onClick={handlePrint}
  className="flex items-center gap-2 rounded-xl border border-border px-3 py-2 text-sm hover:bg-muted print:hidden"
>
  <Printer className="h-4 w-4" />
  {ts("printReport")}
</button>
```

- [ ] **Step 2: Add `@media print` rules in globals.css**

Append to `frontend/src/app/globals.css`:

```css
@media print {
  /* Hide interactive UI */
  .app-nav,
  .bottom-dock,
  button,
  [role="slider"],
  .print-hidden {
    display: none !important;
  }

  /* Ensure page takes full width */
  body {
    background: white !important;
    color: black !important;
  }

  /* Break layout into single column for print */
  .lg\:grid-cols-2 {
    grid-template-columns: 1fr !important;
  }

  /* Avoid page breaks inside key sections */
  .rounded-2xl,
  .rounded-xl {
    break-inside: avoid;
  }

  video {
    max-height: 50vh;
  }
}
```

- [ ] **Step 3: Add translation keys**

`frontend/messages/ru.json` under `"session"`:
```json
"printReport": "Печать отчёта"
```

`frontend/messages/en.json` under `"session"`:
```json
"printReport": "Print Report"
```

- [ ] **Step 4: Verify**

```bash
cd frontend && bunx tsc --noEmit && bunx biome check 'src/app/(app)/sessions/[id]/page.tsx' src/app/globals.css
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/(app)/sessions/[id]/page.tsx frontend/src/app/globals.css frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(frontend): browser-native PDF print export on session detail"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] Diagnostics on session detail (Task 1)
- [x] Skeleton overlay enhancements: hover labels (Task 2), CoM dot (Task 3), knee angle arcs (Task 4)
- [x] PDF export via browser print (Task 5)

**2. Placeholder scan:** None. All code provided.

**3. Type consistency:**
- `useDiagnostics()` returns `{ findings: DiagnosticsFinding[] }` — confirmed in `frontend/src/lib/api/metrics.ts`
- `DiagnosticsList` accepts `findings: DiagnosticsFinding[]` — confirmed in `frontend/src/components/coach/diagnostics-list.tsx`
- `SkeletonCanvas` props unchanged except event handlers added to canvas element

---

## Execution Handoff

**Plan complete and saved to `data/plans/2026-05-01-session-detail-diagnostics-skeleton-pdf.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review

**Which approach?**
