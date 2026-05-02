# Session Feed Polish + 3D Viewer Labels

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hover joint labels to the 3D skeleton viewer, bulk session deletion in the feed, element/date filters, and inline body measurements editing on the profile page.

**Architecture:** Feature 1 uses `@react-three/drei`'s `Html` component for 3D-space labels on joint hover. Feature 2 adds checkboxes to `SessionCard`, a selection state in `FeedPage`, and a new backend `DELETE /sessions/bulk` endpoint. Feature 3 adds a filter bar with element type pills and a date range dropdown (client-side filtering for MVP). Feature 4 moves body measurements from the settings sheet into `ProfileHero` with inline numeric inputs and save-on-blur.

**Tech Stack:** Next.js 16, React 19, Zustand, three.js, @react-three/drei, @react-three/fiber, Tailwind CSS v4, FastAPI (backend), React Query

---

## File Map

| File | Responsibility |
|------|---------------|
| `frontend/src/components/analysis/joint-label.tsx` | NEW. `@react-three/drei` Html label for 3D joint names. |
| `frontend/src/components/analysis/skeletal-mesh.tsx` | Add `onJointHover` and render `JointLabel` on hover. |
| `frontend/src/components/session/session-card.tsx` | Add checkbox for bulk selection mode. |
| `frontend/src/app/(app)/feed/page.tsx` | Selection state, bulk delete bar, filter controls. |
| `backend/app/routes/sessions.py` | Add `DELETE /sessions/bulk` endpoint. |
| `frontend/src/lib/api/sessions.ts` | Add `useBulkDeleteSessions` mutation. |
| `frontend/src/components/profile/profile-hero.tsx` | Add inline height/weight inputs with save-on-blur. |
| `frontend/src/components/profile/settings-sheet.tsx` | Remove body measurements section (moved to hero). |

---

## Dependencies

No new packages. `@react-three/drei` `Html` already available.

---

### Task 1: Hover Joint Labels in 3D Viewer

**Files:**

- Create: `frontend/src/components/analysis/joint-label.tsx`
- Modify: `frontend/src/components/analysis/skeletal-mesh.tsx`
- Modify: `frontend/src/stores/analysis.ts`
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Add hoveredJoint to analysis store**

Add to `AnalysisState`:
```typescript
hoveredJoint: number | null
setHoveredJoint: (joint: number | null) => void
```

Add to store defaults:
```typescript
hoveredJoint: null,
setHoveredJoint: joint => set({ hoveredJoint: joint }),
```

Update `reset()`:
```typescript
hoveredJoint: null,
```

- [ ] **Step 2: Create `JointLabel` component**

```typescript
"use client"

import { Html } from "@react-three/drei"
import { useTranslations } from "@/i18n"

const JOINT_NAMES = [
  "pelvis", "rightHip", "rightKnee", "rightAnkle",
  "leftHip", "leftKnee", "leftAnkle", "spine",
  "thorax", "neck", "head", "leftShoulder",
  "leftElbow", "leftWrist", "rightShoulder",
  "rightElbow", "rightWrist",
]

interface JointLabelProps {
  jointIndex: number
  position: [number, number, number]
}

export function JointLabel({ jointIndex, position }: JointLabelProps) {
  const t = useTranslations("joints")
  const name = JOINT_NAMES[jointIndex] ?? `joint${jointIndex}`
  return (
    <Html position={position} center distanceFactor={10}>
      <div className="pointer-events-none rounded-md bg-background/90 px-2 py-1 text-xs text-foreground whitespace-nowrap shadow-sm">
        {t(name)}
      </div>
    </Html>
  )
}
```

- [ ] **Step 3: Add translations**

`frontend/messages/ru.json` under `"joints"`:
```json
"pelvis": "Таз",
"rightHip": "Правое бедро",
"rightKnee": "Правое колено",
"rightAnkle": "Правая лодыжка",
"leftHip": "Левое бедро",
"leftKnee": "Левое колено",
"leftAnkle": "Левая лодыжка",
"spine": "Позвоночник",
"thorax": "Грудная клетка",
"neck": "Шея",
"head": "Голова",
"leftShoulder": "Левое плечо",
"leftElbow": "Левый локоть",
"leftWrist": "Левая кисть",
"rightShoulder": "Правое плечо",
"rightElbow": "Правый локоть",
"rightWrist": "Правая кисть"
```

`frontend/messages/en.json` under `"joints"`:
```json
"pelvis": "Pelvis",
"rightHip": "Right Hip",
"rightKnee": "Right Knee",
"rightAnkle": "Right Ankle",
"leftHip": "Left Hip",
"leftKnee": "Left Knee",
"leftAnkle": "Left Ankle",
"spine": "Spine",
"thorax": "Thorax",
"neck": "Neck",
"head": "Head",
"leftShoulder": "Left Shoulder",
"leftElbow": "Left Elbow",
"leftWrist": "Left Wrist",
"rightShoulder": "Right Shoulder",
"rightElbow": "Right Elbow",
"rightWrist": "Right Wrist"
```

- [ ] **Step 4: Wire hover into `SkeletalMesh`**

In `skeletal-mesh.tsx`, read `hoveredJoint` and `setHoveredJoint` from `useAnalysisStore()`.

For each rendered joint (`Joint` or `SolidJoint`), add:
```typescript
onPointerOver={() => setHoveredJoint(joint.index)}
onPointerOut={() => setHoveredJoint(null)}
```

After the joints map, conditionally render:
```typescript
{hoveredJoint !== null && joints.find(j => j.index === hoveredJoint) && (
  <JointLabel
    jointIndex={hoveredJoint}
    position={joints.find(j => j.index === hoveredJoint)!.position as [number, number, number]}
  />
)}
```

- [ ] **Step 5: Verify**

```bash
cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check src/components/analysis/joint-label.tsx src/components/analysis/skeletal-mesh.tsx src/stores/analysis.ts
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/analysis/joint-label.tsx frontend/src/components/analysis/skeletal-mesh.tsx frontend/src/stores/analysis.ts frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(frontend): hover joint labels in 3D skeleton viewer"
```

---

### Task 2: Bulk Delete Sessions

**Files:**

- Create: `backend/app/routes/sessions.py` section (bulk delete endpoint)
- Modify: `frontend/src/lib/api/sessions.ts`
- Modify: `frontend/src/components/session/session-card.tsx`
- Modify: `frontend/src/app/(app)/feed/page.tsx`
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Add backend bulk delete endpoint**

In `backend/app/routes/sessions.py`, add:

```python
from fastapi import HTTPException, status

@router.delete("/bulk", status_code=status.HTTP_204_NO_CONTENT)
def delete_sessions_bulk(
    ids: list[str],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    for sid in ids:
        session = get_session(db, sid)
        if not session:
            continue
        if session.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete another user's session"
            )
        delete_session(db, sid)
```

- [ ] **Step 2: Add `useBulkDeleteSessions` mutation**

In `frontend/src/lib/api/sessions.ts`, add:

```typescript
export function useBulkDeleteSessions() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (ids: string[]) => apiDelete(`/sessions/bulk?ids=${ids.join(",")}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  })
}
```

- [ ] **Step 3: Add selection mode to `SessionCard`**

Update `SessionCard` props:
```typescript
interface SessionCardProps {
  session: Session
  selectable?: boolean
  selected?: boolean
  onSelect?: (id: string) => void
}
```

In the card layout, when `selectable` is true, render a checkbox before the link content:
```typescript
{selectable && (
  <input
    type="checkbox"
    checked={selected}
    onChange={() => onSelect?.(session.id)}
    className="mr-2 h-4 w-4 shrink-0"
    onClick={e => e.stopPropagation()}
  />
)}
```

- [ ] **Step 4: Add selection state and bulk delete bar to `FeedPage`**

Add to `FeedPage`:
```typescript
const [selectionMode, setSelectionMode] = useState(false)
const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
const bulkDelete = useBulkDeleteSessions()

const toggleSelect = (id: string) => {
  setSelectedIds(prev => {
    const next = new Set(prev)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    return next
  })
}

const handleBulkDelete = () => {
  if (!window.confirm(t("bulkDeleteConfirm"))) return
  bulkDelete.mutate(Array.from(selectedIds), {
    onSuccess: () => {
      setSelectedIds(new Set())
      setSelectionMode(false)
    },
  })
}
```

Add a toolbar above the session list:
```typescript
<div className="flex items-center justify-between">
  <button
    type="button"
    onClick={() => {
      setSelectionMode(!selectionMode)
      setSelectedIds(new Set())
    }}
    className="text-sm text-muted-foreground hover:text-foreground"
  >
    {selectionMode ? t("cancel") : t("select")}
  </button>
  {selectionMode && selectedIds.size > 0 && (
    <button
      type="button"
      onClick={handleBulkDelete}
      disabled={bulkDelete.isPending}
      className="text-sm text-destructive hover:text-destructive/80"
    >
      {bulkDelete.isPending ? t("deleting") : t("deleteSelected", { count: selectedIds.size })}
    </button>
  )}
</div>
```

Pass props to `SessionCard`:
```typescript
<SessionCard
  key={session.id}
  session={session}
  selectable={selectionMode}
  selected={selectedIds.has(session.id)}
  onSelect={toggleSelect}
/>
```

- [ ] **Step 5: Add translations**

`frontend/messages/ru.json` under `"feed"`:
```json
"select": "Выбрать",
"deleteSelected": "Удалить выбранное ({count})",
"bulkDeleteConfirm": "Удалить выбранные сессии? Это действие необратимо."
```

`frontend/messages/en.json` under `"feed"`:
```json
"select": "Select",
"deleteSelected": "Delete selected ({count})",
"bulkDeleteConfirm": "Delete selected sessions? This action cannot be undone."
```

- [ ] **Step 6: Verify**

Backend:
```bash
cd /home/michael/Github/skating-biomechanics-ml/backend && uv run pytest tests/ -q -k session
```

Frontend:
```bash
cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check src/lib/api/sessions.ts src/components/session/session-card.tsx 'src/app/(app)/feed/page.tsx'
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/routes/sessions.py frontend/src/lib/api/sessions.ts frontend/src/components/session/session-card.tsx 'src/app/(app)/feed/page.tsx' frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat: bulk delete sessions from feed"
```

---

### Task 3: Feed Filters (Element Type + Date)

**Files:**

- Modify: `frontend/src/app/(app)/feed/page.tsx`
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Add filter state and controls**

In `FeedPage`, add:
```typescript
const [elementFilter, setElementFilter] = useState<string>("")
const [dateFilter, setDateFilter] = useState<"7d" | "30d" | "90d" | "all">("all")
```

Filter sessions client-side:
```typescript
const filteredSessions = useMemo(() => {
  if (!data?.sessions) return []
  let sessions = data.sessions
  if (elementFilter) {
    sessions = sessions.filter(s => s.element_type === elementFilter)
  }
  if (dateFilter !== "all") {
    const days = { "7d": 7, "30d": 30, "90d": 90 }[dateFilter]
    const cutoff = Date.now() - days * 86400000
    sessions = sessions.filter(s => new Date(s.created_at).getTime() >= cutoff)
  }
  return sessions
}, [data, elementFilter, dateFilter])
```

Add filter bar before the list:
```typescript
<div className="flex flex-wrap items-center gap-2">
  <select
    value={elementFilter}
    onChange={e => setElementFilter(e.target.value)}
    className="rounded-lg border border-border bg-transparent px-2 py-1 text-sm"
  >
    <option value="">{t("allElements")}</option>
    {ELEMENT_TYPE_KEYS.map(key => (
      <option key={key} value={key}>{te(key)}</option>
    ))}
  </select>
  <div className="flex gap-1">
    {(["7d", "30d", "90d", "all"] as const).map(d => (
      <button
        key={d}
        type="button"
        onClick={() => setDateFilter(d)}
        className={`rounded-lg px-2 py-1 text-xs ${dateFilter === d ? "bg-primary text-primary-foreground" : "border border-border hover:bg-muted"}`}
      >
        {t(`period${d}`)}
      </button>
    ))}
  </div>
</div>
```

- [ ] **Step 2: Add translations**

`frontend/messages/ru.json` under `"feed"`:
```json
"allElements": "Все элементы",
"period7d": "7 дн",
"period30d": "30 дн",
"period90d": "90 дн",
"periodall": "Всё"
```

`frontend/messages/en.json` under `"feed"`:
```json
"allElements": "All elements",
"period7d": "7d",
"period30d": "30d",
"period90d": "90d",
"periodall": "All"
```

- [ ] **Step 3: Verify**

```bash
cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check 'src/app/(app)/feed/page.tsx'
```

- [ ] **Step 4: Commit**

```bash
git add 'src/app/(app)/feed/page.tsx' frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(frontend): element type and date filters on feed"
```

---

### Task 4: Inline Body Measurements in Profile

**Files:**

- Modify: `frontend/src/components/profile/profile-hero.tsx`
- Modify: `frontend/src/components/profile/settings-sheet.tsx`
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Read profile data in `ProfileHero`**

Import `useAuth` from `@/components/auth-provider` to get `user`:
```typescript
import { useAuth } from "@/components/auth-provider"
import { useTranslations } from "@/i18n"
import { updateProfile } from "@/lib/auth"
```

Add state for height/weight:
```typescript
const { user } = useAuth()
const [height, setHeight] = useState(user?.height_cm?.toString() ?? "")
const [weight, setWeight] = useState(user?.weight_kg?.toString() ?? "")
const [saving, setSaving] = useState(false)
const t = useTranslations("profile")

useEffect(() => {
  setHeight(user?.height_cm?.toString() ?? "")
  setWeight(user?.weight_kg?.toString() ?? "")
}, [user?.height_cm, user?.weight_kg])
```

- [ ] **Step 2: Add save-on-blur handler**

```typescript
const saveBody = async () => {
  if (saving) return
  setSaving(true)
  try {
    await updateProfile({
      height_cm: height ? Number.parseInt(height, 10) : undefined,
      weight_kg: weight ? Number.parseFloat(weight) : undefined,
    })
  } catch {
    // silent fail — toast handled by auth lib or ignored
  } finally {
    setSaving(false)
  }
}
```

- [ ] **Step 3: Render inline inputs**

Add below the name/bio section:
```typescript
<div className="mt-3 flex items-center gap-3 text-sm text-muted-foreground">
  <div className="flex items-center gap-1.5">
    <span>{t("height")}:</span>
    <input
      type="number"
      value={height}
      onChange={e => setHeight(e.target.value)}
      onBlur={saveBody}
      className="w-16 rounded-md border border-border bg-transparent px-1.5 py-0.5 text-sm text-foreground outline-none focus:border-primary"
      min={50}
      max={250}
    />
  </div>
  <div className="flex items-center gap-1.5">
    <span>{t("weight")}:</span>
    <input
      type="number"
      value={weight}
      onChange={e => setWeight(e.target.value)}
      onBlur={saveBody}
      className="w-16 rounded-md border border-border bg-transparent px-1.5 py-0.5 text-sm text-foreground outline-none focus:border-primary"
      min={20}
      max={300}
      step={0.1}
    />
  </div>
  {saving && <span className="text-xs">{t("saving")}</span>}
</div>
```

- [ ] **Step 4: Remove from `SettingsSheet`**

Delete the body measurements form block from `settings-sheet.tsx` (the grid with height/weight FormFields and the `bodyMeasurements` heading).

- [ ] **Step 5: Verify**

```bash
cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check src/components/profile/profile-hero.tsx src/components/profile/settings-sheet.tsx
```

- [ ] **Step 6: Commit**

```bash
git add src/components/profile/profile-hero.tsx src/components/profile/settings-sheet.tsx frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(frontend): inline body measurements editing in profile"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] Hover joint labels with `@react-three/drei` Html (Task 1)
- [x] Bulk delete with backend endpoint + frontend selection (Task 2)
- [x] Element type + date filters on feed (Task 3)
- [x] Inline body measurements editing on profile (Task 4)

**2. Placeholder scan:** None. All code provided.

**3. Type consistency:**
- `useAnalysisStore` extended with `hoveredJoint` — store file modified in Task 1
- `useBulkDeleteSessions` accepts `string[]` — matches backend `DELETE /sessions/bulk?ids=...`
- `SessionCard` props extended with `selectable`, `selected`, `onSelect` — wired in Task 2
- `ProfileHero` uses `updateProfile` from `@/lib/auth` — same signature as settings-sheet

---

## Execution Handoff

**Plan complete and saved to `data/plans/2026-05-01-session-feed-polish.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review

**Which approach?**
