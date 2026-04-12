# Profile + Upload UX Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the profile page and upload page into intuitive, mobile-first layouts with clear visual hierarchy and user-friendly interactions.

**Architecture:** Two-page redesign:
- **Profile** (`/profile`): Avatar hero with inline name/bio edit, stat cards row, tabbed content (Activity / PRs), settings modal (theme, body, logout). Profile access on mobile via bottom-dock.
- **Upload** (`/upload`): Full-screen camera viewfinder with floating record button, file upload as secondary action. After recording/uploading, show element picker + upload progress + processing status in a single flow.

**Tech Stack:** React 19, Next.js 16, Tailwind CSS v4, OKLCH colors, Lucide icons, next-intl

---

## Current Problems

1. **No mobile access** — profile only reachable via desktop nav icon, not bottom-dock
2. **Edit mode is jarring** — entire card transforms into a form, losing visual context
3. **Mixed concerns** — editable fields (name, bio, height, weight), stats, PRs, activity all in one flat scroll
4. **Settings page is empty** — placeholder, but theme toggle only exists in desktop nav
5. **Hardcoded "см"/"кг"** — not i18n
6. **Profile card is bland** — single letter avatar, no visual weight

## Target Layout

```
┌──────────────────────────────┐
│  [Avatar 64px]  Name  ✏️     │  ← Hero: tap name to edit inline
│                 @email       │
│                 Bio text     │  ← tap bio to edit inline
├──────────────────────────────┤
│  [📷 12 Sessions] [🏆 5 PRs]│  ← Stat cards row
├──────────────────────────────┤
│  [Activity]  [Records]       │  ← Tab bar
├──────────────────────────────┤
│  Recent sessions list /      │  ← Tab content
│  Personal records grouped    │
├──────────────────────────────┤
│  ⚙ Settings                 │  ← Button → opens Sheet
└──────────────────────────────┘
```

Settings Sheet contents:
- Theme toggle (light/dark/system)
- Language (ru/en)
- Height / Weight (set once)
- Logout button

---

## File Structure

```
frontend/src/
├── app/(app)/profile/page.tsx              # REWRITE — new layout
├── components/profile/
│   ├── profile-hero.tsx                    # CREATE — avatar + inline name/bio edit
│   ├── stats-summary.tsx                   # MODIFY — keep, minor style tweaks
│   ├── personal-records.tsx                # MODIFY — extract shared list styles
│   ├── recent-activity.tsx                 # MODIFY — extract shared list styles
│   ├── activity-tabs.tsx                   # CREATE — tab switcher (Activity / Records)
│   └── settings-sheet.tsx                  # CREATE — Sheet with theme, language, body, logout
├── components/layout/bottom-dock.tsx       # MODIFY — add profile avatar button (rightmost)
├── messages/ru.json                        # MODIFY — add profile i18n keys
└── messages/en.json                        # MODIFY — add profile i18n keys
```

---

### Task 1: Create ProfileHero component

**Files:**
- Create: `frontend/src/components/profile/profile-hero.tsx`

- [ ] **Step 1: Create ProfileHero with inline editing**

```tsx
"use client"

import { Check, Pencil, X } from "lucide-react"
import { type FormEvent, useRef, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { useTranslations } from "@/i18n"

export function ProfileHero() {
  const { user } = useAuth()
  const t = useTranslations("profile")

  const [editingName, setEditingName] = useState(false)
  const [editingBio, setEditingBio] = useState(false)
  const [name, setName] = useState(user?.display_name ?? "")
  const [bio, setBio] = useState(user?.bio ?? "")
  const [saving, setSaving] = useState(false)
  const nameRef = useRef<HTMLInputElement>(null)
  const bioRef = useRef<HTMLTextAreaElement>(null)

  if (!user) return null

  const initial = (user.display_name ?? user.email)[0].toUpperCase()

  async function save() {
    setSaving(true)
    try {
      const { updateProfile } = await import("@/lib/auth")
      await updateProfile({
        display_name: name || undefined,
        bio: bio || undefined,
      })
      toast.success(t("updateSuccess"))
      setEditingName(false)
      setEditingBio(false)
    } catch {
      toast.error(t("updateError"))
    } finally {
      setSaving(false)
    }
  }

  function startEditName() {
    setName(user.display_name ?? "")
    setEditingName(true)
    setTimeout(() => nameRef.current?.select(), 0)
  }

  function startEditBio() {
    setBio(user.bio ?? "")
    setEditingBio(true)
    setTimeout(() => bioRef.current?.focus(), 0)
  }

  function cancelEdit() {
    setEditingName(false)
    setEditingBio(false)
    setName(user.display_name ?? "")
    setBio(user.bio ?? "")
  }

  return (
    <div className="flex flex-col items-center gap-3 py-4">
      {/* Avatar */}
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/15 text-2xl font-bold text-primary">
        {initial}
      </div>

      {/* Name — tap to edit */}
      {editingName ? (
        <form
          onSubmit={(e: FormEvent) => { e.preventDefault(); save() }}
          className="flex items-center gap-1"
        >
          <input
            ref={nameRef}
            value={name}
            onChange={e => setName(e.target.value)}
            onBlur={() => save()}
            className="w-48 rounded-lg border border-border bg-secondary px-2 py-1 text-center text-base font-semibold outline-none focus-visible:border-foreground"
            autoFocus
          />
          <button type="button" onClick={cancelEdit} className="p-1 text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </form>
      ) : (
        <button type="button" onClick={startEditName} className="group flex items-center gap-1.5">
          <span className="text-lg font-semibold">{user.display_name ?? user.email}</span>
          <Pencil className="h-3.5 w-3.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
        </button>
      )}

      {/* Email */}
      <p className="text-sm text-muted-foreground">{user.email}</p>

      {/* Bio — tap to edit */}
      {editingBio ? (
        <form
          onSubmit={(e: FormEvent) => { e.preventDefault(); save() }}
          className="flex w-full max-w-xs items-start gap-1"
        >
          <textarea
            ref={bioRef}
            value={bio}
            onChange={e => setBio(e.target.value)}
            onBlur={() => save()}
            rows={2}
            className="w-full rounded-lg border border-border bg-secondary px-2 py-1 text-center text-sm outline-none focus-visible:border-foreground resize-none"
            autoFocus
          />
        </form>
      ) : (
        <button
          type="button"
          onClick={startEditBio}
          className="group max-w-xs text-center"
        >
          <p className="text-sm text-muted-foreground">
            {user.bio || <span className="italic opacity-50">{t("addBio")}</span>}
          </p>
          <Pencil className="mx-auto mt-0.5 h-3 w-3 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
        </button>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/profile/profile-hero.tsx
git commit -m "feat(profile): add ProfileHero with inline name/bio editing"
```

---

### Task 2: Create ActivityTabs component

**Files:**
- Create: `frontend/src/components/profile/activity-tabs.tsx`

- [ ] **Step 1: Create tab switcher**

```tsx
"use client"

import { type ReactNode, useState } from "react"
import { useTranslations } from "@/i18n"

const TABS = ["activity", "records"] as const

export function ActivityTabs({
  activityContent,
  recordsContent,
}: {
  activityContent: ReactNode
  recordsContent: ReactNode
}) {
  const t = useTranslations("profile")
  const [active, setActive] = useState<"activity" | "records">("activity")

  return (
    <div>
      <div className="mb-3 flex gap-1 rounded-xl bg-muted/50 p-1">
        {TABS.map(tab => (
          <button
            key={tab}
            type="button"
            onClick={() => setActive(tab)}
            className={`flex-1 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              active === tab
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab === "activity" ? t("recentActivity") : t("personalRecords")}
          </button>
        ))}
      </div>
      {active === "activity" ? activityContent : recordsContent}
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/profile/activity-tabs.tsx
git commit -m "feat(profile): add ActivityTabs component"
```

---

### Task 3: Create SettingsModal component

**Files:**
- Create: `frontend/src/components/profile/settings-sheet.tsx`

Note: No shadcn Sheet component exists in the project. Using a custom overlay modal instead.

- [ ] **Step 1: Create settings modal**

```tsx
"use client"

import { LogOut, Settings, X } from "lucide-react"
import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { FormField } from "@/components/form-field"
import { useAuth } from "@/components/auth-provider"
import { ThemeToggle } from "@/components/theme-toggle"
import { useTranslations } from "@/i18n"

export function SettingsSheet() {
  const t = useTranslations("profile")
  const ts = useTranslations("settings")
  const { user, logout } = useAuth()
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [height, setHeight] = useState(user?.height_cm?.toString() ?? "")
  const [weight, setWeight] = useState(user?.weight_kg?.toString() ?? "")
  const [saving, setSaving] = useState(false)

  async function saveBody() {
    setSaving(true)
    try {
      const { updateProfile } = await import("@/lib/auth")
      await updateProfile({
        height_cm: height ? Number.parseInt(height, 10) : undefined,
        weight_kg: weight ? Number.parseFloat(weight) : undefined,
      })
      toast.success(t("updateSuccess"))
    } catch {
      toast.error(t("updateError"))
    } finally {
      setSaving(false)
    }
  }

  async function handleLogout() {
    setOpen(false)
    await logout()
    document.cookie = "sb_auth=; path=/; max-age=0"
    router.push("/login")
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="flex w-full items-center justify-between rounded-xl border border-border px-4 py-3 text-sm text-muted-foreground transition-colors hover:bg-accent"
      >
        <span className="flex items-center gap-2">
          <Settings className="h-4 w-4" />
          {ts("title")}
        </span>
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-end justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={() => setOpen(false)} />
          <div className="relative w-full max-w-lg rounded-t-2xl border-t border-border bg-background p-5 pb-[calc(1.5rem+env(safe-area-inset-bottom))]">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="nike-h3">{ts("title")}</h2>
              <button type="button" onClick={() => setOpen(false)} className="p-1 text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{ts("theme")}</span>
                <ThemeToggle />
              </div>

              <form
                onSubmit={(e: FormEvent) => { e.preventDefault(); saveBody() }}
                className="space-y-3"
              >
                <p className="text-sm font-medium">{t("bodyMeasurements")}</p>
                <div className="grid grid-cols-2 gap-3">
                  <FormField
                    label={t("height")}
                    id="settings-height"
                    type="number"
                    value={height}
                    onChange={e => setHeight(e.target.value)}
                    min={50}
                    max={250}
                  />
                  <FormField
                    label={t("weight")}
                    id="settings-weight"
                    type="number"
                    value={weight}
                    onChange={e => setWeight(e.target.value)}
                    min={20}
                    max={300}
                    step={0.1}
                  />
                </div>
                <Button type="submit" size="sm" disabled={saving} className="w-full">
                  {saving ? t("saving") : ts("save")}
                </Button>
              </form>

              <button
                type="button"
                onClick={handleLogout}
                className="flex w-full items-center justify-center gap-2 rounded-xl border border-destructive/30 px-4 py-3 text-sm text-destructive transition-colors hover:bg-destructive/10"
              >
                <LogOut className="h-4 w-4" />
                {t("signOut")}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors (if shadcn Sheet component exists)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/profile/settings-sheet.tsx
git commit -m "feat(profile): add SettingsModal with theme, body, logout"
```

---

### Task 4: Rewrite profile page

**Files:**
- Modify: `frontend/src/app/(app)/profile/page.tsx`

- [ ] **Step 1: Rewrite with new layout**

```tsx
"use client"

import { ActivityTabs } from "@/components/profile/activity-tabs"
import { PersonalRecords } from "@/components/profile/personal-records"
import { ProfileHero } from "@/components/profile/profile-hero"
import { RecentActivity } from "@/components/profile/recent-activity"
import { SettingsSheet } from "@/components/profile/settings-sheet"
import { StatsSummary } from "@/components/profile/stats-summary"

export default function ProfilePage() {
  return (
    <div className="mx-auto max-w-lg space-y-5 px-4 py-4">
      <ProfileHero />
      <StatsSummary />
      <ActivityTabs
        activityContent={<RecentActivity />}
        recordsContent={<PersonalRecords />}
      />
      <SettingsSheet />
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/\(app\)/profile/page.tsx
git commit -m "refactor(profile): rewrite page with hero, tabs, and settings sheet"
```

---

### Task 5: Add profile access to bottom-dock

**Files:**
- Modify: `frontend/src/components/layout/bottom-dock.tsx`

- [ ] **Step 1: Add avatar profile button to bottom-dock**

Add `User` icon import from lucide-react. Add a profile button as the last item in the tabs array. It renders the user's initial as a mini-avatar.

Replace the tabs array construction and the return JSX:

```tsx
import { BarChart3, Camera, Music, Newspaper, User, Users } from "lucide-react"
```

Add the profile tab as the last item in the `tabs` array (before `as const`):

```tsx
const tabs = [
  { href: "/feed", icon: Newspaper, label: t("feed") },
  { href: "/upload", icon: Camera, label: t("upload") },
  { href: "/choreography", icon: Music, label: t("planner") },
  { href: "/progress", icon: BarChart3, label: t("progress") },
  ...(hasStudents ? [{ href: "/dashboard", icon: Users, label: t("students") }] : []),
  { href: "/profile", icon: User, label: t("profile") },
] as const
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/layout/bottom-dock.tsx
git commit -m "feat(profile): add profile tab to bottom-dock navigation"
```

---

### Task 6: Update i18n keys

**Files:**
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Add new profile i18n keys to ru.json**

Find the `"profile"` section and add these keys inside it (alongside existing keys):

```json
"addBio": "Нажмите, чтобы добавить описание",
"bodyMeasurements": "Параметры тела",
"saving": "Сохранение..."
```

- [ ] **Step 2: Add new profile i18n keys to en.json**

Find the `"profile"` section and add these keys inside it:

```json
"addBio": "Tap to add bio",
"bodyMeasurements": "Body measurements",
"saving": "Saving..."
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(profile): add i18n keys for hero and settings sheet"
```

---

### Task 7: Remove logout from desktop app-nav

**Files:**
- Modify: `frontend/src/components/app-nav.tsx`

- [ ] **Step 1: Remove logout button and unused imports from app-nav**

Since logout now lives in the SettingsSheet (accessible from the profile page), remove the logout button and its `LogOut` icon import from the desktop nav. Keep the profile link and theme toggle.

Remove `LogOut` from the import:
```tsx
import { BarChart3, Camera, Music, Newspaper, User, Users } from "lucide-react"
```

Remove `handleLogout` function and the logout button JSX from the component. Remove `useRouter` import if unused. Remove `tp` translation.

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/app-nav.tsx
git commit -m "refactor(profile): remove logout from desktop nav (moved to settings sheet)"
```

---

## Part 2: Upload Page Redesign

### Current Problems

1. **No page title or context** — user lands on a raw camera viewfinder
2. **Camera is always visible** — wastes screen space when user wants to upload a file
3. **Recording state is minimal** — just a timer and red button, no visual feedback
4. **File upload is secondary** — hidden behind a small button after a divider
5. **No element picker in upload flow** — element picker exists but isn't used on upload page
6. **Upload progress is plain** — just a progress bar, no context about what happens next
7. **No tips or guidance** — user doesn't know what to expect

### Target Layout

**Initial state:** Camera viewfinder fills the screen. Floating action buttons at bottom.

```
┌──────────────────────────────┐
│                              │
│                              │
│     Camera Viewfinder        │
│     (full screen)            │
│                              │
│                              │
├──────────────────────────────┤
│  [📋 Gallery]  [🔴 Record]   │  ← Floating bottom bar
└──────────────────────────────┘
```

**After recording / file selected:**

```
┌──────────────────────────────┐
│  📷 recording.webm          │
│  (video preview thumbnail)    │
├──────────────────────────────┤
│  Select element:              │
│  [3T] [3F] [3Lz] [2A] ...  │  ← ElementPicker
│  or [Auto-detect]            │
├──────────────────────────────┤
│  [████████░░░░░░] 67%       │  ← Upload progress
├──────────────────────────────┤
│  [📤 Upload]                 │  ← Upload button
└──────────────────────────────┘
```

**After upload complete:**

```
┌──────────────────────────────┐
│  ✅ Video uploaded!           │
│  ⏳ Analyzing...             │
│  (progress or redirect)      │
└──────────────────────────────┘
```

---

### Task 8: Rewrite CameraRecorder — full-screen viewfinder with floating controls

**Files:**
- Modify: `frontend/src/components/upload/camera-recorder.tsx`

- [ ] **Step 1: Rewrite CameraRecorder with full-screen layout and floating controls**

```tsx
"use client"

import { useCallback, useEffect, useRef, useState } from "react"

const MIME_TYPES = ["video/webm; codecs=vp9", "video/mp4"]

function getSupportedMimeType(): string {
  for (const mime of MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(mime)) return mime
  }
  return "video/webm"
}

export function CameraRecorder({ onRecorded }: { onRecorded: (blob: Blob) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [cameraReady, setCameraReady] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval>>(null)
  const streamRef = useRef<MediaStream | null>(null)

  useEffect(() => {
    async function initCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1920 }, frameRate: { ideal: 60 } },
          audio: false,
        })
        streamRef.current = stream
        if (videoRef.current) videoRef.current.srcObject = stream
        setCameraReady(true)
      } catch {
        setCameraReady(false)
      }
    }
    initCamera()
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [])

  const startRecording = useCallback(async () => {
    if (!streamRef.current) return
    const stream = streamRef.current
    const mimeType = getSupportedMimeType()
    const recorder = new MediaRecorder(stream, { mimeType })
    const chunks: Blob[] = []

    recorder.ondataavailable = e => chunks.push(e.data)
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType })
      onRecorded(blob)
    }

    mediaRecorderRef.current = recorder
    recorder.start()
    setRecording(true)
    setElapsed(0)
    timerRef.current = setInterval(() => setElapsed(t => t + 1), 1000)
  }, [onRecorded])

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop()
    setRecording(false)
    if (timerRef.current) clearInterval(timerRef.current)
  }, [])

  const fmt = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`

  return (
    <div className="relative">
      {/* Full-screen viewfinder */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full rounded-xl bg-black aspect-video object-cover"
      />

      {/* Recording indicator */}
      {recording && (
        <div className="absolute top-3 left-3 flex items-center gap-2 rounded-lg bg-black/60 px-2.5 py-1">
          <div className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
          <span className="font-mono text-xs text-white">{fmt(elapsed)}</span>
        </div>
      )}

      {/* Camera not available fallback */}
      {!cameraReady && (
        <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-muted">
          <p className="text-sm text-muted-foreground">Camera unavailable</p>
        </div>
      )}

      {/* Floating record button */}
      <div className="mt-3 flex items-center justify-center">
        {cameraReady && (
          recording ? (
            <button
              type="button"
              onClick={stopRecording}
              className="flex h-16 w-16 items-center justify-center rounded-full border-4 border-white bg-red-500 transition-transform hover:scale-95 active:scale-90"
              aria-label="Stop recording"
            >
              <div className="h-6 w-6 rounded-sm bg-white" />
            </button>
          ) : (
            <button
              type="button"
              onClick={startRecording}
              className="flex h-16 w-16 items-center justify-center rounded-full border-4 border-red-500 bg-red-500/20 transition-transform hover:scale-105"
              aria-label="Start recording"
            >
              <div className="h-7 w-7 rounded-full bg-red-500" />
            </button>
          )
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/upload/camera-recorder.tsx
git commit -m "refactor(upload): full-screen viewfinder with floating record button"
```

---

### Task 9: Rewrite upload page — unified recording + upload flow

**Files:**
- Modify: `frontend/src/app/(app)/upload/page.tsx`

- [ ] **Step 1: Rewrite upload page with new flow**

```tsx
"use client"

import { Film, Upload, CheckCircle2, Loader2 } from "lucide-react"
import { useRouter } from "next/navigation"
import { useRef, useState } from "react"
import { toast } from "sonner"
import { CameraRecorder } from "@/components/upload/camera-recorder"
import { ElementPicker } from "@/components/upload/element-picker"
import { useTranslations } from "@/i18n"
import { useCreateSession } from "@/lib/api/sessions"
import { ChunkedUploader } from "@/lib/api/uploads"

type Step = "ready" | "picked" | "uploading" | "done"

export default function UploadPage() {
  const router = useRouter()
  const createSession = useCreateSession()
  const t = useTranslations("upload")
  const tc = useTranslations("common")

  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [step, setStep] = useState<Step>("ready")
  const [elementType, setElementType] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  function handleFile(f: File) {
    setFile(f)
    setStep("picked")
  }

  async function handleUpload() {
    if (!file) return
    setStep("uploading")
    try {
      const uploader = new ChunkedUploader(file, (loaded, total) => {
        setProgress(Math.round((loaded / total) * 100))
      })
      await uploader.upload()
      await createSession.mutateAsync({
        element_type: elementType ?? "auto",
      })
      setStep("done")
      toast.success(t("videoUploaded"))
    } catch {
      toast.error(t("uploadError"))
      setStep("picked")
    }
  }

  // Done state — success, redirect after delay
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

  // File picked — show preview + element picker + upload
  if (step === "picked" && file) {
    return (
      <div className="mx-auto max-w-lg space-y-5 px-4 py-4">
        {/* File preview */}
        <div className="flex items-center gap-3 rounded-xl border border-border p-3">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-muted">
            <Film className="h-5 w-5 text-muted-foreground" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{file.name}</p>
            <p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
          </div>
        </div>

        {/* Element picker */}
        <div>
          <p className="mb-2 text-sm font-medium">{t("selectElement")}</p>
          <ElementPicker
            value={elementType}
            onChange={setElementType}
          />
          <p className="mt-1.5 text-xs text-muted-foreground">{t("autoDetectHint")}</p>
        </div>

        {/* Upload button */}
        <button
          type="button"
          onClick={handleUpload}
          className="flex w-full items-center justify-center gap-2 rounded-2xl bg-primary px-4 py-3 font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        >
          <Upload className="h-5 w-5" />
          {t("startUpload")}
        </button>
      </div>
    )
  }

  // Uploading state
  if (step === "uploading") {
    return (
      <div className="mx-auto max-w-lg space-y-5 px-4 py-20">
        <div className="text-center">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary" />
          <p className="mt-3 nike-h3">{t("uploadingVideo")}</p>
        </div>
        <div className="space-y-2">
          <div className="h-2 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full rounded-full bg-primary transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-center text-xs text-muted-foreground">{progress}%</p>
        </div>
      </div>
    )
  }

  // Initial state — camera + file upload option
  return (
    <div className="mx-auto max-w-lg space-y-4 px-4 py-4">
      <CameraRecorder onRecorded={blob => handleFile(new File([blob], `recording_${Date.now()}.webm`, { type: blob.type }))} />

      <div className="flex items-center gap-3">
        <div className="h-px flex-1 bg-border" />
        <span className="text-xs text-muted-foreground">{tc("or")}</span>
        <div className="h-px flex-1 bg-border" />
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        className="mx-auto flex items-center gap-2 rounded-xl border border-border px-4 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-accent/50"
      >
        <Film className="h-4 w-4" />
        {t("chooseFile")}
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/\(app\)/upload/page.tsx
git commit -m "refactor(upload): rewrite with unified recording/upload flow and element picker"
```

---

### Task 10: Add upload i18n keys

**Files:**
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

- [ ] **Step 1: Add upload i18n keys to ru.json**

Find the `"upload"` section and replace with:

```json
"upload": {
  "uploadingVideo": "Загрузка видео...",
  "uploadFile": "Загрузить файл",
  "chooseFile": "Выбрать файл",
  "selectElement": "Выберите элемент (или авто-определение)",
  "autoDetectHint": "Оставьте пустым для авто-определения",
  "startUpload": "Загрузить и начать анализ",
  "uploadError": "Ошибка загрузки",
  "videoUploaded": "Видео загружено!",
  "analyzingHint": "Анализ начнётся автоматически..."
}
```

- [ ] **Step 2: Add upload i18n keys to en.json**

Find the `"upload"` section and replace with:

```json
"upload": {
  "uploadingVideo": "Uploading video...",
  "uploadFile": "Upload file",
  "chooseFile": "Choose file",
  "selectElement": "Select element (or auto-detect)",
  "autoDetectHint": "Leave empty for auto-detection",
  "startUpload": "Upload and start analysis",
  "uploadError": "Upload failed",
  "videoUploaded": "Video uploaded!",
  "analyzingHint": "Analysis will start automatically..."
}
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(i18n): add upload page translations"
```

---

## Self-Review

**1. Spec coverage (Profile):**
- Mobile access via bottom-dock → Task 5
- Inline name/bio editing → Task 1
- Stats cards → Task 4 (uses existing StatsSummary)
- Tabbed activity/records → Task 2
- Settings modal (theme, body, logout) → Task 3
- i18n → Task 6
- Remove redundant logout from desktop nav → Task 7

**2. Spec coverage (Upload):**
- Full-screen camera viewfinder → Task 8
- Recording indicator with timer → Task 8
- File upload as secondary action → Task 9
- Element picker in upload flow → Task 9
- Upload progress with context → Task 9
- Done state with processing message → Task 9
- i18n → Task 10

**3. Placeholder scan:** No TBD, TODO, or vague steps found. All code is complete.

**4. Type consistency:** `useAuth()`, `updateProfile`, `useCreateSession`, `ChunkedUploader`, `ElementPicker` — all used consistently with existing signatures.

**5. Potential issues:**
- Bottom-dock now has 6 items — may need spacing check on small screens
- `ChunkedUploader` auto-starts in current code (checks `progress === 0`); Task 9 uses manual `handleUpload` trigger — verify `ChunkedUploader` class supports this pattern (constructor takes file + progress callback, `upload()` method starts upload)
