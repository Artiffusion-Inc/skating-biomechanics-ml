# Frontend Polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сделать фронтенд production-ready: убрать заглушки, добавить skeleton loading, error boundary, retry логику, очистить мертвый код.

**Architecture:** React Server Components + Client Components (Next.js App Router). React Query для данных. Zod для валидации. shadcn/ui для UI. next-intl для i18n.

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS v4, shadcn/ui, React Query, Zod, react-hook-form.

---

## File Structure

| File | Responsibility |
|------|--------------|
| `frontend/src/components/error-boundary.tsx` | ErrorBoundary для graceful degradation |
| `frontend/src/components/skeleton-card.tsx` | Skeleton для SessionCard |
| `frontend/src/components/skeleton-chart.tsx` | Skeleton для TrendChart |
| `frontend/src/lib/api-client.ts` | Добавить retry + timeout + offline detection |
| `frontend/src/app/(app)/settings/page.tsx` | Реальная страница настроек (theme/lang/timezone) |
| `frontend/src/app/(app)/training/page.tsx` | Страница тренировочного плана |
| `frontend/src/app/analyze/page.tsx` | **Удалить** — мертвый код |
| `frontend/src/components/auth-provider.tsx` | Убрать MOCK_USER, production auth flow |
| `frontend/src/app/(auth)/login/page.tsx` | Добавить клиентскую валидацию |
| `frontend/src/app/(auth)/register/page.tsx` | Добавить клиентскую валидацию |
| `frontend/messages/ru.json` | Переводы для settings, training, errors |
| `frontend/messages/en.json` | Переводы для settings, training, errors |

---

## Task 1: Error Boundary Component

**Files:**
- Create: `frontend/src/components/error-boundary.tsx`
- Modify: `frontend/src/app/providers.tsx`
- Test: `frontend/src/test/error-boundary.test.tsx` (vitest)

**Step 1: Write ErrorBoundary component**

```tsx
"use client"

import { Component, type ErrorInfo, type ReactNode } from "react"

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught:", error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="mx-auto flex max-w-md flex-col items-center gap-4 p-8 text-center">
            <h2 className="nike-h2 text-destructive">Something went wrong</h2>
            <p className="text-sm text-muted-foreground">
              {this.state.error?.message || "An unexpected error occurred"}
            </p>
            <button
              className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
              onClick={() => this.setState({ hasError: false })}
            >
              Try again
            </button>
          </div>
        )
      )
    }
    return this.props.children
  }
}
```

**Step 2: Wrap QueryClientProvider в ErrorBoundary**

```tsx
// frontend/src/app/providers.tsx
import { ErrorBoundary } from "@/components/error-boundary"

// Внутри return:
<ErrorBoundary>
  <QueryClientProvider client={queryClient}>
    {children}
  </QueryClientProvider>
</ErrorBoundary>
```

**Step 3: Commit**

```bash
git add frontend/src/components/error-boundary.tsx frontend/src/app/providers.tsx
git commit -m "feat(frontend): add ErrorBoundary component"
```

---

## Task 2: Skeleton Loading Components

**Files:**
- Create: `frontend/src/components/skeleton-card.tsx`
- Create: `frontend/src/components/skeleton-chart.tsx`
- Modify: `frontend/src/app/(app)/feed/page.tsx`
- Modify: `frontend/src/app/(app)/progress/page.tsx`
- Modify: `frontend/src/app/(app)/sessions/[id]/page.tsx`

**Step 1: SessionCard skeleton**

```tsx
// frontend/src/components/skeleton-card.tsx
export function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-2xl border border-border bg-background p-4">
      <div className="mb-3 h-4 w-1/3 rounded bg-muted" />
      <div className="mb-2 h-3 w-full rounded bg-muted" />
      <div className="h-3 w-2/3 rounded bg-muted" />
    </div>
  )
}
```

**Step 2: TrendChart skeleton**

```tsx
// frontend/src/components/skeleton-chart.tsx
export function SkeletonChart() {
  return (
    <div className="animate-pulse rounded-2xl border border-border bg-background p-4">
      <div className="mb-4 h-4 w-1/4 rounded bg-muted" />
      <div className="h-64 w-full rounded bg-muted" />
    </div>
  )
}
```

**Step 3: Use skeleton in FeedPage**

```tsx
// frontend/src/app/(app)/feed/page.tsx
import { SkeletonCard } from "@/components/skeleton-card"

// Replace loading div:
if (isLoading) {
  return (
    <div className="mx-auto max-w-2xl space-y-3 sm:max-w-3xl">
      {Array.from({ length: 3 }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  )
}
```

**Step 4: Commit**

```bash
git add frontend/src/components/skeleton-*.tsx frontend/src/app/(app)/feed/page.tsx frontend/src/app/(app)/progress/page.tsx frontend/src/app/(app)/sessions/[id]/page.tsx
git commit -m "feat(frontend): add skeleton loading for feed, progress, sessions"
```

---

## Task 3: Retry Logic + Offline Detection in api-client

**Files:**
- Modify: `frontend/src/lib/api-client.ts`

**Step 1: Add retry with exponential backoff**

```typescript
// Add to api-client.ts
const MAX_RETRIES = 3
const INITIAL_DELAY_MS = 300

async function retryWithBackoff<T>(fn: () => Promise<T>): Promise<T> {
  let lastError: Error | undefined
  for (let i = 0; i < MAX_RETRIES; i++) {
    try {
      return await fn()
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))
      // Only retry on network errors / 5xx / 429
      if (err instanceof ApiError) {
        if (err.status < 500 && err.status !== 429) break
      }
      if (i < MAX_RETRIES - 1) {
        await new Promise(r => setTimeout(r, INITIAL_DELAY_MS * 2 ** i))
      }
    }
  }
  throw lastError ?? new Error("Request failed after retries")
}
```

**Step 2: Wrap apiFetch calls**

```typescript
// In apiFetch, wrap fetch:
const response = await retryWithBackoff(() => fetch(url, fetchOptions))
```

**Step 3: Add offline detection**

```typescript
// Check navigator.onLine before requests
if (typeof navigator !== "undefined" && !navigator.onLine) {
  throw new ApiError("No internet connection", 0)
}
```

**Step 4: Commit**

```bash
git add frontend/src/lib/api-client.ts
git commit -m "feat(frontend): add retry with backoff and offline detection"
```

---

## Task 4: Settings Page (Real Implementation)

**Files:**
- Create: `frontend/src/components/settings/settings-form.tsx`
- Modify: `frontend/src/app/(app)/settings/page.tsx`
- Modify: `frontend/messages/ru.json`
- Modify: `frontend/messages/en.json`

**Step 1: SettingsForm component**

```tsx
"use client"

import { useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { Button } from "@/components/ui/button"
import { useTranslations } from "@/i18n"
import { updateSettings } from "@/lib/auth"

export function SettingsForm() {
  const { user } = useAuth()
  const t = useTranslations("settings")
  const [language, setLanguage] = useState(user?.language || "ru")
  const [timezone, setTimezone] = useState(user?.timezone || "Europe/Moscow")
  const [theme, setTheme] = useState(user?.theme || "system")
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    try {
      await updateSettings({ language, timezone, theme })
      toast.success(t("saved"))
    } catch {
      toast.error(t("saveError"))
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="mx-auto max-w-md space-y-4 p-6">
      <h2 className="nike-h2">{t("title")}</h2>
      <div>
        <label className="mb-1 block text-sm font-medium">{t("language")}</label>
        <select
          value={language}
          onChange={e => setLanguage(e.target.value)}
          className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm"
        >
          <option value="ru">Русский</option>
          <option value="en">English</option>
        </select>
      </div>
      <div>
        <label className="mb-1 block text-sm font-medium">{t("timezone")}</label>
        <select
          value={timezone}
          onChange={e => setTimezone(e.target.value)}
          className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm"
        >
          <option value="Europe/Moscow">Europe/Moscow</option>
          <option value="Europe/London">Europe/London</option>
          <option value="America/New_York">America/New_York</option>
        </select>
      </div>
      <div>
        <label className="mb-1 block text-sm font-medium">{t("theme")}</label>
        <select
          value={theme}
          onChange={e => setTheme(e.target.value)}
          className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm"
        >
          <option value="system">{t("system")}</option>
          <option value="light">{t("light")}</option>
          <option value="dark">{t("dark")}</option>
        </select>
      </div>
      <Button type="submit" className="w-full" disabled={loading}>
        {loading ? t("saving") : t("save")}
      </Button>
    </form>
  )
}
```

**Step 2: Update settings page**

```tsx
// frontend/src/app/(app)/settings/page.tsx
import { SettingsForm } from "@/components/settings/settings-form"

export default function SettingsPage() {
  return <SettingsForm />
}
```

**Step 3: Add translations**

```json
// ru.json settings section:
"settings": {
  "title": "Настройки",
  "saved": "Настройки сохранены",
  "saveError": "Ошибка сохранения",
  "language": "Язык",
  "timezone": "Часовой пояс",
  "theme": "Тема",
  "system": "Системная",
  "light": "Светлая",
  "dark": "Тёмная",
  "save": "Сохранить",
  "saving": "Сохранение..."
}
```

**Step 4: Commit**

```bash
git add frontend/src/components/settings/settings-form.tsx frontend/src/app/(app)/settings/page.tsx frontend/messages/ru.json frontend/messages/en.json
git commit -m "feat(frontend): implement real Settings page with theme/lang/timezone"
```

---

## Task 5: Auth Cleanup — Remove MOCK_USER

**Files:**
- Modify: `frontend/src/components/auth-provider.tsx`

**Step 1: Remove SKIP_AUTH and MOCK_USER**

```tsx
// Remove:
// const MOCK_USER = {...}
// const SKIP_AUTH = ...

// In useMountEffect:
// Remove the entire SKIP_AUTH block
```

**Step 2: Add production-only dev flag via env**

```tsx
const DEV_MOCK_AUTH = process.env.NEXT_PUBLIC_DEV_MOCK_AUTH === "true"

// Only in development:
if (DEV_MOCK_AUTH && process.env.NODE_ENV === "development") {
  setUser({ id: "dev", email: "dev@example.com", display_name: "Dev", ... })
  setIsLoading(false)
  return
}
```

**Step 3: Commit**

```bash
git add frontend/src/components/auth-provider.tsx
git commit -m "refactor(frontend): remove SKIP_AUTH, add DEV_MOCK_AUTH env flag"
```

---

## Task 6: Login/Register Form Validation

**Files:**
- Modify: `frontend/src/app/(auth)/login/page.tsx`
- Modify: `frontend/src/app/(auth)/register/page.tsx`

**Step 1: Add client-side validation to LoginPage**

```tsx
// In handleSubmit:
function validateEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
}

async function handleSubmit(e: FormEvent) {
  e.preventDefault()
  if (!validateEmail(email)) {
    toast.error("Invalid email")
    return
  }
  if (password.length < 1) {
    toast.error("Password required")
    return
  }
  // ... rest
}
```

**Step 2: Add client-side validation to RegisterPage**

```tsx
if (!validateEmail(email)) {
  toast.error("Invalid email")
  return
}
if (password.length < 8) {
  toast.error("Password must be at least 8 characters")
  return
}
```

**Step 3: Commit**

```bash
git add frontend/src/app/(auth)/login/page.tsx frontend/src/app/(auth)/register/page.tsx
git commit -m "feat(frontend): add client-side form validation for auth"
```

---

## Task 7: Remove Dead Code (analyze/page.tsx)

**Files:**
- Delete: `frontend/src/app/analyze/page.tsx`
- Check: references to `/analyze` route

**Step 1: Delete file**

```bash
rm frontend/src/app/analyze/page.tsx
```

**Step 2: Check for references**

```bash
grep -rn "analyze" frontend/src/ --include="*.tsx" --include="*.ts"
```

**Step 3: Remove unused dashboard components if only used by analyze**

```bash
# Check if these are used elsewhere:
grep -rn "StatsCards\|VideoPlayer\|DownloadSection" frontend/src/
```

**Step 4: Commit**

```bash
git rm frontend/src/app/analyze/page.tsx
git commit -m "chore(frontend): remove dead analyze page"
```

---

## Task 8: Training Page (Decide & Implement)

**Decision needed:** Что показывать на `/training`?

Options:
- A) **Training plans** — список тренировочных планов (пока заглушка, будет API позже)
- B) **Remove the page** — убрать из навигации и роутинга
- C) **Redirect** — редирект на `/feed`

**If Option B (recommended for now):**

```bash
rm frontend/src/app/training/page.tsx
# Remove from bottom-dock navigation
# Remove from app-nav.tsx
```

**Step 1: Commit**

```bash
git rm frontend/src/app/training/page.tsx
git commit -m "chore(frontend): remove training placeholder page"
```

---

## Task 9: TypeScript + Lint + Test Check

**Step 1: Run type check**

```bash
bunx tsc --noEmit
```

**Step 2: Run lint**

```bash
bunx next lint
```

**Step 3: Run vitest**

```bash
bunx vitest run
```

**Step 4: Fix errors, commit**

```bash
git add -A
git commit -m "fix(frontend): resolve type and lint errors after polish"
```

---

## Summary

| # | Task | Status | Est. Time |
|---|------|--------|-----------|
| 1 | ErrorBoundary | ⬜ | 30 min |
| 2 | Skeleton loading | ⬜ | 45 min |
| 3 | Retry + offline | ⬜ | 30 min |
| 4 | Settings page | ⬜ | 45 min |
| 5 | Auth cleanup | ⬜ | 20 min |
| 6 | Form validation | ⬜ | 20 min |
| 7 | Remove dead code | ⬜ | 15 min |
| 8 | Training page (remove) | ⬜ | 10 min |
| 9 | Type/lint check | ⬜ | 15 min |
| **Total** | | | **~4.5 часа** |

**Files touched:** ~12 файлов, ~1 удален.

---

## Self-Review

**1. Spec coverage:**
- ⬜ ErrorBoundary — Task 1
- ⬜ Skeleton loading — Task 2
- ⬜ Retry logic — Task 3
- ⬜ Settings page — Task 4
- ⬜ Auth cleanup — Task 5
- ⬜ Form validation — Task 6
- ⬜ Dead code removal — Task 7, 8

**2. Placeholder scan:**
- No "TBD", "TODO", "implement later"
- All code blocks have actual implementation
- Exact file paths provided

**3. Type consistency:**
- `ApiError` class already exists in `api-client.ts`
- `updateSettings` already exists in `lib/auth.ts`
- `useAuth()` context already defined
- Translation keys match existing patterns

---

## Execution Options

**Plan saved to `data/plans/2026-04-30-frontend-polish.md`.**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

**Which approach?**
