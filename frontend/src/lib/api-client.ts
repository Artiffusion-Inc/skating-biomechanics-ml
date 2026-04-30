/**
 * Shared API infrastructure: base URL, token storage, typed fetch helper.
 */

import { redirect } from "next/navigation"
import type { z } from "zod"

export const API_BASE = "/api/v1"

export const SKIP_AUTH = process.env.NEXT_PUBLIC_SKIP_AUTH === "true"

// ---------------------------------------------------------------------------
// Token storage
// ---------------------------------------------------------------------------

const TOKEN_KEY = "access_token"
const REFRESH_KEY = "refresh_token"

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(TOKEN_KEY)
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(REFRESH_KEY)
}

export function setTokens(access: string, refresh: string) {
  localStorage.setItem(TOKEN_KEY, access)
  localStorage.setItem(REFRESH_KEY, refresh)
  // biome-ignore lint: sync auth cookie for SSR gating
  document.cookie = "sb_auth=1; path=/; max-age=31536000; SameSite=Lax"
}

export function clearTokens() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(REFRESH_KEY)
  // biome-ignore lint: clear auth cookie on logout
  document.cookie = "sb_auth=; path=/; max-age=0"
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message)
  }
}

// ---------------------------------------------------------------------------
// Typed fetch
// ---------------------------------------------------------------------------

const MAX_RETRIES = 3

function authHeaders(): Record<string, string> {
  const token = getAccessToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function apiFetch<T>(
  path: string,
  schema: z.ZodSchema<T>,
  init?: RequestInit & { auth?: boolean },
): Promise<T> {
  const { auth = true, headers, ...rest } = init ?? {}

  let lastError: ApiError | undefined

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (typeof navigator !== "undefined" && !navigator.onLine) {
      throw new ApiError("No internet connection", 0)
    }

    if (attempt > 0) {
      const delay = 300 * 2 ** (attempt - 1)
      await new Promise(resolve => setTimeout(resolve, delay))
    }

    let res: Response
    try {
      res = await fetch(`${API_BASE}${path}`, {
        ...rest,
        headers: { ...(auth ? authHeaders() : {}), ...headers },
      })
    } catch (error) {
      lastError = new ApiError(error instanceof Error ? error.message : "Network error", 0)
      continue
    }

    if (!res.ok) {
      if (res.status === 401 && !SKIP_AUTH) {
        clearTokens()
        redirect("/login")
      }
      const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
      lastError = new ApiError(body.detail, res.status)
      if (res.status >= 400 && res.status < 500 && res.status !== 429) {
        throw lastError
      }
      continue
    }

    if (res.status === 204) return undefined as T
    return schema.parse(await res.json())
  }

  throw lastError ?? new ApiError("Request failed", 0)
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

export async function apiPost<T>(path: string, schema: z.ZodSchema<T>, body: unknown): Promise<T> {
  return apiFetch<T>(path, schema, {
    method: "POST",
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
  })
}

export async function apiPatch<T>(path: string, schema: z.ZodSchema<T>, body: unknown): Promise<T> {
  return apiFetch<T>(path, schema, {
    method: "PATCH",
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
  })
}

export async function apiDelete(path: string): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE", headers: authHeaders() })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new ApiError(body.detail, res.status)
  }
}
