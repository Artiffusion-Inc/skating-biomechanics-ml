/**
 * Non-auth API wrappers: models, process queue, detect, SSE streaming.
 */

import { z } from "zod"
import { API_BASE, apiFetch } from "@/lib/api-client"
import type { ProcessResponse } from "@/lib/schemas"
import { DetectResponseSchema, ProcessRequestSchema, ProcessResponseSchema } from "@/lib/schemas"

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export interface ModelStatus {
  id: string
  available: boolean
  size_mb: number | null
}

const ModelStatusListSchema = z.array(
  z.object({ id: z.string(), available: z.boolean(), size_mb: z.number().nullable() }),
)

export async function getModels(): Promise<ModelStatus[]> {
  return apiFetch("/models", ModelStatusListSchema, { auth: false })
}

// ---------------------------------------------------------------------------
// Process queue
// ---------------------------------------------------------------------------

const QueueResponseSchema = z.object({ task_id: z.string() })

const TaskStatusSchema = z.object({
  task_id: z.string(),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]),
  progress: z.number(),
  message: z.string(),
  result: ProcessResponseSchema.nullable(),
  error: z.string().nullable(),
})

export interface TaskStatusResponse {
  task_id: string
  status: "pending" | "running" | "completed" | "failed" | "cancelled"
  progress: number
  message: string
  result: ProcessResponse | null
  error: string | null
}

export async function enqueueProcess(
  request: Parameters<typeof ProcessRequestSchema.parse>[0],
): Promise<{ task_id: string }> {
  const validated = ProcessRequestSchema.parse(request)
  return apiFetch("/process/queue", QueueResponseSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(validated),
  })
}

export async function pollTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  return apiFetch(`/process/${taskId}/status`, TaskStatusSchema)
}

export async function cancelProcessing(): Promise<void> {
  await apiFetch("/process/cancel", z.object({ status: z.literal("cancelled") }), {
    method: "POST",
  })
}

export async function cancelQueuedProcess(taskId: string): Promise<void> {
  await apiFetch(
    `/process/${taskId}/cancel`,
    z.object({ status: z.string(), task_id: z.string() }),
    {
      method: "POST",
    },
  )
}

// ---------------------------------------------------------------------------
// Detect (FormData — can't use JSON apiFetch)
// ---------------------------------------------------------------------------

export async function detectPersons(
  file: File,
  tracking = "auto",
): Promise<{ data: unknown; error?: string }> {
  const form = new FormData()
  form.append("video", file)
  const res = await fetch(`${API_BASE}/detect?tracking=${encodeURIComponent(tracking)}`, {
    method: "POST",
    body: form,
  })
  if (!res.ok) return { data: null, error: await res.text() }
  return { data: DetectResponseSchema.parse(await res.json()), error: undefined }
}

// ---------------------------------------------------------------------------
// Process (SSE streaming — custom reader)
// ---------------------------------------------------------------------------

export interface SSECallbacks {
  onProgress?: (progress: number, message: string) => void
  onResult?: (result: unknown) => void
  onError?: (error: string) => void
}

export async function processVideo(
  request: Parameters<typeof ProcessRequestSchema.parse>[0],
  callbacks: SSECallbacks,
): Promise<void> {
  const validated = ProcessRequestSchema.parse(request)
  const res = await fetch(`${API_BASE}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(validated),
  })

  if (!res.ok) {
    callbacks.onError?.(await res.text())
    return
  }

  const reader = res.body?.getReader()
  if (!reader) {
    callbacks.onError?.("No response stream")
    return
  }

  const decoder = new TextDecoder()
  let buffer = ""

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split("\n")
    buffer = lines.pop() || ""

    let event = ""
    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim()
      } else if (line.startsWith("data:")) {
        const raw = line.slice(5).trim()
        if (!raw) continue
        try {
          const parsed = JSON.parse(raw)
          if (event === "progress") callbacks.onProgress?.(parsed.progress, parsed.message)
          else if (event === "result") callbacks.onResult?.(parsed)
          else if (event === "error") callbacks.onError?.(parsed.error)
        } catch {
          // skip malformed JSON
        }
      }
    }
  }
}
