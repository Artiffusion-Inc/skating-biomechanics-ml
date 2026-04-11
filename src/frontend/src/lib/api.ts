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
