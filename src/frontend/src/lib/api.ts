import { DetectResponseSchema, ProcessRequestSchema } from "@/lib/schemas"

const API_BASE = "/api"

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
  if (!res.ok) {
    const text = await res.text()
    return { data: null, error: text }
  }
  const json = await res.json()
  const data = DetectResponseSchema.parse(json) // Zod validation
  return { data, error: undefined }
}

export interface SSECallbacks {
  onProgress?: (progress: number, message: string) => void
  onResult?: (result: unknown) => void
  onError?: (error: string) => void
}

export async function processVideo(
  request: {
    video_path: string
    person_click: { x: number; y: number }
    frame_skip: number
    layer: number
    tracking: string
    export: boolean
    depth?: boolean
    optical_flow?: boolean
    segment?: boolean
    foot_track?: boolean
    matting?: boolean
  },
  callbacks: SSECallbacks,
): Promise<void> {
  // Validate request with Zod
  const validated = ProcessRequestSchema.parse(request)

  const res = await fetch(`${API_BASE}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(validated),
  })

  if (!res.ok) {
    const text = await res.text()
    callbacks.onError?.(text)
    return
  }

  const reader = res.body?.getReader()
  if (!reader) {
    callbacks.onError?.("No response stream")
    return
  }

  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split("\n")
    buffer = lines.pop() || ""

    let currentEvent = ""
    for (const line of lines) {
      if (line.startsWith("event:")) {
        currentEvent = line.slice(6).trim()
      } else if (line.startsWith("data:")) {
        const data = line.slice(5).trim()
        if (!data) continue

        try {
          const parsed = JSON.parse(data)
          if (currentEvent === "progress") {
            callbacks.onProgress?.(parsed.progress, parsed.message)
          } else if (currentEvent === "result") {
            callbacks.onResult?.(parsed)
          } else if (currentEvent === "error") {
            callbacks.onError?.(parsed.error)
          }
        } catch {
          // skip malformed JSON
        }
      }
    }
  }
}
