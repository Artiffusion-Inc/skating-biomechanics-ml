"use client"

import { useRef, useState } from "react"
import { useMountEffect } from "@/lib/useMountEffect"

interface ProcessState {
  status: string
  progress: number
  message: string
  error?: string
}

const IDLE = { state: null, isConnected: false }

export function useProcessStream(taskId: string | null) {
  const [state, setState] = useState<ProcessState | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const esRef = useRef<EventSource | null>(null)

  useMountEffect(() => {
    if (!taskId) return

    const es = new EventSource(`/api/v1/process/${taskId}/stream`)
    esRef.current = es

    es.onopen = () => setIsConnected(true)
    es.onmessage = e => {
      const data = JSON.parse(e.data)
      setState(data)
      if (["completed", "failed", "cancelled"].includes(data.status)) {
        es.close()
        setIsConnected(false)
      }
    }
    es.onerror = () => {
      setIsConnected(false)
      es.close()
    }

    return () => {
      es.close()
      setIsConnected(false)
    }
  })

  if (!taskId) return IDLE
  return { state, isConnected }
}
