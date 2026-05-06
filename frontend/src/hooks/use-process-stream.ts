"use client"

import { useEffect, useRef, useState } from "react"

interface ProcessState {
  status: string
  progress: number
  message: string
  error?: string
}

export function useProcessStream(taskId: string | null) {
  const [state, setState] = useState<ProcessState | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
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
  }, [taskId])

  return { state, isConnected }
}
