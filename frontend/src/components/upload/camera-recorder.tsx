"use client"

import { useCallback, useRef, useState } from "react"
import { useMountEffect } from "@/lib/useMountEffect"

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

  useMountEffect(() => {
    initCamera()
    return () => {
      // Cleanup: stop camera stream
      if (streamRef.current) {
        for (const track of streamRef.current.getTracks()) {
          track.stop()
        }
      }
      // Cleanup: stop timer
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  })

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
        {cameraReady &&
          (recording ? (
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
          ))}
      </div>
    </div>
  )
}
