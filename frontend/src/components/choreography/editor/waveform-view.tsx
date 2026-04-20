"use client"

import { useRef } from "react"
import WaveSurfer from "wavesurfer.js"
import { useTranslations } from "@/i18n"
import { useMountEffect } from "@/lib/useMountEffect"
import { useChoreographyEditor } from "./store"

interface WaveformViewProps {
  audioUrl: string | null
}

interface WaveformRefObj {
  current: WaveSurfer | null
  _cleanup?: () => void
}
export const WaveformViewRef: WaveformRefObj = { current: null }

export function WaveformView({ audioUrl }: WaveformViewProps) {
  const mountedRef = useRef(false)
  const t = useTranslations("choreography.music")
  const setCurrentTime = useChoreographyEditor(s => s.setCurrentTime)
  const setIsPlaying = useChoreographyEditor(s => s.setIsPlaying)

  useMountEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      WaveformViewRef._cleanup?.()
      WaveformViewRef.current = null
    }
  })

  function handleRef(el: HTMLDivElement | null) {
    if (!el || !mountedRef.current) return
    WaveformViewRef._cleanup?.()
    WaveformViewRef.current = null
    if (!audioUrl) return

    const ws = WaveSurfer.create({
      container: el,
      waveColor: "oklch(0.5 0.05 260 / 0.25)",
      progressColor: "oklch(var(--primary) / 0.6)",
      cursorColor: "oklch(0.65 0.25 25)",
      cursorWidth: 2,
      height: 64,
      barWidth: 2,
      barGap: 1,
      barRadius: 1,
      normalize: true,
      hideScrollbar: true,
      minPxPerSec: useChoreographyEditor.getState().pixelsPerSecond,
    })

    const unsubs = [
      ws.on("timeupdate", time => {
        if (mountedRef.current) setCurrentTime(time)
      }),
      ws.on("play", () => {
        if (mountedRef.current) setIsPlaying(true)
      }),
      ws.on("pause", () => {
        if (mountedRef.current) setIsPlaying(false)
      }),
    ]

    ws.load(audioUrl).catch(() => {})
    WaveformViewRef.current = ws
    WaveformViewRef._cleanup = () => {
      for (const u of unsubs) u?.()
      ws.destroy()
      if (WaveformViewRef.current === ws) WaveformViewRef.current = null
    }
  }

  return (
    <div
      key={audioUrl ?? "empty"}
      ref={handleRef}
      className="relative w-full"
      style={{ height: 64 }}
    >
      {!audioUrl && (
        <div className="flex h-full items-center justify-center text-xs text-muted-foreground/50">
          {t("uploadPrompt")}
        </div>
      )}
    </div>
  )
}
