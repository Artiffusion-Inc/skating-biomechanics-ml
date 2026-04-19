"use client"

import { useRef } from "react"
import WaveSurfer from "wavesurfer.js"
import { useChoreographyEditor } from "./store"

interface WaveformViewProps {
  audioUrl: string | null
}

export function WaveformView({ audioUrl }: WaveformViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const { pixelsPerSecond, setCurrentTime, setIsPlaying } =
    useChoreographyEditor()

  function handleContainerRef(el: HTMLDivElement | null) {
    if (!el || wavesurferRef.current) return
    if (!audioUrl) return

    const ws = WaveSurfer.create({
      container: el,
      waveColor: "oklch(var(--muted-foreground) / 0.3)",
      progressColor: "oklch(var(--primary))",
      cursorColor: "oklch(0.6 0.2 25)",
      cursorWidth: 2,
      height: 80,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      hideScrollbar: true,
      minPxPerSec: pixelsPerSecond,
    })

    ws.on("timeupdate", (time) => setCurrentTime(time))
    ws.on("play", () => setIsPlaying(true))
    ws.on("pause", () => setIsPlaying(false))

    ws.load(audioUrl).catch(() => {
      // Graceful fallback if audio fails to load
    })

    wavesurferRef.current = ws
  }

  // Expose play/pause/seek for TransportBar via module-level ref
  WaveformViewRef.current = wavesurferRef.current

  return (
    <div
      ref={handleContainerRef}
      className="relative w-full overflow-hidden rounded-lg border border-border bg-muted/30"
      style={{ height: 80 }}
    >
      {!audioUrl && (
        <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
          Загрузите музыку для отображения waveform
        </div>
      )}
    </div>
  )
}

// Module-level ref for TransportBar to access wavesurfer instance
export const WaveformViewRef = { current: null as WaveSurfer | null }
