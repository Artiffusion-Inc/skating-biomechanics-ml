"use client"

import {
  ArrowLeft,
  Map as MapIcon,
  Pause,
  Play,
  Save,
  SkipBack,
  SkipForward,
  ZoomIn,
} from "lucide-react"
import Link from "next/link"
import { useParams } from "next/navigation"
import { useRef, useState } from "react"
import { startAutoSave } from "@/components/choreography/editor/auto-save"
import { useChoreographyEditor } from "@/components/choreography/editor/store"
import { TrackRow } from "@/components/choreography/editor/track-row"
import { WaveformView, WaveformViewRef } from "@/components/choreography/editor/waveform-view"
import { RinkDiagram } from "@/components/choreography/rink-diagram"
import { useTranslations } from "@/i18n"
import { useMusicAnalysis, useProgram, useSaveProgram } from "@/lib/api/choreography"
import { useMountEffect } from "@/lib/useMountEffect"

export default function ProgramEditorPage() {
  const { id } = useParams<{ id: string }>()
  const tc = useTranslations("common")
  const t = useTranslations("choreography")
  const { data: program, isLoading } = useProgram(id)
  const saveProgram = useSaveProgram()
  const editor = useChoreographyEditor()
  const unsubRef = useRef<(() => void) | null>(null)
  const [showRink, setShowRink] = useState(true)

  const musicAnalysisId = program?.music_analysis_id
  const { data: musicAnalysis } = useMusicAnalysis(musicAnalysisId ?? undefined)

  const audioUrl = musicAnalysis?.audio_url ?? null
  const musicDuration = musicAnalysis?.duration_sec ?? 180
  const beatMarkers = musicAnalysis?.peaks ?? []
  const phraseMarkers = (musicAnalysis?.structure ?? []).map(s => s.start)

  // Initialize store once when program data arrives (inline, no useEffect needed)
  const initRef = useRef<string | null>(null)
  if (program && initRef.current !== program.id) {
    initRef.current = program.id
    editor.initFromProgram(
      program,
      audioUrl,
      musicDuration,
      beatMarkers,
      phraseMarkers,
      musicAnalysis?.bpm ?? 0,
    )
  }

  // Start auto-save once store is initialized
  const initialized = program && editor.programId === program.id
  const autoSaveRef = useRef(false)
  if (initialized && !autoSaveRef.current) {
    autoSaveRef.current = true
    unsubRef.current = startAutoSave(
      data => saveProgram.mutate(data),
      () => saveProgram.isPending,
    )
  }

  // Cleanup auto-save on unmount
  useMountEffect(() => {
    return () => {
      if (unsubRef.current) unsubRef.current()
    }
  })

  const layout = initialized
    ? {
        elements: editor.elements.map(el => ({
          code: el.code,
          goe: el.goe,
          timestamp: el.timestamp,
          position: el.position ?? null,
          is_back_half: false,
          is_jump_pass: el.trackType === "jumps",
          jump_pass_index: el.jumpPassIndex ?? null,
        })),
        total_tes: editor.getLayoutForSave().total_tes,
        back_half_indices: editor.getLayoutForSave().back_half_indices,
      }
    : null

  function handleSave() {
    if (!program) return
    if (unsubRef.current) unsubRef.current()
    const { layout: saveLayout } = editor.getLayoutForSave()
    saveProgram.mutate(
      { id, title: editor.title, layout: { elements: saveLayout } },
      {
        onSuccess: () => {
          unsubRef.current = startAutoSave(
            data => saveProgram.mutate(data),
            () => saveProgram.isPending,
          )
        },
      },
    )
  }

  const tes = layout?.total_tes ?? 0

  if (isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center text-muted-foreground">
        {tc("loading")}
      </div>
    )
  }

  if (!program) {
    return (
      <div className="flex flex-1 items-center justify-center text-muted-foreground">
        {t("notFound")}
      </div>
    )
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-background">
      {/* ── Top bar ── */}
      <header className="flex h-11 shrink-0 items-center gap-2 border-b border-border px-3">
        <Link href="/choreography" className="text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" />
        </Link>
        <input
          type="text"
          value={editor.title}
          onChange={e => editor.setTitle(e.target.value)}
          placeholder={t("untitled")}
          className="min-w-0 flex-1 bg-transparent text-sm font-semibold outline-none placeholder:text-muted-foreground"
        />
        {tes > 0 && (
          <span className="shrink-0 rounded-md bg-primary/10 px-2 py-0.5 text-xs font-semibold text-primary">
            TES {tes.toFixed(1)}
          </span>
        )}
        <button
          type="button"
          onClick={handleSave}
          disabled={saveProgram.isPending}
          className="flex shrink-0 items-center gap-1 rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          <Save className="h-3.5 w-3.5" />
          {t("save")}
        </button>
      </header>

      {/* ── Main area ── */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* ── Timeline column ── */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          {/* Waveform + transport */}
          <div className="shrink-0 border-b border-border">
            <WaveformView audioUrl={audioUrl} />
            <TransportInline />
          </div>

          {/* Track area */}
          <div className="flex min-h-0 flex-1 flex-col divide-y divide-border">
            <TrackRow type="jumps" />
            <TrackRow type="spins" />
            <TrackRow type="sequences" />
          </div>
        </div>

        {/* ── Rink panel ── */}
        <aside
          className="flex shrink-0 flex-col border-t border-border bg-muted/30 lg:w-80 lg:border-l lg:border-t-0"
          style={{ maxHeight: showRink ? "50%" : undefined }}
        >
          {/* Toggle tab — visible on mobile only */}
          <button
            type="button"
            onClick={() => setShowRink(v => !v)}
            className="flex h-8 w-full items-center justify-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground lg:hidden"
          >
            <MapIcon className="h-3.5 w-3.5" />
            {showRink ? t("rink.hide") : t("rink.show")}
          </button>
          <div className={`min-h-0 flex-1 overflow-auto p-3 ${showRink ? "" : "hidden lg:block"}`}>
            <RinkDiagram />
          </div>
        </aside>
      </div>
    </div>
  )
}

/* ── Inline transport below waveform ── */
function TransportInline() {
  const {
    isPlaying,
    currentTime,
    musicDuration,
    pixelsPerSecond,
    snapMode,
    setCurrentTime,
    setPixelsPerSecond,
    setSnapMode,
  } = useChoreographyEditor()
  const t = useTranslations("choreography")
  const ws = WaveformViewRef.current

  function fmt(s: number) {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, "0")}`
  }

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 text-xs">
      <button
        type="button"
        onClick={() => ws?.playPause()}
        className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-primary-foreground hover:bg-primary/90"
        aria-label={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? <Pause className="h-3 w-3" /> : <Play className="ml-0.5 h-3 w-3" />}
      </button>

      <button
        type="button"
        onClick={() => ws?.setTime(Math.max(0, ws?.getCurrentTime() - 5))}
        className="rounded p-1 text-muted-foreground hover:text-foreground"
        aria-label="-5s"
      >
        <SkipBack className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={() => ws?.setTime(Math.min(ws?.getDuration() ?? 0, ws?.getCurrentTime() + 5))}
        className="rounded p-1 text-muted-foreground hover:text-foreground"
        aria-label="+5s"
      >
        <SkipForward className="h-3.5 w-3.5" />
      </button>

      <div className="flex flex-1 items-center gap-1.5 tabular-nums">
        <span className="w-10 text-right text-foreground">{fmt(currentTime)}</span>
        <input
          type="range"
          min={0}
          max={musicDuration || 180}
          step={0.1}
          value={currentTime}
          onChange={e => {
            const v = Number.parseFloat(e.target.value)
            setCurrentTime(v)
            ws?.setTime(v)
          }}
          className="h-1 flex-1 cursor-pointer accent-primary"
          aria-label="Seek"
        />
        <span className="w-10 text-muted-foreground">{fmt(musicDuration)}</span>
      </div>

      <div className="hidden items-center gap-1 text-muted-foreground sm:flex">
        <ZoomIn className="h-3 w-3" />
        <input
          type="range"
          min={2}
          max={60}
          step={1}
          value={pixelsPerSecond}
          onChange={e => setPixelsPerSecond(Number(e.target.value))}
          className="h-1 w-16 cursor-pointer accent-primary"
          aria-label="Zoom"
        />
      </div>

      <select
        value={snapMode}
        onChange={e => setSnapMode(e.target.value as "beats" | "phrases" | "off")}
        className="h-6 rounded border border-border bg-background px-1 text-[11px]"
        aria-label="Snap"
      >
        <option value="off">{t("snapOff")}</option>
        <option value="beats">{t("snapBeats")}</option>
        <option value="phrases">{t("snapPhrases")}</option>
      </select>
    </div>
  )
}
