"use client"

import { useTranslations } from "@/i18n"
import type { Layout } from "@/types/choreography"

interface ScoreBarProps {
  layout: Layout | null
  discipline: "mens_singles" | "womens_singles"
  segment: "short_program" | "free_skate"
}

export function ScoreBar({ layout, discipline, segment }: ScoreBarProps) {
  const t = useTranslations("choreography")

  const tes = layout?.total_tes ?? 0
  const jumpCount = layout?.elements.filter(e => e.is_jump_pass).length ?? 0
  const spinCount =
    layout?.elements.filter(e => !e.is_jump_pass && e.code.includes("Sp")).length ?? 0

  const maxJumps = segment === "short_program" ? (discipline === "mens_singles" ? 3 : 3) : 7
  const maxSpins = segment === "short_program" ? 3 : 3

  const duration = segment === "short_program" ? "2:40" : "4:10"

  return (
    <div className="flex items-center gap-4 overflow-x-auto text-sm sm:gap-6">
      <Stat label={t("score.tes")} value={tes.toFixed(2)} highlight />
      <Stat label={t("score.total")} value={tes > 0 ? (tes * 1.4).toFixed(2) : "0.00"} highlight />
      <div className="h-4 w-px shrink-0 bg-border" />
      <Stat label={t("score.duration")} value={duration} />
      <Stat
        label={t("score.jumps")}
        value={`${jumpCount}/${maxJumps}`}
        warn={jumpCount > maxJumps}
      />
      <Stat
        label={t("score.spins")}
        value={`${spinCount}/${maxSpins}`}
        warn={spinCount > maxSpins}
      />
    </div>
  )
}

function Stat({
  label,
  value,
  highlight,
  warn,
}: {
  label: string
  value: string
  highlight?: boolean
  warn?: boolean
}) {
  return (
    <div className="shrink-0">
      <p className="text-[10px] leading-tight text-muted-foreground">{label}</p>
      <p
        className={`text-sm font-medium leading-tight ${
          highlight ? "text-primary" : warn ? "text-[oklch(var(--score-bad))]" : ""
        }`}
      >
        {value}
      </p>
    </div>
  )
}
