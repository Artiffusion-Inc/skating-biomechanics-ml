"use client"

import Link from "next/link"
import { Award, Clock, Loader2 } from "lucide-react"
import type { Session } from "@/types"

const ELEMENT_NAMES: Record<string, string> = {
  three_turn: "Тройка", waltz_jump: "Вальсовый", toe_loop: "Перекидной",
  flip: "Флип", salchow: "Сальхов", loop: "Петля",
  lutz: "Лютц", axel: "Аксель",
}

function relativeTime(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return "только что"
  if (mins < 60) return `${mins} мин назад`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours} ч назад`
  const days = Math.floor(hours / 24)
  return `${days} дн назад`
}

function scoreColor(score: number | null): string {
  if (score === null) return "text-muted-foreground"
  if (score >= 0.8) return "text-green-500"
  if (score >= 0.5) return "text-amber-500"
  return "text-red-500"
}

export function SessionCard({ session }: { session: Session }) {
  const hasPR = session.metrics.some((m) => m.is_pr)

  return (
    <Link href={`/sessions/${session.id}`} className="block">
      <div className="rounded-2xl border border-border p-4 hover:bg-accent/30 transition-colors">
        <div className="flex items-start justify-between">
          <div>
            <p className="font-medium">{ELEMENT_NAMES[session.element_type] ?? session.element_type}</p>
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {relativeTime(session.created_at)}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {hasPR && <Award className="h-4 w-4 text-amber-500" />}
            {session.overall_score !== null && (
              <span className={`text-sm font-medium ${scoreColor(session.overall_score)}`}>
                {Math.round(session.overall_score * 100)}%
              </span>
            )}
          </div>
        </div>

        {session.status !== "done" ? (
          <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            {session.status === "processing" ? "Анализ..." : "Загрузка..."}
          </div>
        ) : (
          <div className="mt-2 flex gap-3 text-xs text-muted-foreground">
            {session.metrics.slice(0, 2).map((m) => (
              <span key={m.metric_name}>{m.metric_name}: {m.metric_value.toFixed(2)}</span>
            ))}
          </div>
        )}
      </div>
    </Link>
  )
}
