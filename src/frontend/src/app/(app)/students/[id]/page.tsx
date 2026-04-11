"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { useDiagnostics, useTrend } from "@/lib/api/metrics"
import { DiagnosticsList } from "@/components/coach/diagnostics-list"
import { TrendChart } from "@/components/progress/trend-chart"
import { PeriodSelector } from "@/components/progress/period-selector"

const ELEMENTS = [
  { id: "three_turn", label: "Тройка" }, { id: "waltz_jump", label: "Вальсовый" },
  { id: "toe_loop", label: "Перекидной" }, { id: "flip", label: "Флип" },
  { id: "salchow", label: "Сальхов" }, { id: "loop", label: "Петля" },
  { id: "lutz", label: "Лютц" }, { id: "axel", label: "Аксель" },
]

export default function StudentProfilePage() {
  const { id } = useParams<{ id: string }>()
  const [tab, setTab] = useState<"progress" | "diagnostics">("progress")
  const [element, setElement] = useState("waltz_jump")
  const [metric, setMetric] = useState("max_height")
  const [period, setPeriod] = useState("30d")

  const { data: trend } = useTrend(id, element, metric, period)
  const { data: diag } = useDiagnostics(id)

  return (
    <div className="mx-auto max-w-2xl space-y-4 sm:max-w-3xl">
      <div className="flex gap-2">
        <Link href="/dashboard" className="text-sm text-muted-foreground hover:underline">&larr; Назад</Link>
      </div>

      <div className="flex gap-1 rounded-lg bg-muted p-1">
        <button onClick={() => setTab("progress")} className={`flex-1 rounded-md px-3 py-2 text-sm font-medium ${tab === "progress" ? "bg-background shadow-sm" : ""}`}>
          Прогресс
        </button>
        <button onClick={() => setTab("diagnostics")} className={`flex-1 rounded-md px-3 py-2 text-sm font-medium ${tab === "diagnostics" ? "bg-background shadow-sm" : ""}`}>
          Диагностика
        </button>
      </div>

      {tab === "progress" && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-1.5 sm:gap-2">
            {ELEMENTS.map((el) => (
              <button key={el.id} onClick={() => setElement(el.id)} className={`truncate rounded-xl border p-1.5 text-center text-[11px] sm:p-2 sm:text-xs ${element === el.id ? "border-primary bg-primary/10" : "border-border"}`}>
                {el.label}
              </button>
            ))}
          </div>
          <select value={metric} onChange={(e) => setMetric(e.target.value)} className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm">
            <option value="max_height">Высота прыжка</option>
            <option value="airtime">Время полёта</option>
            <option value="landing_knee_stability">Стабильность приземления</option>
            <option value="rotation_speed">Скорость вращения</option>
          </select>
          <PeriodSelector value={period} onChange={setPeriod} />
          {trend && <TrendChart data={trend} />}
        </div>
      )}

      {tab === "diagnostics" && diag && <DiagnosticsList findings={diag.findings} />}
    </div>
  )
}
