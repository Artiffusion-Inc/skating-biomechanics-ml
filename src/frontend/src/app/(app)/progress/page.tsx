"use client"

import { useState } from "react"
import { useMetricRegistry, useTrend } from "@/lib/api/metrics"
import { TrendChart } from "@/components/progress/trend-chart"
import { PeriodSelector } from "@/components/progress/period-selector"

const ELEMENTS = [
  { id: "three_turn", label: "Тройка" }, { id: "waltz_jump", label: "Вальсовый" },
  { id: "toe_loop", label: "Перекидной" }, { id: "flip", label: "Флип" },
  { id: "salchow", label: "Сальхов" }, { id: "loop", label: "Петля" },
  { id: "lutz", label: "Лютц" }, { id: "axel", label: "Аксель" },
]

export default function ProgressPage() {
  const { data: registry } = useMetricRegistry()
  const [element, setElement] = useState("waltz_jump")
  const [metric, setMetric] = useState("max_height")
  const [period, setPeriod] = useState("30d")
  const { data: trend } = useTrend(undefined, element, metric, period)

  const availableMetrics = registry
    ? Object.entries(registry).filter(([, v]) => (v as any).element_types.includes(element))
    : []

  return (
    <div className="max-w-2xl mx-auto space-y-4">
      <div className="grid grid-cols-4 gap-2">
        {ELEMENTS.map((el) => (
          <button
            key={el.id}
            onClick={() => setElement(el.id)}
            className={`rounded-xl border p-2 text-center text-xs ${element === el.id ? "border-primary bg-primary/10" : "border-border"}`}
          >
            {el.label}
          </button>
        ))}
      </div>

      <div className="space-y-2">
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
          className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
        >
          {availableMetrics.map(([name, def]) => (
            <option key={name} value={name}>{(def as any).label_ru}</option>
          ))}
        </select>
        <PeriodSelector value={period} onChange={setPeriod} />
      </div>

      {trend && <TrendChart data={trend} />}
    </div>
  )
}
