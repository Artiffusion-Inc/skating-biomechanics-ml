"use client"

import { ResponsiveContainer, LineChart, Line, ReferenceArea, XAxis, YAxis } from "recharts"
import type { TrendResponse } from "@/types"

const TREND_LABELS: Record<string, string> = { improving: "Улучшение", stable: "Стабильно", declining: "Ухудшение" }

export function TrendChart({ data }: { data: TrendResponse }) {
  if (!data.data_points.length) {
    return <p className="text-center text-muted-foreground py-10">Нет данных</p>
  }

  const refMin = data.reference_range?.min
  const refMax = data.reference_range?.max

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span>{data.metric_name}</span>
        <span className={data.trend === "improving" ? "text-green-500" : data.trend === "declining" ? "text-red-500" : "text-muted-foreground"}>
          {TREND_LABELS[data.trend]}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data.data_points} margin={{ top: 10, right: 10, bottom: 0, left: -10 }}>
          {refMin !== undefined && refMax !== undefined && (
            <ReferenceArea y1={refMin} y2={refMax} fill="#22c55e" fillOpacity={0.1} ifOverflow="extendDomain" />
          )}
          <XAxis dataKey="date" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="value" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
      {data.current_pr !== null && (
        <p className="text-sm text-amber-500 font-medium">PR: {data.current_pr.toFixed(3)}</p>
      )}
    </div>
  )
}
