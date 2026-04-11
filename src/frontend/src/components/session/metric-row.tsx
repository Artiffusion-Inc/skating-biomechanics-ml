"use client"

import { MetricBadge } from "./metric-badge"

interface MetricRowProps {
  name: string
  label: string
  value: number
  unit: string
  isInRange: boolean | null
  isPr: boolean
  prevBest: number | null
  refRange: [number, number] | null
}

function rangeColor(inRange: boolean | null): string {
  if (inRange === null) return "text-muted-foreground"
  return inRange ? "text-green-500" : "text-red-500"
}

export function MetricRow({ label, value, unit, isInRange, isPr, prevBest }: MetricRowProps) {
  const delta = isPr && prevBest !== null ? value - prevBest : null
  const deltaStr = delta !== null ? `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}` : null

  return (
    <div className="flex items-center justify-between py-2 border-b border-border last:border-0">
      <div>
        <span className="text-sm">{label}</span>
        {deltaStr && <MetricBadge text={deltaStr} />}
      </div>
      <span className={`text-sm font-mono ${rangeColor(isInRange)}`}>
        {value.toFixed(2)} {unit}
      </span>
    </div>
  )
}
