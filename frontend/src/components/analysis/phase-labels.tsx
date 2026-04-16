"use client"

import type { PhasesData } from "@/types"

interface PhaseLabelsProps {
  phases: PhasesData
  currentFrame: number
  width: number
}

export function PhaseLabels({ phases, currentFrame, width }: PhaseLabelsProps) {
  if (!phases.takeoff && !phases.peak && !phases.landing) return null

  // Calculate label positions (normalized 0-1)
  const takeoffX = phases.takeoff !== undefined ? (phases.takeoff / currentFrame) * width : null
  const peakX = phases.peak !== undefined ? (phases.peak / currentFrame) * width : null
  const landingX = phases.landing !== undefined ? (phases.landing / currentFrame) * width : null

  return (
    <div className="absolute top-2 left-0 right-0 flex justify-between px-4">
      {takeoffX !== null && (
        <div
          className="absolute top-0 rounded-full bg-green-500/80 px-2 py-1 text-xs font-medium text-white"
          style={{ left: `${takeoffX}px` }}
        >
          Takeoff
        </div>
      )}
      {peakX !== null && (
        <div
          className="absolute top-0 rounded-full bg-yellow-500/80 px-2 py-1 text-xs font-medium text-white"
          style={{ left: `${peakX}px` }}
        >
          Peak
        </div>
      )}
      {landingX !== null && (
        <div
          className="absolute top-0 rounded-full bg-red-500/80 px-2 py-1 text-xs font-medium text-white"
          style={{ left: `${landingX}px` }}
        >
          Landing
        </div>
      )}
    </div>
  )
}
