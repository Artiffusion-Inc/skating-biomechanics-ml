"use client"

import { useState } from "react"
import { useMountEffect } from "@/lib/useMountEffect"
import { useTranslations } from "@/i18n"

function SkeletonPose() {
  const [frame, setFrame] = useState(0)

  useMountEffect(() => {
    const id = setInterval(() => setFrame(f => (f + 1) % 60), 50)
    return () => clearInterval(id)
  })

  const basePoints = [
    { x: 0.5, y: 0.15 },
    { x: 0.5, y: 0.3 },
    { x: 0.38, y: 0.32 },
    { x: 0.3, y: 0.48 },
    { x: 0.22, y: 0.62 },
    { x: 0.62, y: 0.32 },
    { x: 0.7, y: 0.48 },
    { x: 0.78, y: 0.62 },
    { x: 0.5, y: 0.52 },
    { x: 0.42, y: 0.68 },
    { x: 0.36, y: 0.85 },
    { x: 0.32, y: 0.98 },
    { x: 0.58, y: 0.68 },
    { x: 0.64, y: 0.85 },
    { x: 0.68, y: 0.98 },
  ]

  const points = basePoints.map((p, i) => {
    const offset = Math.sin((frame + i * 10) * 0.1) * 0.015
    return { x: p.x + offset, y: p.y + offset * 0.5 }
  })

  const lines = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [8, 12],
    [12, 13],
    [13, 14],
  ] as const

  return (
    <svg viewBox="0 0 1 1" className="absolute inset-0 h-full w-full">
      <title>Skeleton overlay</title>
      {lines.map(([a, b]) => (
        <line
          key={`${a}-${b}`}
          x1={points[a].x}
          y1={points[a].y}
          x2={points[b].x}
          y2={points[b].y}
          stroke="rgba(255,255,255,0.9)"
          strokeWidth="0.008"
          strokeLinecap="round"
        />
      ))}
      {points.map(p => (
        <circle
          key={`pt-${p.x}-${p.y}`}
          cx={p.x}
          cy={p.y}
          r="0.012"
          fill="rgba(255,255,255,0.95)"
        />
      ))}
    </svg>
  )
}

export function DemoSection() {
  const t = useTranslations("landing")

  return (
    <section className="relative mx-auto max-w-[1400px] px-6 py-32">
      <div className="mb-20 text-center">
        <p className="mb-4 text-xs font-medium uppercase tracking-[0.25em] text-muted-foreground">
          {t("demoEyebrow")}
        </p>
        <h2 className="text-[clamp(1.75rem,4vw,3rem)] font-medium leading-[1.1] tracking-[-0.02em]">
          {t("demoHeadline")}
        </h2>
      </div>

      <div
        className="relative mx-auto aspect-video max-w-4xl overflow-hidden rounded-2xl border border-border shadow-2xl"
        style={{ backgroundColor: "#000" }}
      >
        <div
          className="absolute inset-0"
          style={{
            background: "linear-gradient(to bottom right, oklch(0.24 0.03 240), oklch(0.22 0.02 240), oklch(0.24 0.03 240))",
          }}
        />

        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse at 50% 100%, rgba(200,220,255,0.3) 0%, transparent 70%)",
            opacity: 0.2,
          }}
        />

        <SkeletonPose />

        <div
          className="absolute top-4 left-4 rounded-lg px-3 py-1.5 text-xs font-mono backdrop-blur-sm"
          style={{ backgroundColor: "rgba(0,0,0,0.6)", color: "rgba(255,255,255,0.9)" }}
        >
          RTMO • 17kp • 30fps
        </div>
        <div
          className="absolute right-4 bottom-4 rounded-lg px-3 py-1.5 text-xs font-mono backdrop-blur-sm"
          style={{ backgroundColor: "rgba(0,0,0,0.6)", color: "rgba(255,255,255,0.9)" }}
        >
          H3.6M Format
        </div>

        <div
          className="absolute top-3 left-3 h-6 w-6"
          style={{ borderTop: "2px solid rgba(255,255,255,0.4)", borderLeft: "2px solid rgba(255,255,255,0.4)" }}
        />
        <div
          className="absolute top-3 right-3 h-6 w-6"
          style={{ borderTop: "2px solid rgba(255,255,255,0.4)", borderRight: "2px solid rgba(255,255,255,0.4)" }}
        />
        <div
          className="absolute right-3 bottom-3 h-6 w-6"
          style={{ borderRight: "2px solid rgba(255,255,255,0.4)", borderBottom: "2px solid rgba(255,255,255,0.4)" }}
        />
        <div
          className="absolute bottom-3 left-3 h-6 w-6"
          style={{ borderBottom: "2px solid rgba(255,255,255,0.4)", borderLeft: "2px solid rgba(255,255,255,0.4)" }}
        />
      </div>

      <p className="mx-auto mt-8 max-w-xl text-center text-sm text-muted-foreground">
        {t("demoCaption")}
      </p>
    </section>
  )
}
