"use client"

import { useTranslations } from "@/i18n"

export function MetricsSection() {
  const t = useTranslations("landing")

  const metrics = [
    { value: "17", label: t("metricSkeletonPoints"), description: t("metricSkeletonFormat") },
    { value: "12+", label: t("metricMetrics"), description: t("metricMetricsDesc") },
    { value: "3D", label: t("metric3D"), description: t("metric3DDesc") },
    { value: "<15", label: t("metricSeconds"), description: t("metricSecondsDesc") },
  ]

  return (
    <section className="relative overflow-hidden border-y border-border">
      {/* Ice gradient band */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(90deg, var(--ice-surface) 0%, transparent 30%, transparent 70%, var(--ice-surface) 100%)",
          opacity: 0.5,
        }}
      />

      <div className="relative mx-auto max-w-[1400px] px-6 py-24">
        <div className="grid gap-16 sm:grid-cols-2 lg:grid-cols-4">
          {metrics.map(m => (
            <div key={m.label} className="group text-center">
              <p
                className="metric-giant transition-colors group-hover:text-[var(--ice-deep)]"
                style={{ color: "var(--foreground)" }}
              >
                {m.value}
              </p>
              <p className="mt-3 text-sm font-medium tracking-wide uppercase">
                {m.label}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">{m.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
