"use client"

import { useTranslations } from "@/i18n"

export function MetricsSection() {
  const t = useTranslations("landing")

  const metrics = [
    { value: "17", label: t("metricSkeletonPoints"), description: t("metricSkeletonFormat") },
    { value: "12+", label: t("metricMetrics"), description: t("metricMetricsDesc") },
    { value: "3D", label: t("metric3D"), description: t("metric3DDesc") },
    { value: "< 15", label: t("metricSeconds"), description: t("metricSecondsDesc") },
  ]

  return (
    <section className="relative border-y border-border bg-muted/30">
      <div className="mx-auto max-w-[1400px] px-6 py-20">
        <div className="grid gap-12 sm:grid-cols-2 lg:grid-cols-4">
          {metrics.map(m => (
            <div key={m.label} className="text-center">
              <p className="text-[clamp(2.5rem,5vw,4rem)] font-medium leading-none tracking-[-0.03em]">
                {m.value}
              </p>
              <p className="mt-2 text-sm font-medium">{m.label}</p>
              <p className="mt-1 text-xs text-muted-foreground">{m.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
