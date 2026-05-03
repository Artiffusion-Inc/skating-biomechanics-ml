"use client"

import { Video, BarChart3, Users } from "lucide-react"
import { useTranslations } from "@/i18n"

const icons = [Video, BarChart3, Users]

export function FeaturesSection() {
  const t = useTranslations("landing")

  const features = [
    {
      icon: icons[0],
      title: t("featureUploadTitle"),
      description: t("featureUploadDesc"),
    },
    {
      icon: icons[1],
      title: t("featureMetricsTitle"),
      description: t("featureMetricsDesc"),
    },
    {
      icon: icons[2],
      title: t("featureCompareTitle"),
      description: t("featureCompareDesc"),
    },
  ]

  return (
    <section id="features" className="relative mx-auto max-w-[1400px] px-4 py-20 sm:px-6 md:py-32">
      {/* Diagonal streak */}
      <div className="diagonal-streak" />

      <div className="mb-12 md:mb-20">
        <p className="mb-4 text-xs font-medium uppercase tracking-[0.3em] text-muted-foreground">
          {t("featuresTitle")}
        </p>
        <h2 className="max-w-2xl text-[clamp(1.75rem,4vw,3rem)] font-medium leading-[1.1] tracking-[-0.02em]">
          {t("featuresHeadline")}
        </h2>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {features.map((feature, i) => (
          <div
            key={feature.title}
            className="group relative overflow-hidden rounded-2xl border border-border/60 bg-background p-6 md:p-8 transition-colors hover:border-foreground/20"
          >
            {/* Step watermark */}
            <span className="step-watermark">
              {String(i + 1).padStart(2, "0")}
            </span>

            <div className="relative z-10">
              <div
                className="mb-6 flex h-12 w-12 items-center justify-center rounded-full transition-colors group-hover:bg-foreground group-hover:text-background"
                style={{ background: "oklch(0.92 0.02 240 / 0.6)" }}
              >
                <feature.icon className="h-5 w-5" style={{ color: "var(--ice-deep)" }} />
              </div>
              <h3 className="mb-3 text-lg font-medium">{feature.title}</h3>
              <p className="text-sm leading-relaxed text-muted-foreground">
                {feature.description}
              </p>
            </div>

            {/* Hover ice accent */}
            <div
              className="absolute right-0 bottom-0 h-32 w-32 rounded-full opacity-0 blur-3xl transition-opacity group-hover:opacity-30"
              style={{ background: "var(--ice-glow)" }}
            />
          </div>
        ))}
      </div>
    </section>
  )
}
