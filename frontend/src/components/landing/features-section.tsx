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
    <section id="features" className="relative mx-auto max-w-[1400px] px-6 py-32">
      <div className="mb-20 text-center">
        <p className="mb-4 text-xs font-medium uppercase tracking-[0.25em] text-muted-foreground">
          {t("featuresTitle")}
        </p>
        <h2 className="text-[clamp(1.75rem,4vw,3rem)] font-medium leading-[1.1] tracking-[-0.02em]">
          {t("featuresHeadline")}
        </h2>
      </div>

      <div className="grid gap-8 md:grid-cols-3">
        {features.map((feature, i) => (
          <div
            key={feature.title}
            className="group relative rounded-2xl border border-border bg-background p-8 transition-colors hover:border-foreground/20"
          >
            <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-full bg-muted transition-colors group-hover:bg-foreground group-hover:text-background">
              <feature.icon className="h-5 w-5" />
            </div>
            <p className="mb-2 text-xs font-medium text-muted-foreground">
              {t("step", { n: i + 1 })}
            </p>
            <h3 className="mb-3 text-lg font-medium">{feature.title}</h3>
            <p className="text-sm leading-relaxed text-muted-foreground">{feature.description}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
