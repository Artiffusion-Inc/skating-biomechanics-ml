"use client"

import { Button } from "@/components/ui/button"
import { useTranslations } from "@/i18n"

export function CTASection() {
  const t = useTranslations("landing")

  return (
    <section className="relative mx-auto max-w-[1400px] px-6 py-32">
      <div className="relative overflow-hidden rounded-3xl bg-foreground px-8 py-20 text-center text-background">
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E\")",
          }}
        />

        <div className="relative z-10">
          <p className="mb-4 text-xs font-medium uppercase tracking-[0.25em] text-background/60">
            {t("ctaEyebrow")}
          </p>
          <h2 className="mx-auto max-w-2xl text-[clamp(1.75rem,4vw,3rem)] font-medium leading-[1.1] tracking-[-0.02em]">
            {t("ctaHeadline")}
          </h2>
          <p className="mx-auto mt-4 max-w-lg text-sm leading-relaxed text-background/70">
            {t("ctaSubtitle")}
          </p>
          <div className="mt-8 flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
            <Button
              size="lg"
              variant="secondary"
              className="h-14 rounded-full px-10 text-base font-medium"
              asChild
            >
              <a href="/register">{t("ctaAction")}</a>
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="h-14 rounded-full border-background/30 px-8 text-base font-medium text-background hover:bg-background/10"
              asChild
            >
              <a href="/login">{t("ctaLogin")}</a>
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}
