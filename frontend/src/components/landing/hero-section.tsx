"use client"

import { useState } from "react"
import { useMountEffect } from "@/lib/useMountEffect"
import { Button } from "@/components/ui/button"
import { ChevronDown } from "lucide-react"
import { useTranslations } from "@/i18n"

export function HeroSection() {
  const [mounted, setMounted] = useState(false)
  const t = useTranslations("landing")

  useMountEffect(() => {
    setMounted(true)
  })

  return (
    <section className="hero-section relative flex min-h-[100dvh] items-center justify-center overflow-hidden">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-1/2 -left-1/2 h-[200%] w-[200%] bg-[radial-gradient(circle_at_40%_40%,rgba(0,0,0,0.03)_0%,transparent_50%)]" />
        <div className="absolute -bottom-1/2 -right-1/2 h-[200%] w-[200%] bg-[radial-gradient(circle_at_60%_60%,rgba(0,0,0,0.03)_0%,transparent_50%)]" />
      </div>

      <div
        className="pointer-events-none absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(0,0,0,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.1) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      <div className="relative z-10 mx-auto max-w-[1400px] px-6 py-20 text-center">
        <p
          className={`hero-eyebrow mb-6 text-xs font-medium uppercase tracking-[0.25em] text-muted-foreground ${mounted ? "hero-visible" : ""}`}
        >
          {t("eyebrow")}
        </p>

        <h1
          className={`hero-headline mx-auto max-w-4xl text-[clamp(2.5rem,6vw,5rem)] font-medium leading-[1.05] tracking-[-0.03em] text-foreground ${mounted ? "hero-visible" : ""}`}
        >
          {t("headline")}
          <br />
          {t("headlineLine2")}
        </h1>

        <p
          className={`hero-subtitle mx-auto mt-6 max-w-xl text-lg leading-relaxed text-muted-foreground ${mounted ? "hero-visible" : ""}`}
        >
          {t("subtitle")}
        </p>

        <div
          className={`hero-cta mt-10 flex flex-col items-center gap-4 sm:flex-row sm:justify-center ${mounted ? "hero-visible" : ""}`}
        >
          <Button
            size="lg"
            className="h-14 rounded-full px-10 text-base font-medium"
            asChild
          >
            <a href="/register">{t("ctaPrimary")}</a>
          </Button>
          <Button
            variant="ghost"
            size="lg"
            className="h-14 rounded-full px-8 text-base font-medium text-muted-foreground hover:text-foreground"
            asChild
          >
            <a href="#features">{t("ctaSecondary")}</a>
          </Button>
        </div>
      </div>

      <div className={`hero-scroll absolute bottom-8 left-1/2 -translate-x-1/2 ${mounted ? "hero-visible" : ""}`}>
        <div className="hero-bounce">
          <ChevronDown className="h-5 w-5 text-muted-foreground/50" />
        </div>
      </div>
    </section>
  )
}
