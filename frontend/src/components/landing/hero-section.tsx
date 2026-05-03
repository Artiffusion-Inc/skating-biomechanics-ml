"use client"

import { useState } from "react"
import { useMountEffect } from "@/lib/useMountEffect"
import { Button } from "@/components/ui/button"
import { ChevronDown } from "lucide-react"
import { useTranslations } from "@/i18n"
import { UnicornScene } from "@/components/unicorn-scene"

export function HeroSection() {
  const [mounted, setMounted] = useState(false)
  const t = useTranslations("landing")

  useMountEffect(() => {
    setMounted(true)
  })

  return (
    <section className="hero-section relative flex min-h-[100dvh] items-center overflow-hidden ice-gradient">
      {/* Unicorn Studio WebGL background — replace YOUR_PROJECT_ID */}
      <div className="absolute inset-0 z-0">
        <UnicornScene
          projectId="YOUR_PROJECT_ID"
          className="h-full w-full"
          lazy={false}
        />
      </div>

      {/* Diagonal ice streak */}
      <div className="diagonal-streak" />

      {/* Subtle noise texture overlay */}
      <div
        className="pointer-events-none absolute inset-0 z-[1] opacity-[0.4]"
        style={{
          backgroundImage:
            "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E\")",
          backgroundSize: "128px 128px",
        }}
      />

      {/* Grid lines — ice rink markings suggestion */}
      <div
        className="pointer-events-none absolute inset-0 z-[1] opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(0,0,0,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.3) 1px, transparent 1px)",
          backgroundSize: "80px 80px",
        }}
      />

      <div className="relative z-10 mx-auto w-full max-w-[1400px] px-4 py-16 sm:px-6 sm:py-20">
        <div className="grid items-center gap-8 md:gap-12 lg:grid-cols-2">
          {/* Left: text */}
          <div className="text-left">
            <p
              className={`hero-eyebrow mb-4 md:mb-6 text-xs font-medium uppercase tracking-[0.3em] text-muted-foreground ${mounted ? "hero-visible" : ""}`}
            >
              {t("eyebrow")}
            </p>

            <h1
              className={`hero-headline max-w-2xl text-[clamp(2.5rem,5.5vw,4.5rem)] font-medium leading-[1.05] tracking-[-0.03em] text-foreground ${mounted ? "hero-visible" : ""}`}
            >
              {t("headline")}
              <br />
              <span style={{ color: "var(--ice-deep)" }}>{t("headlineLine2")}</span>
            </h1>

            <p
              className={`hero-subtitle mt-4 md:mt-6 max-w-md text-lg leading-relaxed text-muted-foreground ${mounted ? "hero-visible" : ""}`}
            >
              {t("subtitle")}
            </p>

            <div
              className={`hero-cta mt-8 md:mt-10 flex flex-col items-start gap-4 sm:flex-row ${mounted ? "hero-visible" : ""}`}
            >
              <Button
                size="lg"
                className="h-14 rounded-full px-10 text-base font-medium"
                style={{ background: "var(--ice-deep)", color: "white" }}
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

          {/* Right: abstract ice composition */}
          <div
            className={`relative hidden lg:block ${mounted ? "hero-visible hero-eyebrow" : ""}`}
          >
            <div className="relative aspect-square max-w-lg">
              {/* Ice glow orb */}
              <div
                className="absolute inset-0 rounded-full opacity-20 blur-3xl"
                style={{ background: "var(--ice-glow)" }}
              />
              {/* Floating cards — glassmorphism */}
              <div
                className="absolute top-[10%] left-[10%] rounded-2xl p-5"
                style={{
                  background: "oklch(0.95 0.01 240 / 0.45)",
                  backdropFilter: "blur(16px)",
                  border: "1px solid oklch(0.85 0.03 240 / 0.5)",
                  boxShadow: "0 4px 24px oklch(0.42 0.12 240 / 0.15)",
                }}
              >
                <p className="text-xs font-mono" style={{ color: "var(--ice-deep)" }}>CoM</p>
                <p className="text-2xl font-medium text-foreground">1.24 m</p>
              </div>
              <div
                className="absolute right-[15%] bottom-[20%] rounded-2xl p-5"
                style={{
                  background: "oklch(0.95 0.01 240 / 0.45)",
                  backdropFilter: "blur(16px)",
                  border: "1px solid oklch(0.85 0.03 240 / 0.5)",
                  boxShadow: "0 4px 24px oklch(0.42 0.12 240 / 0.15)",
                }}
              >
                <p className="text-xs font-mono" style={{ color: "var(--ice-deep)" }}>Rotation</p>
                <p className="text-2xl font-medium text-foreground">540°</p>
              </div>
              <div
                className="absolute top-[40%] right-[5%] rounded-2xl p-4"
                style={{
                  background: "oklch(0.95 0.01 240 / 0.45)",
                  backdropFilter: "blur(16px)",
                  border: "1px solid oklch(0.85 0.03 240 / 0.5)",
                  boxShadow: "0 4px 24px oklch(0.42 0.12 240 / 0.15)",
                }}
              >
                <p className="text-xs font-mono" style={{ color: "var(--ice-deep)" }}>Airtime</p>
                <p className="text-xl font-medium text-foreground">0.72 s</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div
        className={`hero-scroll absolute bottom-8 left-1/2 -translate-x-1/2 ${mounted ? "hero-visible" : ""}`}
      >
        <div className="hero-bounce">
          <ChevronDown className="h-5 w-5" style={{ color: "var(--ice-deep)" }} />
        </div>
      </div>
    </section>
  )
}
