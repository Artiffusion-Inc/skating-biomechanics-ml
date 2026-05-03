"use client"

import { useState, useCallback } from "react"
import { RoleSelect } from "./role-select"
import { TourSlider } from "./tour-slider"

export type UserRole = "skater" | "coach" | "choreographer"

export interface OnboardingData {
  role: UserRole
  source: string
  skipped: boolean
}

interface OnboardingFlowProps {
  onComplete: (data: OnboardingData) => void
}

export function OnboardingFlow({ onComplete }: OnboardingFlowProps) {
  const [step, setStep] = useState<"role" | "tour">("role")
  const [data, setData] = useState<Partial<OnboardingData>>({})

  const handleRoleSelect = useCallback((role: UserRole, source: string) => {
    setData({ role, source })
    setStep("tour")
  }, [])

  const handleSkip = useCallback(() => {
    onComplete({
      role: data.role ?? "skater",
      source: data.source ?? "",
      skipped: true,
    })
  }, [data, onComplete])

  const handleTourComplete = useCallback(() => {
    onComplete({
      role: data.role ?? "skater",
      source: data.source ?? "",
      skipped: false,
    })
  }, [data, onComplete])

  return (
    <div className="fixed inset-0 z-[100] bg-background">
      {step === "role" && <RoleSelect onSelect={handleRoleSelect} onSkip={handleSkip} />}
      {step === "tour" && data.role && (
        <TourSlider role={data.role} onComplete={handleTourComplete} onSkip={handleSkip} />
      )}
    </div>
  )
}
