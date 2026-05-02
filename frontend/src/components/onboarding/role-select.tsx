"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import type { UserRole } from "./onboarding-flow"

interface RoleSelectProps {
  onSelect: (role: UserRole, source: string) => void
  onSkip: () => void
}

const ROLES: { id: UserRole; label: string; description: string }[] = [
  { id: "skater", label: "Фигурист", description: "Загружайте видео, получайте аналитику и отслеживайте прогресс" },
  { id: "coach", label: "Тренер", description: "Приглашайте учеников, следите за прогрессом и давайте обратную связь" },
  { id: "choreographer", label: "Хореограф", description: "Создавайте программы, планируйте элементы и визуализируйте раскладку" },
]

const SOURCES = [
  "От друга",
  "В соцсетях",
  "На соревнованиях",
  "От тренера",
  "Другое",
]

export function RoleSelect({ onSelect, onSkip }: RoleSelectProps) {
  const [selectedRole, setSelectedRole] = useState<UserRole | null>(null)
  const [source, setSource] = useState("")

  const handleContinue = () => {
    if (selectedRole) {
      onSelect(selectedRole, source)
    }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-12">
      <div className="mx-auto w-full max-w-lg">
        <div className="mb-6 flex items-center justify-end">
          <button
            onClick={onSkip}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Пропустить
          </button>
        </div>

        <div className="mb-10 text-center">
          <h1 className="mb-3 text-2xl font-medium tracking-tight text-foreground">
            Кто вы?
          </h1>
          <p className="text-sm text-muted-foreground">
            Выберите роль, которая описывает вас сейчас. Можно изменить позже.
          </p>
        </div>

        <div className="space-y-3">
          {ROLES.map((role) => (
            <button
              key={role.id}
              onClick={() => setSelectedRole(role.id)}
              className={cn(
                "w-full rounded-[1.25rem] border p-5 text-left transition-all duration-200",
                selectedRole === role.id
                  ? "border-ice-deep bg-ice-deep/5"
                  : "border-border bg-card hover:bg-accent"
              )}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-base font-medium text-foreground">{role.label}</p>
                  <p className="mt-1 text-sm text-muted-foreground">{role.description}</p>
                </div>
                <div
                  className={cn(
                    "ml-4 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border-2",
                    selectedRole === role.id
                      ? "border-ice-deep bg-ice-deep"
                      : "border-muted-foreground/30"
                  )}
                >
                  {selectedRole === role.id && (
                    <svg className="h-3 w-3 text-white" viewBox="0 0 12 12" fill="currentColor">
                      <path d="M10.28 2.28a1 1 0 00-1.41 0L4 7.17 2.41 5.59a1 1 0 10-1.42 1.41l2.17 2.17a1 1 0 001.42 0l5.7-5.7a1 1 0 000-1.41z" />
                    </svg>
                  )}
                </div>
              </div>
            </button>
          ))}
        </div>

        <div className="mt-8">
          <label className="mb-2 block text-sm font-medium text-foreground">
            Как вы узнали о нас?
            <span className="ml-1 text-muted-foreground font-normal">(опционально)</span>
          </label>
          <select
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="w-full rounded-lg border border-border bg-background px-3 py-2.5 text-sm text-foreground outline-none transition-colors focus:border-foreground"
          >
            <option value="">Выберите вариант</option>
            {SOURCES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        <div className="mt-8">
          <Button
            size="lg"
            className="h-12 w-full rounded-full bg-ice-deep text-base font-medium text-white hover:bg-ice-glow hover:text-foreground"
            disabled={!selectedRole}
            onClick={handleContinue}
          >
            Продолжить
          </Button>
        </div>
      </div>
    </div>
  )
}
