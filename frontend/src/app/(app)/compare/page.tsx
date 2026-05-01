"use client"

import { SessionComparison } from "@/components/session/session-comparison"
import { useTranslations } from "@/i18n"

export default function ComparePage() {
  const t = useTranslations("compare")
  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-4">
      <h1 className="text-xl font-semibold">{t("title")}</h1>
      <SessionComparison />
    </div>
  )
}
