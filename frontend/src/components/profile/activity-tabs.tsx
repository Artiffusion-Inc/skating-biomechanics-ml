"use client"

import { type ReactNode, useState } from "react"
import { useTranslations } from "@/i18n"

const TABS = ["activity", "records"] as const

export function ActivityTabs({
  activityContent,
  recordsContent,
}: {
  activityContent: ReactNode
  recordsContent: ReactNode
}) {
  const t = useTranslations("profile")
  const [active, setActive] = useState<"activity" | "records">("activity")

  return (
    <div>
      <div className="mb-3 flex gap-1 rounded-xl bg-muted/50 p-1">
        {TABS.map(tab => (
          <button
            key={tab}
            type="button"
            onClick={() => setActive(tab)}
            className={`flex-1 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              active === tab
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab === "activity" ? t("recentActivity") : t("personalRecords")}
          </button>
        ))}
      </div>
      {active === "activity" ? activityContent : recordsContent}
    </div>
  )
}
