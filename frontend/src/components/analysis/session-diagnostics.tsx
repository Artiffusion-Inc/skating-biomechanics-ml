"use client"

import { useTranslations } from "@/i18n"
import { useDiagnostics } from "@/lib/api/metrics"
import { DiagnosticsList } from "@/components/coach/diagnostics-list"

interface Props {
  elementType: string
}

export function SessionDiagnostics({ elementType }: Props) {
  const ts = useTranslations("sessions")
  const { data, isLoading } = useDiagnostics()

  if (isLoading) return <div className="h-20 animate-pulse rounded-xl bg-muted" />
  if (!data?.findings?.length) return null

  const filtered = data.findings.filter(f => f.element === elementType)
  if (!filtered.length) return null

  return (
    <div className="rounded-2xl border border-border p-3 sm:p-4">
      <h2 className="text-sm font-medium mb-2">{ts("recommendations")}</h2>
      <DiagnosticsList findings={filtered} />
    </div>
  )
}
