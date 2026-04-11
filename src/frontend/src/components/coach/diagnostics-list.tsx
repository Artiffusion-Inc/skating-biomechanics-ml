"use client"

import { AlertTriangle, Info } from "lucide-react"
import type { DiagnosticsFinding } from "@/types"

export function DiagnosticsList({ findings }: { findings: DiagnosticsFinding[] }) {
  if (!findings.length) {
    return <p className="text-sm text-muted-foreground">Проблем не обнаружено</p>
  }

  return (
    <div className="space-y-2">
      {findings.map((f, i) => (
        <div
          key={i}
          className={`rounded-xl border p-3 ${
            f.severity === "warning" ? "border-amber-300 bg-amber-50 dark:bg-amber-950/20" : "border-border bg-muted/30"
          }`}
        >
          <div className="flex items-start gap-2">
            {f.severity === "warning" ? (
              <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
            ) : (
              <Info className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
            )}
            <div>
              <p className="text-sm font-medium">{f.message}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{f.detail}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
