"use client"

import { Music, Plus } from "lucide-react"
import Link from "next/link"
import { useTranslations } from "@/i18n"
import { usePrograms } from "@/lib/api/choreography"

export default function ChoreographyPage() {
  const t = useTranslations("choreography")
  const tc = useTranslations("common")
  const { data, isLoading } = usePrograms()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20 text-muted-foreground">
        {tc("loading")}
      </div>
    )
  }

  if (!data?.programs.length) {
    return (
      <div className="mx-auto max-w-2xl space-y-4 px-4 py-6 sm:max-w-3xl">
        <div className="flex items-center justify-between">
          <h1 className="nike-h2">{t("title")}</h1>
          <Link
            href="/choreography/new"
            className="flex items-center gap-1.5 rounded-xl bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            <Plus className="h-4 w-4" />
            New
          </Link>
        </div>
        <div className="flex flex-col items-center gap-4 py-20">
          <Music className="h-10 w-10 text-muted-foreground" />
          <p className="text-muted-foreground">{t("noPrograms")}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4 px-4 py-6 sm:max-w-3xl">
      <div className="flex items-center justify-between">
        <h1 className="nike-h2">{t("title")}</h1>
        <Link
          href="/choreography/new"
          className="flex items-center gap-1.5 rounded-xl bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        >
          <Plus className="h-4 w-4" />
          New
        </Link>
      </div>
      <div className="space-y-2">
        {data.programs.map(p => (
          <Link
            key={p.id}
            href={`/choreography/programs/${p.id}`}
            className="block rounded-2xl border border-border p-3 transition-colors hover:bg-accent/30"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">{p.title || `${p.segment} — ${p.discipline}`}</p>
                <p className="text-xs text-muted-foreground">{p.season.replace("_", "/")}</p>
              </div>
              {p.estimated_total !== null && (
                <span className="text-sm font-bold">{p.estimated_total.toFixed(2)}</span>
              )}
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
