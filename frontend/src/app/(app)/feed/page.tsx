"use client"

import { useState } from "react"
import Link from "next/link"
import { SessionCard } from "@/components/session/session-card"
import { SkeletonCard } from "@/components/skeleton-card"
import { useTranslations } from "@/i18n"
import { useSessions, useBulkDeleteSessions } from "@/lib/api/sessions"

export default function FeedPage() {
  const { data, isLoading } = useSessions()
  const tf = useTranslations("feed")
  const tc = useTranslations("common")

  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const bulkDelete = useBulkDeleteSessions()

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleBulkDelete = () => {
    if (!window.confirm(tf("bulkDeleteConfirm"))) return
    bulkDelete.mutate(Array.from(selectedIds), {
      onSuccess: () => {
        setSelectedIds(new Set())
        setSelectionMode(false)
      },
    })
  }

  if (isLoading) {
    return (
      <div className="mx-auto max-w-2xl space-y-3 sm:max-w-3xl">
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </div>
    )
  }

  if (!data?.sessions.length) {
    return (
      <div className="flex flex-col items-center gap-4 py-20">
        <p className="text-muted-foreground">{tf("noSessions")}</p>
        <p className="text-sm text-muted-foreground">{tf("noSessionsHint")}</p>
        <Link
          href="/upload"
          className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          {tf("recordVideo")}
        </Link>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-3 sm:max-w-3xl">
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => {
            setSelectionMode(!selectionMode)
            setSelectedIds(new Set())
          }}
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          {selectionMode ? tc("cancel") : tf("select")}
        </button>
        {selectionMode && selectedIds.size > 0 && (
          <button
            type="button"
            onClick={handleBulkDelete}
            disabled={bulkDelete.isPending}
            className="text-sm text-destructive hover:text-destructive/80"
          >
            {bulkDelete.isPending
              ? tc("saving")
              : tf("deleteSelected", { count: selectedIds.size })}
          </button>
        )}
      </div>

      {data.sessions.map(session => (
        <SessionCard
          key={session.id}
          session={session}
          selectable={selectionMode}
          selected={selectedIds.has(session.id)}
          onSelect={toggleSelect}
        />
      ))}
    </div>
  )
}
