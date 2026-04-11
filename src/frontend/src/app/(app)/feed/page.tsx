"use client"

import Link from "next/link"
import { useSessions } from "@/lib/api/sessions"
import { SessionCard } from "@/components/session/session-card"

export default function FeedPage() {
  const { data, isLoading } = useSessions()

  if (isLoading) {
    return <div className="flex items-center justify-center py-20 text-muted-foreground">Загрузка...</div>
  }

  if (!data?.sessions.length) {
    return (
      <div className="flex flex-col items-center gap-4 py-20">
        <p className="text-muted-foreground">Нет записей</p>
        <p className="text-sm text-muted-foreground">Запишите видео для анализа</p>
        <Link href="/upload" className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors">
          Записать видео
        </Link>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-3 sm:max-w-3xl">
      {data.sessions.map((session) => (
        <SessionCard key={session.id} session={session} />
      ))}
    </div>
  )
}
