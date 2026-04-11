"use client"

import { useSessions } from "@/lib/api/sessions"
import { SessionCard } from "@/components/session/session-card"

export default function FeedPage() {
  const { data, isLoading } = useSessions()

  if (isLoading) {
    return <div className="flex items-center justify-center py-20 text-muted-foreground">Загрузка...</div>
  }

  if (!data?.sessions.length) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">Нет записей</p>
        <p className="text-sm text-muted-foreground mt-1">Загрузите первое видео</p>
      </div>
    )
  }

  return (
    <div className="space-y-3 max-w-lg mx-auto">
      {data.sessions.map((session) => (
        <SessionCard key={session.id} session={session} />
      ))}
    </div>
  )
}
