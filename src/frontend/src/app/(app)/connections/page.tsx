"use client"

import { useState } from "react"
import { toast } from "sonner"
import { useInvite, useRelationships, usePendingInvites, useAcceptInvite, useEndRelationship } from "@/lib/api/relationships"

export default function ConnectionsPage() {
  const { data: rels } = useRelationships()
  const { data: pending } = usePendingInvites()
  const invite = useInvite()
  const acceptInvite = useAcceptInvite()
  const endRel = useEndRelationship()

  const [email, setEmail] = useState("")

  const handleInvite = async () => {
    if (!email) return
    try {
      await invite.mutateAsync({ skater_email: email })
      toast.success("Приглашение отправлено")
      setEmail("")
    } catch {
      toast.error("Ошибка отправки")
    }
  }

  const activeRels = (rels?.relationships ?? []).filter((r) => r.status === "active")

  return (
    <div className="max-w-lg mx-auto space-y-6">
      <h1 className="text-lg font-semibold">Связи</h1>

      <div className="space-y-2">
        <p className="text-sm font-medium">Пригласить ученика</p>
        <div className="flex gap-2">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="email@example.com"
            className="flex-1 rounded-xl border border-border bg-background px-3 py-2 text-sm"
          />
          <button onClick={handleInvite} className="rounded-xl bg-primary text-primary-foreground px-4 py-2 text-sm">
            Пригласить
          </button>
        </div>
      </div>

      {pending && pending.relationships.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Входящие приглашения</p>
          {pending.relationships.map((r) => (
            <div key={r.id} className="flex items-center justify-between rounded-xl border border-border p-3">
              <span className="text-sm">{r.coach_name ?? r.coach_id}</span>
              <button onClick={() => acceptInvite.mutateAsync(r.id)} className="rounded-lg bg-primary px-3 py-1 text-xs text-primary-foreground">
                Принять
              </button>
            </div>
          ))}
        </div>
      )}

      {activeRels.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Активные связи</p>
          {activeRels.map((r) => (
            <div key={r.id} className="flex items-center justify-between rounded-xl border border-border p-3">
              <span className="text-sm">{r.skater_name ?? r.skater_id}</span>
              <button onClick={() => endRel.mutateAsync(r.id)} className="text-xs text-muted-foreground hover:text-red-500">
                Завершить
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
