"use client"

import { useState } from "react"
import { toast } from "sonner"
import { useTranslations } from "@/i18n"
import {
  useAcceptConnection,
  useEndConnection,
  useInviteConnection,
  usePendingConnections,
  useConnections,
} from "@/lib/api/connections"

export default function ConnectionsPage() {
  const { data: conns } = useConnections()
  const { data: pending } = usePendingConnections()
  const invite = useInviteConnection()
  const acceptConn = useAcceptConnection()
  const endConn = useEndConnection()
  const t = useTranslations("toast")
  const tc = useTranslations("connections")

  const [email, setEmail] = useState("")

  const handleInvite = async () => {
    if (!email) return
    try {
      await invite.mutateAsync({ to_user_email: email, connection_type: "coaching" })
      toast.success(t("invitationSent"))
      setEmail("")
    } catch {
      toast.error(t("inviteSendError"))
    }
  }

  const activeConns = (conns?.connections ?? []).filter(r => r.status === "active")

  return (
    <div className="mx-auto max-w-2xl space-y-6 sm:max-w-3xl">
      <h1 className="text-lg font-semibold">{tc("title")}</h1>

      <div className="space-y-2">
        <p className="text-sm font-medium">{tc("inviteStudent")}</p>
        <div className="flex gap-2">
          <input
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            placeholder="email@example.com"
            className="flex-1 rounded-xl border border-border bg-background px-3 py-2.5 text-sm"
          />
          <button
            type="button"
            onClick={handleInvite}
            className="whitespace-nowrap rounded-xl bg-primary text-primary-foreground px-4 py-2.5 text-sm"
          >
            {tc("invite")}
          </button>
        </div>
      </div>

      {pending && pending.connections.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">{tc("incomingInvites")}</p>
          {pending.connections.map(r => (
            <div
              key={r.id}
              className="flex items-center justify-between rounded-xl border border-border p-3"
            >
              <span className="text-sm truncate mr-2">{r.from_user_name ?? r.from_user_id}</span>
              <button
                type="button"
                onClick={() => acceptConn.mutateAsync(r.id)}
                className="shrink-0 rounded-lg bg-primary px-3 py-1.5 text-xs text-primary-foreground"
              >
                {tc("accept")}
              </button>
            </div>
          ))}
        </div>
      )}

      {activeConns.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">{tc("activeConnections")}</p>
          {activeConns.map(r => (
            <div
              key={r.id}
              className="flex items-center justify-between rounded-xl border border-border p-3"
            >
              <span className="text-sm truncate mr-2">{r.to_user_name ?? r.to_user_id}</span>
              <button
                type="button"
                onClick={() => endConn.mutateAsync(r.id)}
                className="shrink-0 text-xs text-muted-foreground hover:text-destructive"
              >
                {tc("endConnection")}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
