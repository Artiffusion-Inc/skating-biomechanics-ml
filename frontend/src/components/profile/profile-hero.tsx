"use client"

import { Check, Pencil, X } from "lucide-react"
import { type FormEvent, useRef, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { useTranslations } from "@/i18n"

export function ProfileHero() {
  const { user } = useAuth()
  const t = useTranslations("profile")

  const [editingName, setEditingName] = useState(false)
  const [editingBio, setEditingBio] = useState(false)
  const [name, setName] = useState("")
  const [bio, setBio] = useState("")
  const nameRef = useRef<HTMLInputElement>(null)
  const bioRef = useRef<HTMLTextAreaElement>(null)

  // Current display values (use edited values when editing, otherwise user values)
  const displayName = editingName ? name : (user?.display_name ?? "")
  const displayBio = editingBio ? bio : (user?.bio ?? "")
  const initial = ((user?.display_name ?? user?.email ?? "")[0] || "?").toUpperCase()

  async function save() {
    try {
      const { updateProfile } = await import("@/lib/auth")
      await updateProfile({
        display_name: name || undefined,
        bio: bio || undefined,
      })
      toast.success(t("updateSuccess"))
      setEditingName(false)
      setEditingBio(false)
    } catch {
      toast.error(t("updateError"))
    }
  }

  function startEditName() {
    if (!user) return
    setName(user.display_name ?? "")
    setEditingName(true)
    setTimeout(() => nameRef.current?.select(), 0)
  }

  function startEditBio() {
    if (!user) return
    setBio(user.bio ?? "")
    setEditingBio(true)
    setTimeout(() => bioRef.current?.focus(), 0)
  }

  function cancelEdit() {
    setEditingName(false)
    setEditingBio(false)
  }

  if (!user) return null

  return (
    <div className="flex flex-col items-center gap-3 py-4">
      {/* Avatar */}
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/15 text-2xl font-bold text-primary">
        {initial}
      </div>

      {/* Name — tap to edit */}
      {editingName ? (
        <form
          onSubmit={(e: FormEvent) => {
            e.preventDefault()
            save()
          }}
          className="flex items-center gap-1"
        >
          <input
            ref={nameRef}
            value={name}
            onChange={e => setName(e.target.value)}
            onBlur={() => save()}
            className="w-48 rounded-lg border border-border bg-secondary px-2 py-1 text-center text-base font-semibold outline-none focus-visible:border-foreground"
          />
          <button
            type="button"
            onClick={cancelEdit}
            className="p-1 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </form>
      ) : (
        <button type="button" onClick={startEditName} className="group flex items-center gap-1.5">
          <span className="text-lg font-semibold">{displayName || user.email}</span>
          <Pencil className="h-3.5 w-3.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
        </button>
      )}

      {/* Email */}
      <p className="text-sm text-muted-foreground">{user.email}</p>

      {/* Bio — tap to edit */}
      {editingBio ? (
        <form
          onSubmit={(e: FormEvent) => {
            e.preventDefault()
            save()
          }}
          className="flex w-full max-w-xs items-start gap-1"
        >
          <textarea
            ref={bioRef}
            value={bio}
            onChange={e => setBio(e.target.value)}
            onBlur={() => save()}
            rows={2}
            className="w-full rounded-lg border border-border bg-secondary px-2 py-1 text-center text-sm outline-none focus-visible:border-foreground resize-none"
          />
        </form>
      ) : (
        <button type="button" onClick={startEditBio} className="group max-w-xs text-center">
          <p className="text-sm text-muted-foreground">
            {displayBio || <span className="italic opacity-50">{t("addBio")}</span>}
          </p>
          <Pencil className="mx-auto mt-0.5 h-3 w-3 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
        </button>
      )}
    </div>
  )
}
