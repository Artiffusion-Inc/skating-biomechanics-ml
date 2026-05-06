import { cookies } from "next/headers"
import { redirect } from "next/navigation"

/**
 * Full-viewport layout — no app shell (header/bottom dock).
 * For DAW-like pages that need the entire viewport.
 */
export default async function FullLayout({ children }: { children: React.ReactNode }) {
  const hasAuth = (await cookies()).get("sb_auth")?.value
  if (!hasAuth) redirect("/login")

  return <div className="flex h-dvh flex-col overflow-hidden">{children}</div>
}
