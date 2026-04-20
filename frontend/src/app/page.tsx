import { cookies } from "next/headers"
import { redirect } from "next/navigation"
import { skipAuth } from "@/lib/env"

export default async function HomePage() {
  if (skipAuth) redirect("/feed")

  const hasAuth = (await cookies()).get("sb_auth")?.value
  redirect(hasAuth ? "/feed" : "/login")
}
