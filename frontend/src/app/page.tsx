import { cookies } from "next/headers"
import { redirect } from "next/navigation"
import { skipAuth } from "@/lib/env"
import { LandingPage } from "@/components/landing"

export default async function HomePage() {
  if (skipAuth) redirect("/feed")

  const hasAuth = (await cookies()).get("sb_auth")?.value
  if (hasAuth) redirect("/feed")

  return <LandingPage />
}
