import { cookies } from "next/headers"
import { redirect } from "next/navigation"
import { LandingPage } from "@/components/landing"

export default async function HomePage() {
  const hasAuth = (await cookies()).get("sb_auth")?.value
  if (hasAuth) redirect("/feed")

  return <LandingPage />
}
