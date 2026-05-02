"use client"

import { useEffect, useState } from "react"
import { usePathname, useRouter } from "next/navigation"
import { fetchMe } from "@/lib/auth"

export function OnboardingGate({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const pathname = usePathname()
  const [checked, setChecked] = useState(false)

  useEffect(() => {
    // Skip check on onboarding page itself
    if (pathname === "/onboarding") {
      setChecked(true)
      return
    }

    // Skip for auth pages
    if (pathname.startsWith("/login") || pathname.startsWith("/register")) {
      setChecked(true)
      return
    }

    const check = async () => {
      const localCompleted = localStorage.getItem("onboarding_completed")
      if (localCompleted) {
        setChecked(true)
        return
      }

      // Fallback: check backend profile
      try {
        const user = await fetchMe()
        if (user.onboarding_role) {
          localStorage.setItem("onboarding_completed", "true")
          localStorage.setItem("onboarding_role", user.onboarding_role)
          setChecked(true)
          return
        }
      } catch {
        // ignore
      }

      router.push("/onboarding")
    }

    check()
  }, [pathname, router])

  if (!checked) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div
          className="h-8 w-8 animate-pulse rounded-full"
          style={{ background: "oklch(0.42 0.12 240 / 0.2)" }}
        />
      </div>
    )
  }

  return <>{children}</>
}
