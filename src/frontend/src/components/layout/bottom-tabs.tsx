"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { BarChart3, Camera, Feed, Users } from "lucide-react"

const skaterTabs = [
  { href: "/feed", icon: Feed, label: "Лента" },
  { href: "/upload", icon: Camera, label: "Запись" },
  { href: "/progress", icon: BarChart3, label: "Прогресс" },
  { href: "/profile", icon: Users, label: "Профиль" },
]

const coachTabs = [
  { href: "/dashboard", icon: Users, label: "Ученики" },
  { href: "/upload", icon: Camera, label: "Запись" },
  { href: "/progress", icon: BarChart3, label: "Прогресс" },
  { href: "/profile", icon: Users, label: "Профиль" },
]

export function BottomTabs({ isCoach }: { isCoach: boolean }) {
  const pathname = usePathname()
  const tabs = isCoach ? coachTabs : skaterTabs

  return (
    <nav className="fixed inset-x-0 bottom-0 z-50 border-t border-border bg-background md:hidden">
      <div className="flex items-center justify-around h-16">
        {tabs.map((tab) => {
          const active = pathname === tab.href || pathname.startsWith(tab.href + "/")
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 text-xs ${
                active ? "text-foreground" : "text-muted-foreground"
              }`}
            >
              <tab.icon className="h-5 w-5" />
              {tab.label}
            </Link>
          )
        })}
      </div>
    </nav>
  )
}
