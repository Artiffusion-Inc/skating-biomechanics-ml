"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { BarChart3, Camera, Link2, Newspaper, Settings, Users } from "lucide-react"

interface SidebarProps {
  hasStudents: boolean
}

const commonLinks = [
  { href: "/feed", icon: Newspaper, label: "Лента" },
  { href: "/upload", icon: Camera, label: "Загрузить" },
  { href: "/progress", icon: BarChart3, label: "Прогресс" },
  { href: "/connections", icon: Link2, label: "Связи" },
  { href: "/settings", icon: Settings, label: "Настройки" },
]

const coachLinks = [
  { href: "/dashboard", icon: Users, label: "Ученики" },
]

export function Sidebar({ hasStudents }: SidebarProps) {
  const pathname = usePathname()

  const isActive = (href: string) => pathname === href || pathname.startsWith(href + "/")

  return (
    <aside className="hidden md:flex w-56 flex-col border-r border-border bg-background h-[calc(100vh-60px)] sticky top-[60px]">
      <nav className="flex-1 space-y-1 p-4">
        {hasStudents && (
          <>
            {coachLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
                  isActive(link.href) ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-accent/50"
                }`}
              >
                <link.icon className="h-4 w-4" />
                {link.label}
              </Link>
            ))}
            <div className="my-2 border-t border-border" />
          </>
        )}
        {commonLinks.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
              isActive(link.href) ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-accent/50"
            }`}
          >
            <link.icon className="h-4 w-4" />
            {link.label}
          </Link>
        ))}
      </nav>
    </aside>
  )
}
