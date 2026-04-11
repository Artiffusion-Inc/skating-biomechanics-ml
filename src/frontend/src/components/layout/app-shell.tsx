"use client"

import { useQuery } from "@tanstack/react-query"
import { z } from "zod"
import { apiFetch } from "@/lib/api-client"
import { BottomTabs } from "./bottom-tabs"
import { Sidebar } from "./sidebar"

const RelationshipListSchema = z.object({
  relationships: z.array(z.object({ status: z.string() })),
})

export function AppShell({ children }: { children: React.ReactNode }) {
  const { data } = useQuery({
    queryKey: ["relationships"],
    queryFn: () => apiFetch("/relationships", RelationshipListSchema),
  })

  const hasStudents = (data?.relationships ?? []).some(
    (r) => r.status === "active",
  )
  const isCoach = hasStudents

  return (
    <>
      <div className="flex">
        <Sidebar hasStudents={hasStudents} />
        <div className="flex-1 min-w-0">{children}</div>
      </div>
      <BottomTabs isCoach={isCoach} />
      {/* Bottom padding on mobile for tab bar */}
      <div className="h-16 md:hidden" />
    </>
  )
}
