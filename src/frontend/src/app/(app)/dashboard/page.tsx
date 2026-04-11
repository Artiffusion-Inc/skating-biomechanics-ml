"use client"

import { useRelationships } from "@/lib/api/relationships"
import { StudentCard } from "@/components/coach/student-card"

export default function DashboardPage() {
  const { data, isLoading } = useRelationships()

  const students = (data?.relationships ?? []).filter(
    (r) => r.status === "active",
  )

  if (isLoading) return <div className="py-20 text-center text-muted-foreground">Загрузка...</div>

  if (!students.length) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">Нет учеников</p>
        <p className="text-sm text-muted-foreground mt-1">Пригласите первого ученика</p>
      </div>
    )
  }

  return (
    <div className="max-w-lg mx-auto space-y-3">
      <h1 className="text-lg font-semibold">Ученики</h1>
      {students.map((rel) => (
        <StudentCard key={rel.id} rel={rel} />
      ))}
    </div>
  )
}
