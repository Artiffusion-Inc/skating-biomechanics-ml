"use client"

import Link from "next/link"
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
      <div className="flex flex-col items-center gap-4 py-20">
        <p className="text-muted-foreground">Нет учеников</p>
        <p className="text-sm text-muted-foreground">Пригласите фигуриста для отслеживания прогресса</p>
        <Link href="/connections" className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors">
          Пригласить ученика
        </Link>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-3 sm:max-w-3xl">
      <h1 className="nike-h3">Ученики</h1>
      {students.map((rel) => (
        <StudentCard key={rel.id} rel={rel} />
      ))}
    </div>
  )
}
