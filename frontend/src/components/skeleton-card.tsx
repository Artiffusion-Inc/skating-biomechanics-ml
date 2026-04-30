"use client"

export function SkeletonCard() {
  return (
    <div className="rounded-2xl border border-border bg-background p-3 sm:p-4 animate-pulse">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 space-y-2 flex-1">
          <div className="h-4 w-1/3 rounded bg-muted" />
          <div className="h-3 w-1/2 rounded bg-muted" />
        </div>
        <div className="h-4 w-8 rounded bg-muted shrink-0" />
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5">
        <div className="h-3 w-16 rounded bg-muted" />
        <div className="h-3 w-20 rounded bg-muted" />
        <div className="h-3 w-14 rounded bg-muted" />
      </div>
    </div>
  )
}
