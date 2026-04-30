"use client"

export function SkeletonChart() {
  return (
    <div className="space-y-2 animate-pulse">
      <div className="flex items-center justify-between text-sm">
        <div className="h-4 w-1/4 rounded bg-muted" />
        <div className="h-4 w-12 rounded bg-muted" />
      </div>
      <div className="h-[200px] sm:h-[250px] rounded-2xl border border-border bg-background p-3">
        <div className="h-full w-full rounded bg-muted/50" />
      </div>
      <div className="h-4 w-1/4 rounded bg-muted" />
    </div>
  )
}
