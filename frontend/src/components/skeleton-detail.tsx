"use client"

export function SkeletonDetail() {
  return (
    <div className="mx-auto max-w-2xl space-y-6 sm:max-w-3xl animate-pulse">
      <div className="space-y-2">
        <div className="h-5 w-1/3 rounded bg-muted" />
        <div className="h-3 w-24 rounded bg-muted" />
      </div>

      <div className="rounded-xl border border-border bg-background p-3 sm:p-4">
        <div className="h-[200px] w-full rounded bg-muted/50" />
      </div>

      <div className="rounded-2xl border border-border bg-background p-3 sm:p-4 space-y-2">
        <div className="h-4 w-1/4 rounded bg-muted" />
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          <div className="h-10 rounded bg-muted/50" />
          <div className="h-10 rounded bg-muted/50" />
          <div className="h-10 rounded bg-muted/50" />
          <div className="h-10 rounded bg-muted/50" />
          <div className="h-10 rounded bg-muted/50" />
          <div className="h-10 rounded bg-muted/50" />
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-background p-3 sm:p-4 space-y-2">
        <div className="h-4 w-1/4 rounded bg-muted" />
        <div className="space-y-1">
          <div className="h-3 w-full rounded bg-muted" />
          <div className="h-3 w-5/6 rounded bg-muted" />
          <div className="h-3 w-4/5 rounded bg-muted" />
        </div>
      </div>
    </div>
  )
}
