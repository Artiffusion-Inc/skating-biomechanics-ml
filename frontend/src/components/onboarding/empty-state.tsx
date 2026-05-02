"use client"

import Link from "next/link"
import { cn } from "@/lib/utils"

interface EmptyStateProps {
  icon?: React.ReactNode
  title: string
  description: string
  primaryAction?: {
    label: string
    href: string
  }
  secondaryAction?: {
    label: string
    href: string
  }
  className?: string
}

export function EmptyState({
  icon,
  title,
  description,
  primaryAction,
  secondaryAction,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center py-20 px-4 text-center",
        className
      )}
    >
      {icon && (
        <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-[1.25rem] bg-ice-deep/5 text-ice-deep">
          {icon}
        </div>
      )}
      <h3 className="mb-2 text-lg font-medium text-foreground">{title}</h3>
      <p className="mb-8 max-w-sm text-sm text-muted-foreground leading-relaxed">
        {description}
      </p>

      <div className="flex flex-col items-center gap-3 sm:flex-row">
        {primaryAction && (
          <Link
            href={primaryAction.href}
            className="inline-flex h-11 items-center justify-center rounded-full px-8 text-sm font-medium text-white transition-all duration-200 hover:scale-[0.98] active:scale-[0.96]"
            style={{ background: "var(--ice-deep)" }}
          >
            {primaryAction.label}
          </Link>
        )}
        {secondaryAction && (
          <Link
            href={secondaryAction.href}
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            {secondaryAction.label}
          </Link>
        )}
      </div>
    </div>
  )
}
