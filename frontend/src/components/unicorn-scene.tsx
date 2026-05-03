"use client"

import Script from "next/script"
import { useRef } from "react"

interface UnicornSceneProps {
  projectId: string
  className?: string
  style?: React.CSSProperties
  lazy?: boolean
}

export function UnicornScene({ projectId, className, style, lazy = true }: UnicornSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  return (
    <>
      <Script
        src="https://cdn.jsdelivr.net/gh/hiunicornstudio/unicornstudio.js@v2.1.11/dist/unicornStudio.umd.js"
        strategy="lazyOnload"
        onLoad={() => {
          if (typeof window !== "undefined" && "UnicornStudio" in window) {
            const us = (window as unknown as { UnicornStudio: { init: () => void } }).UnicornStudio
            us.init()
          }
        }}
      />
      <div
        ref={containerRef}
        className={className}
        style={style}
        data-us-project={projectId}
        data-us-lazyload={lazy ? "true" : undefined}
        data-us-production="true"
        data-us-scale="0.75"
        data-us-dpi="1.25"
        data-us-fps="45"
      />
    </>
  )
}
