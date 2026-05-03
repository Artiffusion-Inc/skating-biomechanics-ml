"use client"

import Script from "next/script"
import { useRef, useState, useCallback } from "react"

interface UnicornSceneProps {
  projectId: string
  className?: string
  style?: React.CSSProperties
  lazy?: boolean
}

export function UnicornScene({ projectId, className, style, lazy = true }: UnicornSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [ready, setReady] = useState(false)

  const handleLoad = useCallback(() => {
    if (typeof window !== "undefined") {
      const us = (window as unknown as { UnicornStudio?: { init: () => void } }).UnicornStudio
      if (us) {
        us.init()
      }
    }
    setReady(true)
  }, [])

  return (
    <>
      <Script
        src="https://cdn.jsdelivr.net/npm/unicorn-studio@latest/dist/unicornStudio.umd.js"
        strategy="lazyOnload"
        onLoad={handleLoad}
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
