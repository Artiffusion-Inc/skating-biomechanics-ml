/// <reference path="../../../react-three.d.ts" />
"use client"

import { Environment, Grid, OrbitControls, PerspectiveCamera } from "@react-three/drei"
import { Canvas } from "@react-three/fiber"
import { Suspense } from "react"
import { useAnalysisStore } from "@/stores/analysis"
import type { FrameMetrics, PoseData } from "@/types"
import { SkeletalMesh } from "./skeletal-mesh"

interface ThreeJSkeletonViewerProps {
  poseData: PoseData
  frameMetrics: FrameMetrics | null
  className?: string
}

function LoadingFallback() {
  return (
    <div className="flex h-full items-center justify-center text-muted-foreground">
      Загрузка 3D...
    </div>
  )
}

function Scene({
  poseData,
  frameMetrics,
}: {
  poseData: PoseData
  frameMetrics: FrameMetrics | null
}) {
  const { currentFrame } = useAnalysisStore()

  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 0, 1.5]} fov={50} />
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={0.5}
        maxDistance={3}
        target={[0, 0, 0]}
      />

      {/* Lighting using Environment */}
      <Environment preset="city" />

      {/* Skeleton */}
      <SkeletalMesh poseData={poseData} frameMetrics={frameMetrics} currentFrame={currentFrame} />

      {/* Grid helper */}
      <Grid
        args={[1, 10, 0x444444, 0x222222]}
        position={[0, -0.3, 0]}
        cellColor="#444444"
        sectionColor="#222222"
      />
    </>
  )
}

export function ThreeJSkeletonViewer({
  poseData,
  frameMetrics,
  className = "",
}: ThreeJSkeletonViewerProps) {
  return (
    <div
      className={`relative aspect-square bg-gradient-to-br from-slate-900 to-slate-800 ${className}`}
    >
      <Canvas
        dpr={[1, 2]} // Pixel ratio for sharp rendering
        gl={{ antialias: true, alpha: true }}
        className="w-full h-full"
      >
        <Suspense fallback={<LoadingFallback />}>
          <Scene poseData={poseData} frameMetrics={frameMetrics} />
        </Suspense>
      </Canvas>

      {/* Legend */}
      <div className="absolute bottom-2 left-2 rounded-lg bg-black/60 p-2 text-xs text-white">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <div className="h-2 w-2 rounded-full bg-green-500" />
            <span>90-170°</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="h-2 w-2 rounded-full bg-yellow-500" />
            <span>60-190°</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="h-2 w-2 rounded-full bg-red-500" />
            <span className="text-xs">&lt;60° / &gt;190°</span>
          </div>
        </div>
      </div>
    </div>
  )
}
