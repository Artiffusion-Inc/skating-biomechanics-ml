"use client"

import { useLayoutEffect, useRef, useState, useCallback } from "react"
import type { PoseData } from "@/types"

interface SkeletonCanvasProps {
  poseData: PoseData
  currentFrame: number
  width: number
  height: number
}

// H3.6M 17-keypoint skeleton connections
const CONNECTIONS = [
  // Right leg
  [0, 1],
  [1, 2],
  [2, 3],
  // Left leg
  [0, 4],
  [4, 5],
  [5, 6],
  // Spine + head
  [0, 7],
  [7, 8],
  [8, 9],
  [9, 10],
  // Left arm
  [9, 11],
  [11, 12],
  [12, 13],
  // Right arm
  [9, 14],
  [14, 15],
  [15, 16],
]

// Joint colors (COCO 17kp format)
const JOINT_COLORS = [
  "#FF0000", // 0: hip_center (red)
  "#00FF00", // 1: r_hip (green)
  "#00FF00", // 2: r_knee
  "#00FF00", // 3: r_foot
  "#0000FF", // 4: l_hip (blue)
  "#0000FF", // 5: l_knee
  "#0000FF", // 6: l_foot
  "#FFFF00", // 7: spine (yellow)
  "#FFFF00", // 8: thorax
  "#FF00FF", // 9: neck (magenta)
  "#FF00FF", // 10: head
  "#00FFFF", // 11: l_shoulder (cyan)
  "#00FFFF", // 12: l_elbow
  "#00FFFF", // 13: l_wrist
  "#FFA500", // 14: r_shoulder (orange)
  "#FFA500", // 15: r_elbow
  "#FFA500", // 16: r_wrist
]

export function SkeletonCanvas({ poseData, currentFrame, width, height }: SkeletonCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [hoverJoint, setHoverJoint] = useState<number | null>(null)
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 })

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const rect = canvas.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      setMousePos({ x: mx, y: my })

      const frameIndex = poseData.frames.indexOf(currentFrame)
      if (frameIndex === -1) return
      const pose = poseData.poses[frameIndex]
      if (!pose) return

      let closest = -1
      let closestDist = Infinity
      for (let i = 0; i < pose.length; i++) {
        const joint = pose[i]
        if (!joint) continue
        const [x, y, conf] = joint
        if (conf < 0.3) continue
        const dx = x * width - mx
        const dy = y * height - my
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 20 && dist < closestDist) {
          closest = i
          closestDist = dist
        }
      }
      setHoverJoint(closest)
    },
    [currentFrame, poseData, width, height],
  )

  const handleMouseLeave = useCallback(() => setHoverJoint(null), [])

  // biome-ignore lint/correctness/useExhaustiveDependencies: mousePos required for tooltip redraw
  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Find the frame index in sampled data
    const frameIndex = poseData.frames.indexOf(currentFrame)
    if (frameIndex === -1) return

    const pose = poseData.poses[frameIndex]
    if (!pose) return

    // Draw skeleton connections
    ctx.strokeStyle = "rgba(255, 255, 255, 0.6)"
    ctx.lineWidth = 2

    for (const [start, end] of CONNECTIONS) {
      const startJoint = pose[start]
      const endJoint = pose[end]

      if (!startJoint || !endJoint) continue

      const [x1, y1, conf1] = startJoint
      const [x2, y2, conf2] = endJoint

      if (conf1 < 0.3 || conf2 < 0.3) continue // Skip low-confidence joints

      ctx.beginPath()
      ctx.moveTo(x1 * width, y1 * height)
      ctx.lineTo(x2 * width, y2 * height)
      ctx.stroke()
    }

    // Draw joints
    for (let i = 0; i < pose.length; i++) {
      const joint = pose[i]
      if (!joint) continue

      const [x, y, conf] = joint
      if (conf < 0.3) continue // Skip low-confidence joints

      ctx.beginPath()
      ctx.arc(x * width, y * height, 4, 0, Math.PI * 2)
      ctx.fillStyle = JOINT_COLORS[i] || "#FFFFFF"
      ctx.fill()
    }

    // Draw hover label
    if (hoverJoint !== null) {
      const joint = pose[hoverJoint]
      if (joint) {
        const [x, y] = joint
        const px = x * width
        const py = y * height
        ctx.font = "12px Inter, sans-serif"
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
        const text = `Joint ${hoverJoint}`
        const tw = ctx.measureText(text).width
        ctx.fillRect(px + 8, py - 16, tw + 8, 20)
        ctx.fillStyle = "#fff"
        ctx.fillText(text, px + 12, py - 2)
      }
    }
  }, [poseData, currentFrame, width, height, hoverJoint, mousePos])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute inset-0"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    />
  )
}
