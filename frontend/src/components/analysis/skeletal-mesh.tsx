/// <reference path="../../../react-three.d.ts" />
"use client"

import { Line } from "@react-three/drei"
import { useMemo } from "react"
import * as THREE from "three"
import type { FrameMetrics, PoseData } from "@/types"

interface BoneProps {
  start: [number, number, number]
  end: [number, number, number]
}

function Bone({ start, end }: BoneProps) {
  const points = useMemo(
    () => [new THREE.Vector3(...start), new THREE.Vector3(...end)],
    [start, end],
  )
  return <Line points={points} color="#cccccc" lineWidth={3} />
}

interface JointProps {
  position: [number, number, number]
  color: number
}

function Joint({ position, color }: JointProps) {
  // Use a small point for joints instead of sphere
  const points = useMemo(() => [new THREE.Vector3(...position)], [position])
  return <Line points={points} color={`#${color.toString(16).padStart(6, "0")}`} lineWidth={8} />
}

interface SkeletalMeshProps {
  poseData: PoseData
  frameMetrics: FrameMetrics | null
  currentFrame: number
}

// H3.6M 17-keypoint skeleton connections (same as 2D)
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

// Joint radius based on importance
const JOINT_RADIUS = 0.025
// Bone radius (slightly thinner than joints)
const BONE_RADIUS = 0.015

// Color coding based on joint angle quality
function getJointColor(
  jointIndex: number,
  frameMetrics: FrameMetrics | null,
  currentFrame: number,
): number {
  if (!frameMetrics) return 0xc8c8c8 // Default gray

  // Map joint index to metric (simplified)
  let metric: (number | null)[] | null = null
  if (jointIndex <= 3)
    metric = frameMetrics.knee_angles_r // Right leg
  else if (jointIndex <= 6)
    metric = frameMetrics.knee_angles_l // Left leg
  else if (jointIndex >= 11 && jointIndex <= 13)
    metric = frameMetrics.hip_angles_r // Left arm
  else if (jointIndex >= 14) metric = frameMetrics.hip_angles_l // Right arm

  if (!metric || metric.length === 0) return 0xc8c8c8

  // Find closest frame index
  const frameIdx = Math.min(currentFrame, metric.length - 1)
  const angle = metric[frameIdx]

  if (angle === null || angle === undefined) return 0xc8c8c8

  // Color coding: green (90-170°), yellow (60-190°), red (outside)
  if (angle >= 90 && angle <= 170) return 0x4ade80 // Green
  if (angle >= 60 && angle <= 190) return 0xfacc15 // Yellow
  return 0xef4444 // Red
}

export function SkeletalMesh({ poseData, frameMetrics, currentFrame }: SkeletalMeshProps) {
  const { joints, bones } = useMemo(() => {
    // Find the frame index in sampled data
    const frameIndex = poseData.frames.indexOf(currentFrame)
    if (frameIndex === -1) return { joints: [], bones: [] }

    const pose = poseData.poses[frameIndex]
    if (!pose) return { joints: [], bones: [] }

    // Convert normalized [0,1] to 3D space [-0.5, 0.5]
    const scale = 1.0
    const jointPositions: Array<{ position: [number, number, number]; color: number }> = []
    const bonePositions: Array<{ start: [number, number, number]; end: [number, number, number] }> =
      []

    // First pass: collect all joint positions
    for (let i = 0; i < pose.length; i++) {
      const [x, y, conf] = pose[i]
      if (conf < 0.3) continue // Skip low-confidence joints

      jointPositions.push({
        position: [(x - 0.5) * scale, (y - 0.5) * scale, 0],
        color: getJointColor(i, frameMetrics, currentFrame),
      })
    }

    // Second pass: create bones
    for (const [start, end] of CONNECTIONS) {
      const startJoint = pose[start]
      const endJoint = pose[end]

      if (!startJoint || !endJoint) continue

      const [x1, y1, conf1] = startJoint
      const [x2, y2, conf2] = endJoint

      if (conf1 < 0.3 || conf2 < 0.3) continue // Skip low-confidence joints

      bonePositions.push({
        start: [(x1 - 0.5) * scale, (y1 - 0.5) * scale, 0],
        end: [(x2 - 0.5) * scale, (y2 - 0.5) * scale, 0],
      })
    }

    return { joints: jointPositions, bones: bonePositions }
  }, [poseData, frameMetrics, currentFrame])

  return (
    <>
      {/* Bones */}
      {bones.map((bone, i) => (
        <Bone
          key={`bone-${i}`}
          start={bone.start as [number, number, number]}
          end={bone.end as [number, number, number]}
        />
      ))}

      {/* Joints */}
      {joints.map((joint, i) => (
        <Joint
          key={`joint-${i}`}
          position={joint.position as [number, number, number]}
          color={joint.color}
        />
      ))}
    </>
  )
}
