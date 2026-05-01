"use client"

import { Html } from "@react-three/drei"
import { useTranslations } from "next-intl"
import type { Vector3 } from "three"

const JOINT_NAMES = [
  "pelvis",
  "rightHip",
  "rightKnee",
  "rightAnkle",
  "leftHip",
  "leftKnee",
  "leftAnkle",
  "spine",
  "thorax",
  "neck",
  "head",
  "leftShoulder",
  "leftElbow",
  "leftWrist",
  "rightShoulder",
  "rightElbow",
  "rightWrist",
] as const

interface JointLabelProps {
  jointIndex: number
  position: Vector3
}

export function JointLabel({ jointIndex, position }: JointLabelProps) {
  const t = useTranslations("joints")
  const name = JOINT_NAMES[jointIndex]
  if (!name) return null

  return (
    <Html position={position} center>
      <div className="bg-background/90 text-xs whitespace-nowrap rounded-md px-2 py-1 pointer-events-none shadow-sm">
        {t(name)}
      </div>
    </Html>
  )
}
