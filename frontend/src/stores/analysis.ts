import { create } from "zustand"

export interface AnalysisState {
  currentFrame: number
  isPlaying: boolean
  playbackSpeed: number
  selectedJoint: number | null
  cameraPreset: "front" | "side" | "top"

  // Actions
  setCurrentFrame: (frame: number) => void
  setIsPlaying: (playing: boolean) => void
  setPlaybackSpeed: (speed: number) => void
  setSelectedJoint: (joint: number | null) => void
  setCameraPreset: (preset: "front" | "side" | "top") => void
  reset: () => void
}

export const useAnalysisStore = create<AnalysisState>(set => ({
  currentFrame: 0,
  isPlaying: false,
  playbackSpeed: 1.0,
  selectedJoint: null,
  cameraPreset: "front",

  setCurrentFrame: frame => set({ currentFrame: frame }),
  setIsPlaying: playing => set({ isPlaying: playing }),
  setPlaybackSpeed: speed => set({ playbackSpeed: speed }),
  setSelectedJoint: joint => set({ selectedJoint: joint }),
  setCameraPreset: preset => set({ cameraPreset: preset }),

  reset: () =>
    set({
      currentFrame: 0,
      isPlaying: false,
      playbackSpeed: 1.0,
      selectedJoint: null,
      cameraPreset: "front",
    }),
}))
