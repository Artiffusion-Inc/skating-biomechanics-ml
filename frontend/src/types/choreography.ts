// ---------------------------------------------------------------------------
// Music Analysis
// ---------------------------------------------------------------------------

export interface MusicSegment {
  type: string
  start: number
  end: number
}

export interface EnergyCurve {
  timestamps: number[]
  values: number[]
}

export interface MusicAnalysis {
  id: string
  user_id: string
  filename: string
  audio_url: string
  duration_sec: number
  bpm: number | null
  meter: string | null
  structure: MusicSegment[] | null
  energy_curve: EnergyCurve | null
  downbeats: number[] | null
  peaks: number[] | null
  status: "pending" | "analyzing" | "completed" | "failed"
  created_at: string
  updated_at: string
}

export interface UploadMusicResponse {
  music_id: string
  filename: string
}

// ---------------------------------------------------------------------------
// Layout Generation
// ---------------------------------------------------------------------------

export interface LayoutElement {
  code: string
  goe: number
  timestamp: number
  position: { x: number; y: number } | null
  is_back_half: boolean
  is_jump_pass: boolean
  jump_pass_index: number | null
}

export interface Layout {
  elements: LayoutElement[]
  total_tes: number
  back_half_indices: number[]
}

export interface GenerateResponse {
  layouts: Layout[]
}

export interface ValidationResult {
  is_valid: boolean
  errors: string[]
  warnings: string[]
  total_tes: number | null
}

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

export interface ProgramLayout {
  elements: LayoutElement[]
}

export interface ChoreographyProgram {
  id: string
  user_id: string
  music_analysis_id: string | null
  title: string | null
  discipline: "mens_singles" | "womens_singles"
  segment: "short_program" | "free_skate"
  season: string
  layout: ProgramLayout | null
  total_tes: number | null
  estimated_goe: number | null
  estimated_pcs: number | null
  estimated_total: number | null
  is_valid: boolean | null
  validation_errors: string[] | null
  validation_warnings: string[] | null
  created_at: string
  updated_at: string
}

export interface ProgramListResponse {
  programs: ChoreographyProgram[]
  total: number
}

// ---------------------------------------------------------------------------
// Element Inventory
// ---------------------------------------------------------------------------

export interface Inventory {
  jumps: string[]
  spins: string[]
  combinations: string[]
}
