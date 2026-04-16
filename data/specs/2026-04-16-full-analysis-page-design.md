# Full Analysis Page — Design Spec

**Date:** 2026-04-16
**Status:** Draft
**Priority:** HIGH — блокирует запуск пользователей

## Problem Statement

Текущий session detail page показывает только processed video + metrics table. Нет:
- 2D skeleton overlay с keypoints
- 3D reconstruction visualization
- Frame scrubbing timeline
- Joint angles visualization

**Результат:** Платформа не ощущается как полноценный SaaS для биомеханического анализа.

## Solution

Full Analysis Page с:
1. 2D video + skeleton overlay + phase labels
2. 3D skeleton viewer (interactive)
3. Frame scrubbing timeline
4. Joint angles visualization

## Architecture Changes

### Backend Changes

#### 1. Remove CSV bottleneck

**Before:**
```python
# Worker saves CSV to R2
csv_path = out_dir / f"{stem}_biomechanics.csv"
np.save(str(poses_path), poses)
# Frontend fetches and parses
```

**After:**
```python
# Worker returns JSON
return {
    "pose_data": {
        "frames": [0, 10, 20, ...],  # sampled (не все!)
        "poses": [[x,y,c, ...], ...],  # (N_sampled, 17, 3)
        "fps": 30.0
    },
    "frame_metrics": {
        "knee_angles_r": [120, 125, ...],
        "knee_angles_l": [115, 118, ...],
    },
    "summary_metrics": [...],
    "phases": {...},
    "recommendations": [...]
}
```

#### 2. Update Session Model

```python
class Session(Base):
    # Remove
    # csv_url: str | None
    # poses_url: str | None

    # Add
    pose_data: dict | None  # JSONB — sampled poses
    frame_metrics: dict | None  # JSONB — frame-by-frame metrics
```

#### 3. Update SessionResponse Schema

```python
class SessionResponse(BaseModel):
    # ...
    pose_data: PoseData | None
    frame_metrics: FrameMetrics | None
    phases: PhasesData | None
```

### Frontend Changes

#### Component Structure

```
sessions/[id]/page.tsx
├── SessionHeader
├── AnalysisLayout (responsive grid)
│   ├── VideoWithSkeleton (2D)
│   │   ├── VideoPlayer
│   │   ├── SkeletonCanvas
│   │   └── PhaseLabels
│   └── ThreeJSkeletonViewer (3D)
│       ├── Canvas (R3F)
│       ├── SkeletalMesh
│       └── JointAngles
├── PhaseTimeline
├── MetricsTable (existing)
└── Recommendations (existing)
```

#### Data Flow

```
useSession() → {pose_data, frame_metrics, phases}
     ↓
Shared state (Zustand):
  - currentFrame
  - isPlaying
  - playbackSpeed
     ↓
VideoPlayer + SkeletonCanvas + ThreeJSkeletonViewer
```

## Component Specifications

### 1. VideoWithSkeleton

**Purpose:** 2D video с skeleton overlay

**Props:**
```typescript
interface VideoWithSkeletonProps {
  videoUrl: string
  poseData: PoseData
  phases: PhasesData
  currentFrame: number
  onFrameChange: (frame: number) => void
}
```

**Behavior:**
- HTML5 video + canvas overlay
- Skeleton отрисовывается на canvas
- Phase labels (takeoff/peak/landing)
- Frame scrubbing sync

**Skeleton connections (H3.6M 17kp):**
```typescript
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3],  // R leg
  [0, 4], [4, 5], [5, 6],  // L leg
  [0, 7], [7, 8], [8, 9], [9, 10],  // Spine + head
  [9, 11], [11, 12], [12, 13],  // L arm
  [9, 14], [14, 15], [15, 16],  // R arm
]
```

### 2. ThreeJSkeletonViewer

**Purpose:** Interactive 3D skeleton visualization

**Props:**
```typescript
interface ThreeJSkeletonViewerProps {
  poseData: PoseData
  frameMetrics: FrameMetrics
  currentFrame: number
}
```

**Features:**
- OrbitControls (rotate/zoom/pan)
- Skeletal mesh (bones + joints)
- Joint angle visualization
- Color coding: green/yellow/red

**Improved geometry (vs current .glb export):**
```typescript
// Bones — variable radius
function createBone(p1: Vector3, p2: Vector3) {
  const curve = new CatmullRomCurve3([p1, p2])
  const geometry = new TubeGeometry(curve, 16, 0.015, 8, false)
  // Material с metalness/roughness
}

// Joints — proper spheres
function createJoint(pos: Vector3) {
  const geometry = new SphereGeometry(0.025, 32, 32)
  const material = new MeshStandardMaterial({
    color: 0xc8c8c8,
    metalness: 0.3,
    roughness: 0.4,
  })
}
```

### 3. PhaseTimeline

**Purpose:** Frame scrubbing с phase markers

**Props:**
```typescript
interface PhaseTimelineProps {
  totalFrames: number
  phases: PhasesData
  currentFrame: number
  onFrameChange: (frame: number) => void
}
```

**Visual:**
```
[========|========|========]
  flight   landing    recovery
    ↑        ↑         ↑
  takeoff  peak     landing
```

**Features:**
- Colored zones per phase
- Draggable scrubber
- Click marker to jump

### 4. Shared State

```typescript
// stores/analysis.ts
interface AnalysisState {
  currentFrame: number
  isPlaying: boolean
  playbackSpeed: number
  selectedJoint: number | null
}

// Zustand store
export const useAnalysisStore = create<AnalysisState>((set) => ({
  currentFrame: 0,
  isPlaying: false,
  playbackSpeed: 1.0,
  selectedJoint: null,
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  // ...
}))
```

## Technical Requirements

### Libraries

```json
{
  "dependencies": {
    "@react-three/fibre": "^8.x",
    "@react-three/drei": "^9.x",
    "three": "^0.x",
    "zustand": "^4.x",
    "@types/three": "^0.x"
  }
}
```

### Performance

- Sample poses: каждый 10th frame (300 → 30 frames)
- Canvas: requestAnimationFrame
- Three.js: dispose on unmount
- Lazy load pose data

### Error Handling

- Fallback если pose_data null
- Graceful degradation если .npz не загрузился
- Loading states для всех компонентов

## Responsive Design

| Breakpoint | Layout |
|------------|--------|
| Desktop (1024+) | Side-by-side (2D \| 3D) |
| Tablet (768-1023) | Stacked (2D top, 3D bottom) |
| Mobile (<768) | Tabs (2D / 3D toggle) |

## Implementation Tasks

### Phase 1: Backend (1-2 days)

1. [ ] Update Session model (remove csv_url, poses_url)
2. [ ] Add pose_data, frame_metrics columns
3. [ ] Update worker to return JSON
4. [ ] Update SessionResponse schema
5. [ ] Migration script

### Phase 2: Frontend - Data Layer (1 day)

6. [ ] Update types (PoseData, FrameMetrics, PhasesData)
7. [ ] Update useSession hook
8. [ ] Create Zustand store
9. [ ] Test API integration

### Phase 3: Frontend - 2D Visualization (2-3 days)

10. [ ] VideoWithSkeleton component
11. [ ] SkeletonCanvas component
12. [ ] PhaseLabels component
13. [ ] Test 2D visualization

### Phase 4: Frontend - 3D Visualization (3-4 days)

14. [ ] ThreeJSkeletonViewer component
15. [ ] SkeletalMesh component
16. [ ] JointAngles component
17. [ ] OrbitControls setup
18. [ ] Test 3D visualization

### Phase 5: Timeline & Polish (1-2 days)

19. [ ] PhaseTimeline component
20. [ ] Integrate all components
21. [ ] Responsive design
22. [ ] Accessibility (keyboard nav)
23. [ ] Performance optimization

**Total: 8-12 days**

## Success Criteria

- [ ] 2D skeleton overlay работает на всех записях
- [ ] 3D skeleton interactive (rotate/zoom/pan)
- [ ] Frame scrubbing синхронизирован между 2D и 3D
- [ ] Phase markers корректны
- [ ] Mobile responsive
- [ ] Loading states и error handling
- [ ] Performance: < 2s initial load, < 100ms frame update

## Open Questions

1. **Sample rate:** Каждый 10th frame OK? Или больше/меньше?
2. **3D auto-play:** Играть 3D при scrubbing или только по request?
3. **Joint angles:** Показывать все или только selected?
4. **Color scheme:** Использовать текущий (green/yellow/red) или другой?
