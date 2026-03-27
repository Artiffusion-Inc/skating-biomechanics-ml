"""Automatic phase detection for skating elements.

This module detects key phases like takeoff, peak height, and landing
from pose sequences using biomechanical cues.
"""

import numpy as np
from scipy.signal import find_peaks

from skating_biomechanics_ml.analysis.metrics import BiomechanicsAnalyzer, PhaseDetectionResult
from skating_biomechanics_ml.references.element_defs import ElementDef
from skating_biomechanics_ml.types import ElementPhase, NormalizedPose
from skating_biomechanics_ml.utils.geometry import get_mid_hip


class PhaseDetector:
    """Detect phases of skating elements from pose sequences."""

    def detect_phases(
        self,
        poses: NormalizedPose,
        fps: float,
        element_type: str,
    ) -> PhaseDetectionResult:
        """Detect phases for a skating element.

        Args:
            poses: NormalizedPose (num_frames, 33, 2).
            fps: Frame rate.
            element_type: Type of element (jump or step).

        Returns:
            PhaseDetectionResult with detected boundaries and confidence.
        """
        if element_type in ("waltz_jump", "toe_loop", "flip"):
            return self.detect_jump_phases(poses, fps)
        elif element_type == "three_turn":
            return self.detect_three_turn_phases(poses, fps)
        else:
            # Default: entire sequence as one phase
            return PhaseDetectionResult(
                phases=ElementPhase(
                    name=element_type,
                    start=0,
                    takeoff=0,
                    peak=0,
                    landing=0,
                    end=len(poses) - 1,
                ),
                confidence=0.0,
            )

    def detect_jump_phases(self, poses: NormalizedPose, fps: float) -> PhaseDetectionResult:  # noqa: ARG002
        """Detect jump phases: takeoff, peak, landing.

        Args:
            poses: NormalizedPose (num_frames, 33, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with jump phase boundaries.
        """
        # Get hip Y trajectory (lower = higher)
        hip_y = get_mid_hip(poses)[:, 1]

        # Find peaks (local minima in Y = maxima in height)
        # Use prominence to avoid noise
        peaks, properties = find_peaks(-hip_y, prominence=0.02, distance=10)

        if len(peaks) == 0:
            # No clear jump detected
            return PhaseDetectionResult(
                phases=ElementPhase(
                    name="jump",
                    start=0,
                    takeoff=0,
                    peak=len(poses) // 2,
                    landing=len(poses) - 1,
                    end=len(poses) - 1,
                ),
                confidence=0.0,
            )

        # Use highest peak
        peak_idx = peaks[np.argmax(-properties["prominences"])]

        # Detect takeoff: frame where hip Y starts decreasing rapidly
        # Compute derivative of hip Y
        derivative = np.gradient(hip_y)

        # Takeoff: where derivative becomes negative and sustained
        takeoff_idx = self._find_takeoff(derivative, peak_idx)

        # Detect landing: frame where hip Y returns to baseline
        landing_idx = self._find_landing(hip_y, peak_idx, takeoff_idx)

        # Set boundaries
        start_idx = max(0, takeoff_idx - 10)
        end_idx = min(len(poses) - 1, landing_idx + 10)

        phases = ElementPhase(
            name="jump",
            start=start_idx,
            takeoff=takeoff_idx,
            peak=peak_idx,
            landing=landing_idx,
            end=end_idx,
        )

        # Confidence based on peak prominence
        prominence = float(properties["prominences"][np.argmax(-properties["prominences"])])
        confidence = min(1.0, prominence / 0.1)  # Normalize

        return PhaseDetectionResult(phases=phases, confidence=confidence)

    def detect_three_turn_phases(
        self,
        poses: NormalizedPose,
        fps: float,  # noqa: ARG002
    ) -> PhaseDetectionResult:
        """Detect three-turn phases by edge change.

        Args:
            poses: NormalizedPose (num_frames, 33, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with turn phase boundaries.
        """
        # Compute edge indicator
        dummy_def = ElementDef(
            name="three_turn",
            name_ru="тройка",
            rotations=0,
            has_toe_pick=False,
            key_joints=[],
            ideal_metrics={},
        )

        analyzer = BiomechanicsAnalyzer(dummy_def)
        edge_ind = analyzer.compute_edge_indicator(poses, side="left")

        # Find edge change point (zero crossing)
        # Compute derivative to find rapid changes
        edge_derivative = np.gradient(edge_ind)

        # Find peak in derivative (maximum rate of change)
        change_points, _ = find_peaks(np.abs(edge_derivative), prominence=0.1, distance=5)

        if len(change_points) == 0:
            # No clear turn detected
            return PhaseDetectionResult(
                phases=ElementPhase(
                    name="three_turn",
                    start=0,
                    takeoff=0,
                    peak=0,
                    landing=0,
                    end=len(poses) - 1,
                ),
                confidence=0.0,
            )

        # Use most prominent change point as turn center
        turn_center = change_points[np.argmax(np.abs(edge_derivative[change_points]))]

        # Set boundaries around turn
        start_idx = max(0, turn_center - 15)
        end_idx = min(len(poses) - 1, turn_center + 15)

        phases = ElementPhase(
            name="three_turn",
            start=start_idx,
            takeoff=0,  # No takeoff for steps
            peak=turn_center,  # Use peak as turn center
            landing=0,  # No landing for steps
            end=end_idx,
        )

        # Confidence based on edge change magnitude
        max_change = float(np.max(np.abs(edge_derivative)))
        confidence = min(1.0, max_change / 0.5)

        return PhaseDetectionResult(phases=phases, confidence=confidence)

    def _find_takeoff(self, derivative: np.ndarray, peak_idx: int) -> int:
        """Find takeoff frame before peak.

        Args:
            derivative: Hip Y derivative.
            peak_idx: Peak frame index.

        Returns:
            Takeoff frame index.
        """
        # Look backward from peak for sustained negative derivative
        search_start = max(0, peak_idx - 30)
        search_end = peak_idx

        # Find where derivative becomes consistently negative
        for i in range(search_end, search_start, -1):
            # Check if derivative is negative and significant
            if derivative[i] < -0.01:
                # Check if sustained for next few frames
                window = min(5, search_end - i)
                if np.all(derivative[i : i + window] < 0):
                    return i

        # Fallback: fixed frames before peak
        return max(0, peak_idx - 10)

    def _find_landing(
        self,
        hip_y: np.ndarray,
        peak_idx: int,
        takeoff_idx: int,
    ) -> int:
        """Find landing frame after peak.

        Args:
            hip_y: Hip Y coordinates.
            peak_idx: Peak frame index.
            takeoff_idx: Takeoff frame index.

        Returns:
            Landing frame index.
        """
        # Get hip Y at takeoff (baseline)
        baseline_y = hip_y[takeoff_idx]

        # Look forward from peak for return to baseline
        search_start = peak_idx
        search_end = min(len(hip_y), peak_idx + 30)

        for i in range(search_start, search_end):
            # Check if hip Y has returned to near baseline
            if abs(hip_y[i] - baseline_y) < 0.02:
                # Verify it stays near baseline
                window = min(5, search_end - i)
                if np.all(np.abs(hip_y[i : i + window] - baseline_y) < 0.03):
                    return i

        # Fallback: fixed frames after peak
        return min(len(hip_y) - 1, peak_idx + 10)
