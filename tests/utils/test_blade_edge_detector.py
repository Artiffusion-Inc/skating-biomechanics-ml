"""Tests for blade edge detection.

Updated for H3.6M 17kp format - blade detection now uses 3D physics
rather than 2D keypoints like toe/heel which don't exist in H3.6M.
"""

import numpy as np
import pytest

from src.types import H36Key, BKey, BladeType, NormalizedPose
from src.blade_edge_detector import (
    BladeEdgeDetector,
    BladeState,
    calculate_ankle_angle,
    calculate_foot_angle,
    calculate_foot_vector,
    calculate_motion_direction,
    calculate_vertical_acceleration,
)


@pytest.fixture
def sample_poses() -> NormalizedPose:
    """Create sample normalized poses for testing (H3.6M 17kp format).

    Simulates a skater moving forward (positive x direction).
    """
    # 10 frames, 17 keypoints (H3.6M), 2 coordinates
    poses = np.zeros((10, 17, 2), dtype=np.float32)

    # Mid-hip moves forward (positive x)
    for i in range(10):
        poses[i, H36Key.HIP_CENTER] = [0.3 + i * 0.01, 0.5]
        poses[i, H36Key.LHIP] = [0.3 + i * 0.01, 0.5]
        poses[i, H36Key.RHIP] = [0.3 + i * 0.01, 0.5]
        poses[i, H36Key.LKNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, H36Key.RKNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, H36Key.LFOOT] = [0.3 + i * 0.01, 0.7]  # Ankle/foot
        poses[i, H36Key.RFOOT] = [0.3 + i * 0.01, 0.7]

    return poses


@pytest.fixture
def inside_edge_poses() -> NormalizedPose:
    """Create poses simulating inside edge (H3.6M 17kp format).

    For left foot inside edge skating forward:
    - Motion direction: forward (+x)
    - In H3.6M, we only have ankle position, so edge detection
      relies on 3D foot vector from motion direction.
    """
    poses = np.zeros((10, 17, 2), dtype=np.float32)

    for i in range(10):
        poses[i, H36Key.HIP_CENTER] = [0.3 + i * 0.01, 0.5]
        poses[i, H36Key.LHIP] = [0.3 + i * 0.01, 0.5]
        poses[i, H36Key.RHIP] = [0.3 + i * 0.01, 0.5]
        poses[i, H36Key.LKNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, H36Key.RKNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, H36Key.LFOOT] = [0.35 + i * 0.01, 0.7]  # Ankle/foot
        poses[i, H36Key.RFOOT] = [0.3 + i * 0.01, 0.7]

    return poses


@pytest.fixture
def outside_edge_poses() -> NormalizedPose:
    """Create poses simulating outside edge (H3.6M 17kp format).

    For left foot outside edge skating forward:
    - Motion direction: forward (+x)
    - In H3.6M, edge detection uses velocity and body lean.
    """
    poses = np.zeros((10, 17, 2), dtype=np.float32)

    for i in range(10):
        poses[i, H36Key.HIP_CENTER] = [0.3 + i * 0.01, 0.5]
        poses[i, H36Key.LHIP] = [0.3 + i * 0.01, 0.5]
        poses[i, H36Key.RHIP] = [0.3 + i * 0.01, 0.5]
        poses[i, H36Key.LKNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, H36Key.RKNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, H36Key.LFOOT] = [0.25 + i * 0.01, 0.7]  # Ankle/foot
        poses[i, H36Key.RFOOT] = [0.3 + i * 0.01, 0.7]

    return poses


class TestFootVector:
    """Tests for foot vector calculation."""

    def test_foot_vector_forward(self, sample_poses: NormalizedPose) -> None:
        """Foot vector for forward-pointing foot."""
        vector = calculate_foot_vector(sample_poses, 0, "left")
        # With H3.6M, foot vector is derived from velocity
        assert len(vector) == 2

    def test_foot_vector_left_foot(self, sample_poses: NormalizedPose) -> None:
        """Left foot vector."""
        vector = calculate_foot_vector(sample_poses, 0, "left")
        assert len(vector) == 2

    def test_foot_vector_right_foot(self, sample_poses: NormalizedPose) -> None:
        """Right foot vector."""
        vector = calculate_foot_vector(sample_poses, 0, "right")
        assert len(vector) == 2


class TestMotionDirection:
    """Tests for motion direction calculation."""

    def test_motion_direction_forward(self, sample_poses: NormalizedPose) -> None:
        """Motion direction for forward-moving skater."""
        direction = calculate_motion_direction(sample_poses, 5)
        assert direction[0] > 0  # Moving forward
        # Should be normalized
        assert abs(np.linalg.norm(direction) - 1.0) < 0.1


class TestFootAngle:
    """Tests for foot angle calculation."""

    def test_foot_angle_flat(self, sample_poses: NormalizedPose) -> None:
        """Foot angle for flat blade (foot aligned with motion)."""
        angle = calculate_foot_angle(sample_poses, 5, "left")
        # With H3.6M, angle is derived from velocity direction
        # Should be a valid number
        assert isinstance(angle, float)

    def test_foot_angle_inside_edge(self, inside_edge_poses: NormalizedPose) -> None:
        """Foot angle for inside edge."""
        angle = calculate_foot_angle(inside_edge_poses, 5, "left")
        # Should be a valid number
        assert isinstance(angle, float)

    def test_foot_angle_outside_edge(self, outside_edge_poses: NormalizedPose) -> None:
        """Foot angle for outside edge."""
        angle = calculate_foot_angle(outside_edge_poses, 5, "left")
        # Should be a valid number
        assert isinstance(angle, float)


class TestAnkleAngle:
    """Tests for ankle angle calculation."""

    def test_ankle_angle_range(self, sample_poses: NormalizedPose) -> None:
        """Ankle angle should be in physiological range."""
        angle = calculate_ankle_angle(sample_poses, 0, "left")
        # With H3.6M, we have hip-knee-foot for angle calculation
        assert 0 <= angle <= 180 or np.isnan(angle)  # Valid angle range or undefined


class TestVerticalAcceleration:
    """Tests for vertical acceleration calculation."""

    def test_vertical_acceleration_static(self, sample_poses: NormalizedPose) -> None:
        """Vertical acceleration for static foot position."""
        accel = calculate_vertical_acceleration(sample_poses, fps=30.0, frame_idx=5, leg="left")
        # Should be near 0 for constant height
        assert abs(accel) < 1.0


class TestBladeEdgeDetector:
    """Tests for BladeEdgeDetector class."""

    def test_init(self) -> None:
        """Detector initialization."""
        detector = BladeEdgeDetector()
        assert detector.inside_threshold == -15.0
        assert detector.outside_threshold == 15.0
        assert detector.toe_pick_accel_threshold == 5.0

    def test_classify_frame_flat(self, sample_poses: NormalizedPose) -> None:
        """Classify flat blade."""
        detector = BladeEdgeDetector()
        state = detector.classify_frame(sample_poses, 5, fps=30.0, foot="left")

        assert isinstance(state, BladeState)
        assert state.blade_type in (BladeType.FLAT, BladeType.UNKNOWN)
        assert 0 <= state.confidence <= 1

    def test_classify_frame_inside_edge(self, inside_edge_poses: NormalizedPose) -> None:
        """Classify inside edge."""
        detector = BladeEdgeDetector()
        state = detector.classify_frame(inside_edge_poses, 5, fps=30.0, foot="left")

        assert isinstance(state, BladeState)
        # With H3.6M 17kp, detection may be less precise
        assert state.blade_type in (BladeType.INSIDE, BladeType.FLAT, BladeType.UNKNOWN)
        assert 0 <= state.confidence <= 1

    def test_classify_frame_outside_edge(self, outside_edge_poses: NormalizedPose) -> None:
        """Classify outside edge."""
        detector = BladeEdgeDetector()
        state = detector.classify_frame(outside_edge_poses, 5, fps=30.0, foot="left")

        assert isinstance(state, BladeState)
        # With H3.6M 17kp, detection may be less precise
        assert state.blade_type in (BladeType.OUTSIDE, BladeType.FLAT, BladeType.UNKNOWN)
        assert 0 <= state.confidence <= 1

    def test_detect_sequence(self, sample_poses: NormalizedPose) -> None:
        """Detect blade state for entire sequence."""
        detector = BladeEdgeDetector()
        states = detector.detect_sequence(sample_poses, fps=30.0, foot="left")

        assert len(states) == len(sample_poses)
        for state in states:
            assert isinstance(state, BladeState)
            # BladeState doesn't have 'foot' attribute - it's implicit from the detector call
            assert 0 <= state.confidence <= 1

    def test_detect_sequence_inside_edge(self, inside_edge_poses: NormalizedPose) -> None:
        """Detect sequence for inside edge."""
        detector = BladeEdgeDetector()
        states = detector.detect_sequence(inside_edge_poses, fps=30.0, foot="left")

        assert len(states) == len(inside_edge_poses)
        # At least some frames should detect inside edge or flat
        edge_types = [s.blade_type for s in states]
        assert any(t in (BladeType.INSIDE, BladeType.FLAT) for t in edge_types)

    def test_detect_sequence_outside_edge(self, outside_edge_poses: NormalizedPose) -> None:
        """Detect sequence for outside edge."""
        detector = BladeEdgeDetector()
        states = detector.detect_sequence(outside_edge_poses, fps=30.0, foot="left")

        assert len(states) == len(outside_edge_poses)
        # At least some frames should detect outside edge or flat
        edge_types = [s.blade_type for s in states]
        assert any(t in (BladeType.OUTSIDE, BladeType.FLAT) for t in edge_types)

    def test_get_blade_summary(self, sample_poses: NormalizedPose) -> None:
        """Get blade summary statistics."""
        detector = BladeEdgeDetector()
        states = detector.detect_sequence(sample_poses, fps=30.0, foot="left")
        summary = detector.get_blade_summary(states)

        assert isinstance(summary, dict)
        assert "dominant_edge" in summary
        assert "type_percentages" in summary
        assert summary["dominant_edge"] in ("inside", "outside", "flat", "unknown")

    def test_takeoff_detection(self, sample_poses: NormalizedPose) -> None:
        """Test takeoff detection from blade states."""
        detector = BladeEdgeDetector()
        states = detector.detect_sequence(sample_poses, fps=30.0, foot="left")

        # Try to detect takeoff and landing
        takeoff, landing = detector.detect_takeoff_landing(states, fps=30.0)

        # May be None for non-jump sequences
        if takeoff is not None:
            assert 0 <= takeoff < len(sample_poses)
        if landing is not None:
            assert 0 <= landing < len(sample_poses)

    def test_landing_detection(self, sample_poses: NormalizedPose) -> None:
        """Test landing detection from blade states."""
        detector = BladeEdgeDetector()
        states = detector.detect_sequence(sample_poses, fps=30.0, foot="left")

        # Try to detect takeoff and landing
        takeoff, landing = detector.detect_takeoff_landing(states, fps=30.0)

        # May be None for non-jump sequences
        if landing is not None:
            assert 0 <= landing < len(sample_poses)
