"""Tests for multi-person pose tracker."""

import numpy as np
import pytest

from src.detection.pose_tracker import PoseTracker


@pytest.fixture
def tracker():
    """Create a default tracker instance."""
    return PoseTracker(max_disappeared=5, min_hits=2, fps=30.0)


@pytest.fixture
def sample_poses():
    """Create sample poses for testing."""
    # Two different people with different proportions
    poses = np.zeros((2, 33, 2), dtype=np.float32)

    # Person 1: Taller, broader shoulders
    poses[0, 11] = [0.3, 0.2]  # Left shoulder
    poses[0, 12] = [0.7, 0.2]  # Right shoulder
    poses[0, 15] = [0.25, 0.4]  # Left wrist
    poses[0, 16] = [0.75, 0.4]  # Right wrist
    poses[0, 23] = [0.35, 0.5]  # Left hip
    poses[0, 24] = [0.65, 0.5]  # Right hip
    poses[0, 25] = [0.35, 0.7]  # Left knee
    poses[0, 27] = [0.35, 0.9]  # Left ankle

    # Person 2: Shorter, narrower shoulders
    poses[1, 11] = [0.4, 0.25]  # Left shoulder
    poses[1, 12] = [0.6, 0.25]  # Right shoulder
    poses[1, 15] = [0.38, 0.45]  # Left wrist
    poses[1, 16] = [0.62, 0.45]  # Right wrist
    poses[1, 23] = [0.42, 0.5]  # Left hip
    poses[1, 24] = [0.58, 0.5]  # Right hip
    poses[1, 25] = [0.42, 0.65]  # Left knee
    poses[1, 27] = [0.42, 0.8]  # Left ankle

    return poses


class TestPoseTracker:
    """Test PoseTracker functionality."""

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.max_disappeared == 5
        assert tracker.min_hits == 2
        assert tracker.fps == 30.0
        assert tracker.dt == pytest.approx(1.0 / 30.0)
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 0

    def test_empty_update(self, tracker):
        """Test update with no detections."""
        track_ids = tracker.update(np.array([]))
        assert track_ids == []
        assert len(tracker.tracks) == 0

    def test_single_detection_creates_track(self, tracker, sample_poses):
        """Test that a single detection creates a new track."""
        single_pose = sample_poses[0:1]
        track_ids = tracker.update(single_pose)

        assert len(track_ids) == 1
        assert track_ids[0] == 0
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].id == 0
        assert tracker.tracks[0].hits == 1

    def test_multiple_detections_create_multiple_tracks(self, tracker, sample_poses):
        """Test that multiple detections create multiple tracks."""
        track_ids = tracker.update(sample_poses)

        assert len(track_ids) == 2
        assert len(set(track_ids)) == 2  # Unique IDs
        assert len(tracker.tracks) == 2

    def test_track_persistence_across_frames(self, tracker, sample_poses):
        """Test that tracks persist across frames."""
        # Frame 1: Initial detection
        track_ids_1 = tracker.update(sample_poses)

        # Frame 2: Same poses (slightly moved)
        poses_moved = sample_poses.copy()
        poses_moved[:, :2] += 0.01  # Move slightly
        track_ids_2 = tracker.update(poses_moved)

        # Same IDs should be assigned
        assert sorted(track_ids_1) == sorted(track_ids_2)

    def test_biometric_extraction(self, tracker, sample_poses):
        """Test biometric extraction for re-identification."""
        biometrics = tracker._extract_biometrics(sample_poses[0])

        # Check that all expected ratios are present
        expected_keys = [
            "shoulder_width/torso",
            "femur/tibia",
            "arm_span/height",
            "torso/height",
            "shoulder_width/height",
        ]
        for key in expected_keys:
            assert key in biometrics
            assert biometrics[key] > 0

    def test_biometric_distance_same_person(self, tracker, sample_poses):
        """Test biometric distance for same person (different poses)."""
        bio1 = tracker._extract_biometrics(sample_poses[0])
        bio2 = tracker._extract_biometrics(sample_poses[0])  # Same pose

        distance = tracker._biometric_distance(bio1, bio2)
        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_biometric_distance_different_people(self, tracker, sample_poses):
        """Test biometric distance for different people."""
        bio1 = tracker._extract_biometrics(sample_poses[0])
        bio2 = tracker._extract_biometrics(sample_poses[1])

        distance = tracker._biometric_distance(bio1, bio2)
        # Different proportions should give non-zero distance
        assert distance > 0

    def test_confirmed_tracks_filter(self, tracker, sample_poses):
        """Test getting confirmed tracks (above min_hits threshold)."""
        tracker.update(sample_poses)

        # With min_hits=2, first update should not confirm
        confirmed = tracker.get_confirmed_tracks()
        assert len(confirmed) == 0

        # Second update should confirm
        tracker.update(sample_poses)
        confirmed = tracker.get_confirmed_tracks()
        assert len(confirmed) == 2

    def test_remove_lost_tracks(self, tracker):
        """Test that tracks are removed after max_disappeared frames."""
        # Create a track
        pose = np.zeros((1, 33, 2), dtype=np.float32)
        pose[0, 11] = [0.5, 0.3]
        pose[0, 12] = [0.5, 0.3]
        pose[0, 23] = [0.5, 0.6]
        pose[0, 24] = [0.5, 0.6]

        tracker.update(pose)
        assert len(tracker.tracks) == 1

        # Empty updates (no detections)
        for _ in range(tracker.max_disappeared):
            tracker.update(np.array([]))

        # Track should still exist (at max_disappeared)
        assert len(tracker.tracks) == 1

        # One more empty update should remove it
        tracker.update(np.array([]))
        assert len(tracker.tracks) == 0

    def test_mid_hip_calculation(self, tracker, sample_poses):
        """Test mid-hip position calculation."""
        mid_hips = tracker._get_mid_hips(sample_poses)

        assert mid_hips.shape == (2, 2)

        # Mid-hip should be average of left and right hip
        for i in range(2):
            expected = (sample_poses[i, 23] + sample_poses[i, 24]) / 2
            np.testing.assert_array_almost_equal(mid_hips[i], expected)

    def test_kalman_filter_state_transition(self, tracker):
        """Test Kalman filter state transition matrix."""
        # State: [x, y, vx, vy, ax, ay]
        # x(t+dt) = x(t) + vx(t)*dt + 0.5*ax(t)*dt^2

        dt = tracker.dt
        F = tracker.kf.F

        # Check position update row
        assert F[0, 0] == 1  # x
        assert F[0, 2] == pytest.approx(dt)  # vx contribution
        assert F[0, 4] == pytest.approx(0.5 * dt**2)  # ax contribution

    def test_track_state_initialization(self, tracker, sample_poses):
        """Test that track state is initialized correctly."""
        tracker.update(sample_poses)

        for track in tracker.tracks:
            assert track.state is not None
            assert track.state.shape == (6, 1)  # Column vector
            # Position should be non-zero (velocity and acceleration start at 0)
            assert track.state[0, 0] != 0 or track.state[1, 0] != 0
            assert track.state[2, 0] == 0  # vx starts at 0
            assert track.state[3, 0] == 0  # vy starts at 0
            assert track.state[4, 0] == 0  # ax starts at 0
            assert track.state[5, 0] == 0  # ay starts at 0

    def test_association_with_cost_matrix(self, tracker, sample_poses):
        """Test detection-to-track association."""
        # Create tracks
        tracker.update(sample_poses)

        # Get mid-hip positions
        detections = tracker._get_mid_hips(sample_poses)
        predictions = np.array([track.state[:2] for track in tracker.tracks])

        # Associate
        matched, unmatched_dets, unmatched_trks = tracker._associate(
            sample_poses, detections, predictions
        )

        # With 2 tracks and 2 detections, should match both
        assert len(matched) == 2
        assert len(unmatched_dets) == 0
        assert len(unmatched_trks) == 0
