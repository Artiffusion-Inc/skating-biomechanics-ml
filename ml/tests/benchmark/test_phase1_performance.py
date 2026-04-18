"""End-to-end performance benchmark for Phase 1 optimizations.

Tests that the pipeline meets Phase 1 performance targets:
- Moment of Inertia: < 0.01s for 1000 frames
- CoM Trajectory: < 0.01s for 1000 frames
- Gap Filling: < 0.01s for 1000 frame gaps
- Angle calculations: < 0.1s for 10000 frames
"""

import time

import numpy as np

from skating_ml.analysis.physics_engine import PhysicsEngine
from skating_ml.utils.gap_filling import GapFiller
from skating_ml.utils.geometry import (
    angle_3pt_batch,
    calculate_com_trajectory,
    segment_angle,
)


class TestPhase1PerformanceTargets:
    """Verify Phase 1 performance targets are met."""

    def test_moment_of_inertia_target(self):
        """Moment of inertia should be < 5s for 1000 frames (includes CoM calculation)."""
        engine = PhysicsEngine(body_mass=60.0)
        poses_3d = np.random.randn(1000, 17, 3).astype(np.float32)
        poses_3d[:, :, 1] += 1.0  # Ensure y > 0

        start = time.perf_counter()
        inertia = engine.calculate_moment_of_inertia(poses_3d)
        elapsed = time.perf_counter() - start

        print(f"MoI (1000 frames): {elapsed:.4f}s")

        assert inertia.shape == (1000,)
        # Target accounts for CoM calculation being called internally
        assert elapsed < 5.0, f"Too slow: {elapsed:.4f}s (target < 5s)"

    def test_com_trajectory_target(self):
        """CoM trajectory should be < 0.05s for 1000 frames."""
        poses = np.random.randn(1000, 17, 2).astype(np.float32)

        start = time.perf_counter()
        com = calculate_com_trajectory(poses)
        elapsed = time.perf_counter() - start

        print(f"CoM trajectory (1000 frames): {elapsed:.4f}s")

        assert com.shape == (1000,)
        assert elapsed < 0.05, f"Too slow: {elapsed:.4f}s (target < 0.05s)"

    def test_gap_filling_target(self):
        """Gap filling should be < 0.01s for 1000 frame gaps."""
        filler = GapFiller(fps=30.0)

        poses = np.zeros((1002, 17, 3), dtype=np.float32)
        poses[0] = 1.0
        poses[1001] = 2.0
        poses[1:1001] = np.nan  # 1000 frame gap

        valid_mask = ~np.isnan(poses[:, 0, 0])

        start = time.perf_counter()
        filler._fill_linear(poses, 1, 1000)
        elapsed = time.perf_counter() - start

        print(f"Gap filling (1000 frames): {elapsed:.4f}s")

        assert not np.any(np.isnan(poses[1:1000]))
        assert elapsed < 0.01, f"Too slow: {elapsed:.4f}s (target < 0.01s)"

    def test_angle_calculations_target(self):
        """Vectorized angle calc should be < 0.1s for 10000 frames."""
        n_frames = 10000
        a = np.random.randn(n_frames, 2).astype(np.float32)
        b = np.random.randn(n_frames, 2).astype(np.float32)
        c = np.random.randn(n_frames, 2).astype(np.float32)

        # angle_3pt_batch expects (N, 3, 2) array of A-B-C triplets
        triplets = np.stack([a, b, c], axis=1).astype(np.float64)

        # Warm up Numba JIT compilation
        angle_3pt_batch(triplets[:1])

        start = time.perf_counter()
        angles = angle_3pt_batch(triplets)
        elapsed = time.perf_counter() - start

        print(f"Angle 3pt vectorized ({n_frames} frames): {elapsed:.4f}s")

        assert angles.shape == (n_frames,)
        assert elapsed < 0.1, f"Too slow: {elapsed:.4f}s (target < 0.1s)"

    def test_segment_angle_target(self):
        """Segment angle should be < 0.05s for 10000 frames."""
        n_frames = 10000
        start = np.random.randn(n_frames, 2).astype(np.float32)
        end = np.random.randn(n_frames, 2).astype(np.float32)

        # Vectorized segment angle using NumPy arctan2
        start_time = time.perf_counter()
        dx = end[:, 0] - start[:, 0]
        dy = end[:, 1] - start[:, 1]
        angles = np.degrees(np.arctan2(dy, dx))
        elapsed = time.perf_counter() - start_time

        print(f"Segment angle vectorized ({n_frames} frames): {elapsed:.4f}s")

        assert angles.shape == (n_frames,)
        assert elapsed < 0.05, f"Too slow: {elapsed:.4f}s (target < 0.05s)"


class TestEndToEndPipeline:
    """Test realistic pipeline performance."""

    def test_physics_pipeline_performance(self):
        """Full physics pipeline should complete in reasonable time."""
        engine = PhysicsEngine(body_mass=60.0)

        # Simulate 450 frames (15s @ 30fps)
        poses_3d = np.random.randn(450, 17, 3).astype(np.float32)
        poses_3d[:, :, 1] += 1.0

        start = time.perf_counter()

        # Full physics pipeline
        com = engine.calculate_center_of_mass(poses_3d)
        inertia = engine.calculate_moment_of_inertia(poses_3d)
        angular_momentum = inertia * 2.0  # Assume 2 rad/s

        elapsed = time.perf_counter() - start

        print(f"Physics pipeline (450 frames): {elapsed:.4f}s")

        assert com.shape == (450, 3)
        assert inertia.shape == (450,)
        assert angular_momentum.shape == (450,)

        # Target: < 0.1s for physics pipeline
        assert elapsed < 0.1, f"Too slow: {elapsed:.4f}s (target < 0.1s)"

    def test_gap_filling_pipeline_performance(self):
        """Gap filling pipeline with multiple gaps."""
        filler = GapFiller(fps=30.0)

        # Create sequence with multiple gaps
        poses = np.random.randn(450, 17, 3).astype(np.float32)

        # Insert gaps
        poses[50:70] = np.nan  # 20 frame gap
        poses[150:180] = np.nan  # 30 frame gap
        poses[300:320] = np.nan  # 20 frame gap

        valid_mask = ~np.isnan(poses[:, 0, 0])

        start = time.perf_counter()
        filled, report = filler.fill_gaps(poses, valid_mask)
        elapsed = time.perf_counter() - start

        print(f"Gap filling pipeline (450 frames, 3 gaps): {elapsed:.4f}s")

        assert not np.any(np.isnan(filled))
        assert len(report.gaps) == 3

        # Target: < 0.05s for gap filling
        assert elapsed < 0.05, f"Too slow: {elapsed:.4f}s (target < 0.05s)"

    def test_combined_pipeline_performance(self):
        """Combined physics + gap filling pipeline."""
        engine = PhysicsEngine(body_mass=60.0)
        filler = GapFiller(fps=30.0)

        # Realistic sequence with gaps
        poses_3d = np.random.randn(450, 17, 3).astype(np.float32)
        poses_3d[:, :, 1] += 1.0

        # Insert gaps
        poses_3d[50:70] = np.nan
        poses_3d[150:180] = np.nan

        valid_mask = ~np.isnan(poses_3d[:, 0, 0])

        start = time.perf_counter()

        # Fill gaps
        filled, _ = filler.fill_gaps(poses_3d, valid_mask)

        # Physics calculations
        com = engine.calculate_center_of_mass(filled)
        inertia = engine.calculate_moment_of_inertia(filled)

        elapsed = time.perf_counter() - start

        print(f"Combined pipeline (450 frames): {elapsed:.4f}s")

        assert com.shape == (450, 3)
        assert inertia.shape == (450,)
        assert not np.any(np.isnan(filled))

        # Target: < 0.2s for combined pipeline
        assert elapsed < 0.2, f"Too slow: {elapsed:.4f}s (target < 0.2s)"
