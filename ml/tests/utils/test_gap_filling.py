"""Tests for gap filling utility."""

import numpy as np

from src.utils.gap_filling import GapFiller


class TestFillLinear:
    """Tests for linear gap filling."""

    def test_fill_linear_simple_gap(self):
        """Should fill simple gap with linear interpolation."""
        poses = np.zeros((10, 17, 3), dtype=np.float32)
        # Valid frames at start and end
        poses[0] = 1.0
        poses[9] = 2.0
        # Gap in middle
        poses[1:9] = np.nan

        GapFiller._fill_linear(poses, 1, 8)

        # Check interpolation
        assert not np.any(np.isnan(poses[1:9]))
        # Should interpolate from 1.0 to 2.0
        assert np.allclose(poses[0], 1.0)
        assert np.allclose(poses[9], 2.0)
        # Middle should be ~1.5 (actually ~1.44 with linspace weights)
        assert 1.3 < poses[4, 0, 0] < 1.6

    def test_fill_linear_edge_gap(self):
        """Should handle gaps at array edges."""
        poses = np.zeros((10, 17, 3), dtype=np.float32)
        poses[5] = 1.0  # Only valid frame
        poses[:5] = np.nan
        poses[6:] = np.nan

        # Fill left edge
        GapFiller._fill_linear(poses, 0, 4)
        assert not np.any(np.isnan(poses[:5]))
        # Should copy nearest valid frame (except confidence)
        assert np.allclose(poses[0:5, :, 0], poses[5, 0, 0])  # X channel
        # Confidence should be 0 for filled frames
        assert np.all(poses[0:5, :, 2] == 0.0)
        assert poses[5, 0, 2] == 1.0  # Original keeps confidence

    def test_fill_linear_confidence_zeroed(self):
        """Should set confidence to 0 for filled frames."""
        poses = np.zeros((10, 17, 3), dtype=np.float32)
        poses[0, :, 2] = 1.0  # High confidence
        poses[9, :, 2] = 1.0  # High confidence
        poses[1:9] = np.nan

        GapFiller._fill_linear(poses, 1, 8)

        # Confidence should be 0 for interpolated frames
        assert np.all(poses[1:9, :, 2] == 0.0)
        # Original frames should keep confidence
        assert poses[0, 0, 2] == 1.0
        assert poses[9, 0, 2] == 1.0


class TestFillExtrapolation:
    """Tests for velocity-based extrapolation."""

    def test_fill_extrapolation_with_velocity(self):
        """Should extrapolate using velocity from recent frames."""
        poses = np.zeros((20, 17, 3), dtype=np.float32)
        # Create motion: constant velocity to the right
        for i in range(10):
            poses[i, 0, 0] = float(i)  # X increases by 1 each frame

        poses[10:15] = np.nan  # Gap

        valid_mask = ~np.isnan(poses[:, 0, 0])
        GapFiller._fill_extrapolation(poses, 10, 14, seg_offset=0, seg_valid_mask=valid_mask)

        # Should extrapolate with velocity ~1.0
        assert not np.any(np.isnan(poses[10:15]))
        # X should continue increasing
        assert poses[10, 0, 0] > poses[9, 0, 0]
        assert poses[14, 0, 0] > poses[10, 0, 0]

    def test_fill_extrapolation_fallback_to_linear(self):
        """Should fall back to linear if insufficient history."""
        poses = np.zeros((10, 17, 3), dtype=np.float32)
        poses[0] = 1.0  # Only one valid frame before gap
        poses[1:5] = np.nan
        poses[5] = 2.0  # Valid frame after gap

        valid_mask = ~np.isnan(poses[:, 0, 0])
        # This should fall back to linear interpolation
        GapFiller._fill_extrapolation(poses, 1, 4, seg_offset=0, seg_valid_mask=valid_mask)

        # Should be filled
        assert not np.any(np.isnan(poses[1:5]))


class TestGapFiller:
    """Tests for GapFiller class."""

    def test_fill_gaps_short_gap(self):
        """Should fill short gaps with linear interpolation."""
        filler = GapFiller(fps=30.0, short_gap_threshold=10)

        poses = np.zeros((20, 17, 3), dtype=np.float32)
        poses[0] = 1.0
        poses[10] = 2.0
        poses[1:10] = np.nan  # 9 frame gap (short)

        valid_mask = ~np.isnan(poses[:, 0, 0])
        filled, report = filler.fill_gaps(poses, valid_mask)

        assert len(report.gaps) == 1
        assert report.strategy_used[0] == "linear"
        assert not np.any(np.isnan(filled))

    def test_fill_gaps_medium_gap(self):
        """Should fill medium gaps with extrapolation."""
        filler = GapFiller(fps=30.0, short_gap_threshold=10, medium_gap_threshold=30)

        poses = np.zeros((50, 17, 3), dtype=np.float32)
        # Create motion for velocity estimation
        for i in range(10):
            poses[i, 0, 0] = float(i)
        poses[10:25] = np.nan  # 15 frame gap (medium)

        valid_mask = ~np.isnan(poses[:, 0, 0])
        filled, report = filler.fill_gaps(poses, valid_mask)

        assert len(report.gaps) == 1
        assert report.strategy_used[0] == "extrapolation"
        assert not np.any(np.isnan(filled[10:25]))

    def test_fill_gaps_phase_aware(self):
        """Should respect phase boundaries when filling."""
        filler = GapFiller(fps=30.0)

        poses = np.zeros((30, 17, 3), dtype=np.float32)
        poses[0] = 1.0
        poses[10:20] = np.nan  # Gap crosses boundary
        poses[20] = 2.0

        valid_mask = ~np.isnan(poses[:, 0, 0])
        # Boundary at frame 15 - gap should be split
        filled, report = filler.fill_gaps(poses, valid_mask, phase_boundaries=[15])

        # Should create two sub-gaps
        assert len(report.gaps) == 2
        assert not np.any(np.isnan(filled))

    def test_fill_gaps_no_gaps(self):
        """Should return early when no gaps exist."""
        filler = GapFiller(fps=30.0)

        poses = np.ones((10, 17, 3), dtype=np.float32)
        valid_mask = ~np.isnan(poses[:, 0, 0])

        filled, report = filler.fill_gaps(poses, valid_mask)

        assert len(report.gaps) == 0
        np.testing.assert_array_equal(filled, poses)


class TestPerformance:
    """Performance tests for vectorized gap filling."""

    def test_fill_linear_performance(self):
        """Vectorized linear fill should be fast."""
        import time

        # Large gap: 1000 frames
        poses = np.zeros((1002, 17, 3), dtype=np.float32)
        poses[0] = 1.0
        poses[1001] = 2.0
        poses[1:1001] = np.nan

        start = time.perf_counter()
        GapFiller._fill_linear(poses, 1, 1000)
        elapsed = time.perf_counter() - start

        print(f"Linear fill 1000 frames: {elapsed:.4f}s")

        # Should be very fast with vectorization (< 0.01s)
        assert elapsed < 0.01, f"Too slow: {elapsed:.4f}s"

    def test_fill_extrapolation_performance(self):
        """Vectorized extrapolation should be fast."""
        import time

        poses = np.zeros((1020, 17, 3), dtype=np.float32)
        # Create motion history
        for i in range(10):
            poses[i, 0, 0] = float(i)
        poses[10:1010] = np.nan  # 1000 frame gap

        valid_mask = ~np.isnan(poses[:, 0, 0])

        start = time.perf_counter()
        GapFiller._fill_extrapolation(poses, 10, 1009, seg_offset=0, seg_valid_mask=valid_mask)
        elapsed = time.perf_counter() - start

        print(f"Extrapolation fill 1000 frames: {elapsed:.4f}s")

        # Should be very fast with vectorization (< 0.01s)
        assert elapsed < 0.01, f"Too slow: {elapsed:.4f}s"
