"""Tests for biomechanics metrics computation."""

import numpy as np
import pytest

from skating_biomechanics_ml.analysis.metrics import BiomechanicsAnalyzer
from skating_biomechanics_ml.references.element_defs import ElementDef, get_element_def
from skating_biomechanics_ml.types import BKey, ElementPhase, MetricResult


class TestBiomechanicsAnalyzer:
    """Test BiomechanicsAnalyzer."""

    def test_analyzer_initialization(self):
        """Should initialize with element definition."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        assert analyzer._element_def == element_def

    def test_compute_airtime(self):
        """Should compute airtime correctly."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=30,
            peak=45,
            landing=60,
            end=90,
        )

        airtime = analyzer.compute_airtime(phases, fps=30.0)

        assert airtime == pytest.approx(1.0)  # 30 frames / 30 fps

    def test_compute_angle_series(self, sample_normalized_poses):
        """Should compute angle series correctly."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)

        angles = analyzer.compute_angle_series(
            sample_normalized_poses,
            BKey.LEFT_SHOULDER,
            BKey.LEFT_ELBOW,
            BKey.LEFT_WRIST,
        )

        assert len(angles) == 3
        assert all(0 <= a <= 180 for a in angles)

    def test_compute_angular_velocity(self):
        """Should compute angular velocity."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create simple angle series: 0, 10, 20, 30 degrees
        angles = np.array([0, 10, 20, 30], dtype=np.float32)

        velocity = analyzer.compute_angular_velocity(angles, fps=10.0)

        assert len(velocity) == 4
        # Velocity should be ~100 deg/s (10 deg per frame at 10 fps)
        assert velocity[1] == pytest.approx(100, abs=1)

    def test_compute_jump_height(self):
        """Should compute jump height from hip trajectory."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create hip Y trajectory: baseline, peak, return
        hip_y = np.array([0.3, 0.2, 0.1, 0.2, 0.3], dtype=np.float32)

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=0,
            peak=2,
            landing=4,
            end=4,
        )

        height = analyzer.compute_jump_height(hip_y, phases)

        # Height = baseline(0.3) - peak(0.1) = 0.2
        assert height == pytest.approx(0.2)

    def test_compute_arm_position(self, sample_normalized_poses):
        """Should compute arm position score."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        score = analyzer.compute_arm_position(sample_normalized_poses)

        assert 0 <= score <= 1

    def test_compute_trunk_lean(self, sample_normalized_poses):
        """Should compute trunk lean angle."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)

        leans = analyzer.compute_trunk_lean(sample_normalized_poses)

        assert len(leans) == 3
        # All values should be reasonable angles
        assert all(-90 <= lean <= 90 for lean in leans)

    def test_compute_edge_indicator(self, sample_normalized_poses):
        """Should compute edge indicator."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)

        indicator = analyzer.compute_edge_indicator(sample_normalized_poses, side="left")

        assert len(indicator) == 3
        # Values should be in range [-1, 1]
        assert all(-1 <= v <= 1 for v in indicator)

    def test_compute_symmetry(self, sample_normalized_poses):
        """Should compute symmetry score."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=0,
            peak=1,
            landing=2,
            end=2,
        )

        symmetry = analyzer.compute_symmetry(sample_normalized_poses, phases)

        assert 0 <= symmetry <= 1

    def test_analyze_jump(self, sample_normalized_poses):
        """Should analyze jump and return metrics."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=0,
            peak=1,
            landing=2,
            end=2,
        )

        metrics = analyzer.analyze(sample_normalized_poses, phases, fps=30.0)

        assert len(metrics) > 0
        assert all(isinstance(m, MetricResult) for m in metrics)

        # Check specific metrics exist
        metric_names = [m.name for m in metrics]
        assert "airtime" in metric_names

    def test_analyze_step(self, sample_normalized_poses):
        """Should analyze step element and return metrics."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)

        phases = ElementPhase(
            name="three_turn",
            start=0,
            takeoff=0,
            peak=1,
            landing=0,
            end=2,
        )

        metrics = analyzer.analyze(sample_normalized_poses, phases, fps=30.0)

        assert len(metrics) > 0

        # Check step-specific metrics
        metric_names = [m.name for m in metrics]
        assert "trunk_lean" in metric_names


class TestMetricResult:
    """Test MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Should create metric result correctly."""
        result = MetricResult(
            name="test_metric",
            value=0.5,
            unit="s",
            is_good=True,
            reference_range=(0.4, 0.6),
        )

        assert result.name == "test_metric"
        assert result.value == 0.5
        assert result.unit == "s"
        assert result.is_good is True
        assert result.reference_range == (0.4, 0.6)
