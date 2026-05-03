"""Smoke test: verify core inference pipeline without GPU/video.

Runs phase detection, biomechanics metrics, and recommendation generation
on synthetic jump poses. Does NOT require rtmlib, ONNX, or CUDA.
Purpose: CI gate to catch breaking changes in analysis logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.element_defs import get_element_def
from src.analysis.metrics import BiomechanicsAnalyzer
from src.analysis.phase_detector import PhaseDetector
from src.analysis.recommender import Recommender
from src.types import H36Key, MetricResult

FPS = 30.0


# ── Synthetic pose generator ──────────────────────────────────────────


def _make_jump_pose_sequence(n_frames: int = 90) -> np.ndarray:
    """Generate a synthetic waltz-jump pose sequence (N, 17, 2)."""
    poses = np.zeros((n_frames, 17, 2), dtype=np.float32)
    t = np.linspace(0, 1, n_frames)

    ground = 0.85
    com_y = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        if i < 25:
            com_y[i] = ground - 0.03 * np.sin(np.pi * i / 25)
        elif i < 30:
            frac = (i - 25) / 5
            com_y[i] = ground - 0.15 - 0.10 * frac  # deep crouch
        elif i < 35:
            frac = (i - 30) / 5
            com_y[i] = ground - 0.25 + 0.30 * frac  # explosive push
        elif i < 50:
            frac = (i - 35) / 15
            com_y[i] = ground - 0.55 + 0.45 * np.sin(np.pi * frac)  # flight arc (peak lower)
        elif i < 55:
            frac = (i - 50) / 5
            com_y[i] = ground - 0.10 - 0.10 * frac  # landing absorb
        elif i < 60:
            frac = (i - 55) / 5
            com_y[i] = ground - 0.20 + 0.10 * frac  # recovery
        else:
            com_y[i] = ground - 0.10 - 0.05 * np.sin(np.pi * (i - 60) / 30)

    hip_x = 0.2 + 0.6 * t
    for i in range(n_frames):
        hy = com_y[i]
        poses[i, H36Key.HIP_CENTER] = [hip_x[i], hy]
        poses[i, H36Key.RHIP] = [hip_x[i] + 0.05, hy - 0.02]
        poses[i, H36Key.LHIP] = [hip_x[i] - 0.05, hy - 0.02]

        knee_depth = 0.08
        if 25 <= i < 35 or 55 <= i < 60:
            knee_bend = knee_depth
        elif 35 <= i < 55:
            knee_bend = 0.02
        else:
            knee_bend = 0.04
        poses[i, H36Key.RKNEE] = [hip_x[i] + 0.05, hy - knee_bend]
        poses[i, H36Key.LKNEE] = [hip_x[i] - 0.05, hy - knee_bend]
        poses[i, H36Key.RFOOT] = [hip_x[i] + 0.06, hy - knee_bend - 0.12]
        poses[i, H36Key.LFOOT] = [hip_x[i] - 0.06, hy - knee_bend - 0.12]

        lean = 0.03 if 25 <= i < 35 else 0.01
        poses[i, H36Key.SPINE] = [hip_x[i], hy + 0.10]
        poses[i, H36Key.THORAX] = [hip_x[i] + lean, hy + 0.18]
        poses[i, H36Key.NECK] = [hip_x[i] + lean, hy + 0.22]
        poses[i, H36Key.HEAD] = [hip_x[i] + lean, hy + 0.26]

        arm_spread = 0.06 if 30 <= i < 55 else 0.12
        poses[i, H36Key.RSHOULDER] = [hip_x[i] + arm_spread, hy + 0.20]
        poses[i, H36Key.LSHOULDER] = [hip_x[i] - arm_spread, hy + 0.20]
        poses[i, H36Key.RELBOW] = [hip_x[i] + arm_spread + 0.02, hy + 0.15]
        poses[i, H36Key.LELBOW] = [hip_x[i] - arm_spread - 0.02, hy + 0.15]
        poses[i, H36Key.RWRIST] = [hip_x[i] + arm_spread + 0.04, hy + 0.10]
        poses[i, H36Key.LWRIST] = [hip_x[i] - arm_spread - 0.04, hy + 0.10]

    return poses


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_phase_detector_on_synthetic_jump():
    """PhaseDetector should identify takeoff, peak, landing on synthetic jump."""
    poses = _make_jump_pose_sequence(n_frames=90)
    detector = PhaseDetector()
    result = detector.detect_jump_phases(poses, fps=FPS)

    phases = result.phases
    assert phases is not None
    assert 0 < phases.takeoff < phases.peak < phases.landing < len(poses) - 1
    assert result.confidence > 0.0


@pytest.mark.smoke
def test_metrics_computation_on_synthetic_jump():
    """BiomechanicsAnalyzer should produce valid metrics on synthetic jump."""
    poses = _make_jump_pose_sequence(n_frames=90)
    phases = PhaseDetector().detect_jump_phases(poses, fps=FPS).phases
    element_def = get_element_def("waltz_jump")
    assert element_def is not None

    analyzer = BiomechanicsAnalyzer(element_def)
    metrics = analyzer.analyze(poses, phases, fps=FPS)

    assert isinstance(metrics, list)
    assert len(metrics) > 0
    for m in metrics:
        assert isinstance(m, MetricResult)
        assert m.name
        assert m.value is not None
        assert isinstance(m.is_good, bool)
        assert len(m.reference_range) == 2


@pytest.mark.smoke
def test_recommender_produces_advice():
    """Recommender should return Russian advice strings."""
    poses = _make_jump_pose_sequence(n_frames=90)
    phases = PhaseDetector().detect_jump_phases(poses, fps=FPS).phases
    element_def = get_element_def("waltz_jump")
    assert element_def is not None

    analyzer = BiomechanicsAnalyzer(element_def)
    metrics = analyzer.analyze(poses, phases, fps=FPS)
    recommender = Recommender()
    advice = recommender.recommend(metrics, element_type="waltz_jump")

    assert isinstance(advice, list)
    for line in advice:
        assert isinstance(line, str)
        assert len(line) > 0


@pytest.mark.smoke
def test_end_to_end_inference_pipeline():
    """Run full inference chain: poses → phases → metrics → recommendations."""
    poses = _make_jump_pose_sequence(n_frames=90)
    detector = PhaseDetector()
    element_def = get_element_def("waltz_jump")
    assert element_def is not None

    phase_result = detector.detect_jump_phases(poses, fps=FPS)
    assert phase_result.phases is not None

    analyzer = BiomechanicsAnalyzer(element_def)
    metrics = analyzer.analyze(poses, phase_result.phases, fps=FPS)
    assert len(metrics) > 0

    recommender = Recommender()
    advice = recommender.recommend(metrics, element_type="waltz_jump")
    assert isinstance(advice, list)

    # Spot-check: airtime should be positive
    airtime = next((m for m in metrics if m.name == "airtime"), None)
    if airtime:
        assert airtime.value > 0.0
