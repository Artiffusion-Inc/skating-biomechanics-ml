"""Tests for IJS score calculation."""

import pytest

from backend.app.services.choreography.score_calculator import (
    calculate_goe_total,
    calculate_tes,
    goe_factor,
)


def test_goe_factor_low_bv():
    """BV < 2.0: factor = 0.5"""
    assert goe_factor(1.50) == pytest.approx(0.5)


def test_goe_factor_mid_bv():
    """2.0 <= BV < 4.0: factor = 0.7"""
    assert goe_factor(3.30) == pytest.approx(0.7)


def test_goe_factor_high_bv():
    """BV >= 4.0: factor = 1.0"""
    assert goe_factor(5.90) == pytest.approx(1.0)


def test_calculate_goe_total_positive():
    """GOE +3 on 3Lz (BV 5.90, factor 1.0) = 5.90 + 3*1.0 = 8.90"""
    total = calculate_goe_total(5.90, 3)
    assert total == pytest.approx(8.90)


def test_calculate_goe_total_negative():
    """GOE -2 on 2A (BV 3.30, factor 0.7) = 3.30 + (-2)*0.7 = 1.90"""
    total = calculate_goe_total(3.30, -2)
    assert total == pytest.approx(1.90)


def test_calculate_tes_basic():
    """Simple program with 3 elements, no back-half bonus."""
    elements = [
        {"code": "3Lz", "goe": 2},
        {"code": "CSp4", "goe": 1},
        {"code": "StSq4", "goe": 0},
    ]
    result = calculate_tes(elements, back_half_indices=set())
    # 3Lz: 5.90 + 2*1.0 = 7.90  (factor 1.0, BV >= 4.0)
    # CSp4: 3.20 + 1*0.7 = 3.90  (factor 0.7, 2.0 <= BV < 4.0)
    # StSq4: 3.90 + 0*0.7 = 3.90 (factor 0.7, 2.0 <= BV < 4.0)
    assert result == pytest.approx(15.70, abs=0.01)


def test_calculate_tes_with_back_half_bonus():
    """Back-half elements get +10% BV."""
    elements = [
        {"code": "3Lz", "goe": 2},
        {"code": "3F", "goe": 1},
        {"code": "3Lo", "goe": 0},
    ]
    # indices 1 and 2 are in back half
    result = calculate_tes(elements, back_half_indices={1, 2})
    # 3Lz: 5.90 + 2*1.0 = 7.90
    # 3F: (5.30 * 1.10) + 1*1.0 = 5.83 + 1.0 = 6.83
    # 3Lo: (4.90 * 1.10) + 0*1.0 = 5.39 + 0.0 = 5.39
    assert result == pytest.approx(20.12, abs=0.01)


def test_calculate_tes_empty():
    assert calculate_tes([], back_half_indices=set()) == 0.0
