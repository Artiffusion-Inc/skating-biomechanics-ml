"""Tests for SVG rink renderer."""

from backend.app.services.choreography.rink_renderer import render_rink


def test_render_empty_rink():
    svg = render_rink([], width=600, height=300)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert 'viewBox="0 0 60 30"' in svg


def test_render_rink_with_elements():
    elements = [
        {"code": "3Lz", "position": {"x": 15.0, "y": 10.0}, "timestamp": 5.0},
        {"code": "CSp4", "position": {"x": 30.0, "y": 15.0}, "timestamp": 30.0},
    ]
    svg = render_rink(elements, width=1200, height=600)
    assert "3Lz" in svg
    assert "CSp4" in svg


def test_render_rink_dimensions():
    svg = render_rink([], width=800, height=400)
    assert 'width="800"' in svg
    assert 'height="400"' in svg


def test_render_has_center_circle():
    svg = render_rink([])
    assert "cx=" in svg
