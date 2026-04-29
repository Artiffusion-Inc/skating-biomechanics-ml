"""Tests for ml.src.web_helpers.

Covers the stateless pure helper functions used by the Gradio UI.
Heavy dependencies (cv2) are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add ml to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.types import PersonClick
from src.web_helpers import (
    PipelineCancelled,
    choice_to_person_click,
    match_click_to_person,
    persons_to_choices,
    render_person_preview,
)


class TestMatchClickToPerson:
    """Tests for match_click_to_person."""

    def test_empty_persons_returns_none(self):
        assert match_click_to_person([], 0.5, 0.5) is None

    def test_click_outside_all_bboxes_returns_none(self):
        persons = [
            {"bbox": [0.0, 0.0, 0.2, 0.2], "mid_hip": [0.1, 0.1]},
        ]
        result = match_click_to_person(persons, 0.5, 0.5)
        assert result is None

    def test_click_inside_bbox_returns_closest_mid_hip(self):
        persons = [
            {"bbox": [0.0, 0.0, 0.5, 0.5], "mid_hip": [0.1, 0.1], "hits": 1},
            {"bbox": [0.0, 0.0, 0.5, 0.5], "mid_hip": [0.2, 0.2], "hits": 2},
        ]
        result = match_click_to_person(persons, 0.15, 0.15)
        assert result is not None
        # (0.15, 0.15) is closer to (0.1, 0.1) than to (0.2, 0.2)
        assert result["mid_hip"] == [0.1, 0.1]

    def test_click_inside_second_bbox(self):
        persons = [
            {"bbox": [0.0, 0.0, 0.3, 0.3], "mid_hip": [0.1, 0.1], "hits": 1},
            {"bbox": [0.4, 0.4, 0.8, 0.8], "mid_hip": [0.6, 0.6], "hits": 3},
        ]
        result = match_click_to_person(persons, 0.5, 0.5)
        assert result is not None
        assert result["mid_hip"] == [0.6, 0.6]


class TestRenderPersonPreview:
    """Tests for render_person_preview."""

    @patch("src.web_helpers.cv2")
    def test_empty_persons_returns_frame_copy(self, mock_cv2):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = render_person_preview(frame, [])
        assert np.array_equal(result, frame)
        mock_cv2.rectangle.assert_not_called()

    @patch("src.web_helpers.cv2")
    def test_draws_bbox_for_single_person(self, mock_cv2):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        persons = [
            {"bbox": [0.1, 0.1, 0.3, 0.4], "mid_hip": [0.2, 0.25], "hits": 5, "track_id": 1},
        ]

        render_person_preview(frame, persons)

        # cv2.rectangle called twice: once for bbox, once for label bg
        assert mock_cv2.rectangle.call_count == 2
        mock_cv2.putText.assert_called_once()

    @patch("src.web_helpers.cv2")
    def test_selected_person_gets_green_bbox(self, mock_cv2):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        persons = [
            {"bbox": [0.1, 0.1, 0.3, 0.4], "mid_hip": [0.2, 0.25], "hits": 5, "track_id": 1},
        ]

        render_person_preview(frame, persons, selected_idx=0)

        # First rectangle call should be the bbox with green color
        first_call = mock_cv2.rectangle.call_args_list[0]
        assert first_call[0][3] == (0, 255, 0)
        assert first_call[0][4] == 3

    @patch("src.web_helpers.cv2")
    def test_multiple_persons_get_different_colors(self, mock_cv2):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        persons = [
            {"bbox": [0.1, 0.1, 0.2, 0.2], "mid_hip": [0.15, 0.15], "hits": 1, "track_id": 1},
            {"bbox": [0.3, 0.3, 0.4, 0.4], "mid_hip": [0.35, 0.35], "hits": 2, "track_id": 2},
            {"bbox": [0.5, 0.5, 0.6, 0.6], "mid_hip": [0.55, 0.55], "hits": 3, "track_id": 3},
        ]

        render_person_preview(frame, persons)

        # Each person: 2 rectangles (bbox + label bg) + 1 putText
        assert mock_cv2.rectangle.call_count == 6
        assert mock_cv2.putText.call_count == 3

    @patch("src.web_helpers.cv2")
    def test_label_contains_person_number_and_hits(self, mock_cv2):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        persons = [
            {"bbox": [0.1, 0.1, 0.3, 0.4], "mid_hip": [0.2, 0.25], "hits": 7, "track_id": 1},
        ]

        render_person_preview(frame, persons)

        putText_call = mock_cv2.putText.call_args
        text = putText_call[0][1]
        assert "#1" in text
        assert "hits: 7" in text


class TestPersonsToChoices:
    """Tests for persons_to_choices."""

    def test_empty_returns_empty(self):
        assert persons_to_choices([]) == []

    def test_single_person(self):
        persons = [{"hits": 10, "track_id": 3}]
        result = persons_to_choices(persons)
        assert result == ["Person #1 (10 hits, track 3)"]

    def test_multiple_persons(self):
        persons = [
            {"hits": 5, "track_id": 1},
            {"hits": 12, "track_id": 7},
        ]
        result = persons_to_choices(persons)
        assert result == [
            "Person #1 (5 hits, track 1)",
            "Person #2 (12 hits, track 7)",
        ]


class TestChoiceToPersonClick:
    """Tests for choice_to_person_click."""

    def test_converts_choice_to_click(self):
        persons = [
            {"mid_hip": [0.5, 0.5]},
        ]
        result = choice_to_person_click("Person #1 (5 hits, track 1)", persons, 640, 480)
        assert isinstance(result, PersonClick)
        assert result.x == 320  # 0.5 * 640
        assert result.y == 240  # 0.5 * 480

    def test_second_person(self):
        persons = [
            {"mid_hip": [0.1, 0.2]},
            {"mid_hip": [0.8, 0.9]},
        ]
        result = choice_to_person_click("Person #2 (10 hits, track 3)", persons, 100, 200)
        assert result.x == 80  # 0.8 * 100
        assert result.y == 180  # 0.9 * 200

    def test_zero_coordinates(self):
        persons = [
            {"mid_hip": [0.0, 0.0]},
        ]
        result = choice_to_person_click("Person #1 (1 hits, track 0)", persons, 640, 480)
        assert result.x == 0
        assert result.y == 0


class TestPipelineCancelled:
    """Tests for the PipelineCancelled exception."""

    def test_is_exception(self):
        with pytest.raises(PipelineCancelled):
            raise PipelineCancelled("cancelled")

    def test_message(self):
        exc = PipelineCancelled("user stopped")
        assert str(exc) == "user stopped"
