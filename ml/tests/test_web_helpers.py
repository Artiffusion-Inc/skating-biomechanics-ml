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
    _find_model,
    _run_analysis,
    choice_to_person_click,
    match_click_to_person,
    persons_to_choices,
    process_video_pipeline,
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


class TestFindModel:
    """Tests for _find_model."""

    def test_finds_existing_model(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "data" / "models"
        models_dir.mkdir(parents=True)
        model_file = models_dir / "test_model.onnx"
        model_file.write_text("dummy")

        monkeypatch.setattr("src.web_helpers._PROJECT_ROOT", tmp_path)
        result = _find_model("test_model.onnx")
        assert result == str(model_file)

    def test_raises_when_model_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.web_helpers._PROJECT_ROOT", tmp_path)
        with pytest.raises(FileNotFoundError, match="Model not found"):
            _find_model("missing.onnx")


class TestRunAnalysis:
    """Tests for _run_analysis."""

    @patch("src.analysis.element_defs.get_element_def")
    def test_returns_empty_when_element_def_missing(self, mock_get_def):
        mock_get_def.return_value = None
        poses = np.zeros((10, 17, 2))
        result = _run_analysis(poses, 30.0, "unknown")
        assert result == ([], None, [])

    @patch("src.analysis.recommender.Recommender")
    @patch("src.analysis.metrics.BiomechanicsAnalyzer")
    @patch("src.analysis.phase_detector.PhaseDetector")
    @patch("src.analysis.element_defs.get_element_def")
    def test_happy_path(self, mock_get_def, mock_phase, mock_analyzer, mock_recommender):
        mock_get_def.return_value = {"name": "waltz_jump"}

        mock_phase_instance = MagicMock()
        mock_phase_instance.detect_phases.return_value = MagicMock(
            phases={"takeoff": 1, "landing": 5}
        )
        mock_phase.return_value = mock_phase_instance

        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = [{"name": "height", "value": 0.5}]
        mock_analyzer.return_value = mock_analyzer_instance

        mock_recommender_instance = MagicMock()
        mock_recommender_instance.recommend.return_value = ["Bend knees more"]
        mock_recommender.return_value = mock_recommender_instance

        poses = np.zeros((10, 17, 2))
        metrics, phases, recommendations = _run_analysis(poses, 30.0, "waltz_jump")

        assert metrics == [{"name": "height", "value": 0.5}]
        assert phases == {"takeoff": 1, "landing": 5}
        assert recommendations == ["Bend knees more"]


class TestProcessVideoPipeline:
    """Tests for process_video_pipeline."""

    @pytest.fixture
    def mock_prepared(self):
        prepared = MagicMock()
        prepared.meta.width = 640
        prepared.meta.height = 480
        prepared.meta.fps = 30.0
        prepared.meta.num_frames = 10
        prepared.n_valid = 5
        prepared.poses_norm = np.zeros((5, 17, 2))
        prepared.poses_px = np.ones((5, 17, 2)) * 100
        prepared.poses_3d = None
        prepared.confs = np.ones((5, 17))
        prepared.frame_indices = np.arange(5)
        return prepared

    @pytest.fixture
    def mock_video_path(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_text("dummy")
        return str(video)

    def _setup_reader_mock(self, mock_reader_cls, num_frames=1):
        mock_reader = MagicMock()
        frames = [(i, np.zeros((480, 640, 3), dtype=np.uint8)) for i in range(num_frames)]
        frames.append(None)
        mock_reader.get_frame.side_effect = frames
        mock_reader_cls.return_value = mock_reader
        return mock_reader

    def _setup_pipeline_mocks(self, mock_viz_cls, mock_writer_cls):
        mock_pipe = MagicMock()
        mock_pipe.find_pose_idx.return_value = (0, 0)
        mock_pipe.render_frame.return_value = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            MagicMock(custom_data={}),
        )
        mock_pipe.save_exports.return_value = {"poses_path": "poses.npz", "csv_path": "data.csv"}
        mock_viz_cls.return_value = mock_pipe

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer
        return mock_pipe, mock_writer

    @patch("src.utils.frame_buffer.AsyncFrameReader")
    @patch("src.web_helpers.H264Writer")
    @patch("src.visualization.pipeline.VizPipeline")
    @patch("src.visualization.pipeline.prepare_poses")
    def test_happy_path_minimal(
        self,
        mock_prepare,
        mock_viz,
        mock_writer,
        mock_reader,
        mock_prepared,
        mock_video_path,
        tmp_path,
    ):
        mock_prepare.return_value = mock_prepared
        _mock_pipe, mock_writer = self._setup_pipeline_mocks(mock_viz, mock_writer)
        mock_reader_instance = self._setup_reader_mock(mock_reader, num_frames=3)

        output_path = tmp_path / "output.mp4"
        result = process_video_pipeline(
            video_path=mock_video_path,
            person_click=None,
            frame_skip=1,
            layer=0,
            tracking="sports2d",
            blade_3d=False,
            export=False,
            output_path=str(output_path),
        )

        assert result["video_path"] == str(output_path)
        assert result["stats"]["total_frames"] == 10
        assert result["stats"]["valid_frames"] == 5
        assert result["metrics"] == []
        assert result["phases"] is None
        assert result["recommendations"] == []
        mock_reader_instance.start.assert_called_once()
        mock_reader_instance.join.assert_called_once_with(timeout=5)
        mock_writer.close.assert_called_once()

    @patch("src.utils.frame_buffer.AsyncFrameReader")
    @patch("src.web_helpers.H264Writer")
    @patch("src.visualization.pipeline.VizPipeline")
    @patch("src.visualization.pipeline.prepare_poses")
    def test_cancel_event_raises(
        self,
        mock_prepare,
        mock_viz,
        mock_writer,
        mock_reader,
        mock_prepared,
        mock_video_path,
        tmp_path,
    ):
        mock_prepare.return_value = mock_prepared
        _mock_pipe, mock_writer = self._setup_pipeline_mocks(mock_viz, mock_writer)
        mock_reader_instance = self._setup_reader_mock(mock_reader, num_frames=3)

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = True

        output_path = tmp_path / "output.mp4"
        with pytest.raises(PipelineCancelled):
            process_video_pipeline(
                video_path=mock_video_path,
                person_click=None,
                frame_skip=1,
                layer=0,
                tracking="sports2d",
                blade_3d=False,
                export=False,
                output_path=str(output_path),
                cancel_event=cancel_event,
            )

        mock_reader_instance.join.assert_called_once_with(timeout=1)
        mock_writer.close.assert_called_once()

    @patch("src.web_helpers.logger")
    @patch("src.web_helpers._find_model")
    @patch("src.utils.frame_buffer.AsyncFrameReader")
    @patch("src.web_helpers.H264Writer")
    @patch("src.visualization.pipeline.VizPipeline")
    @patch("src.visualization.pipeline.prepare_poses")
    def test_model_not_found_branches(
        self,
        mock_prepare,
        mock_viz,
        mock_writer,
        mock_reader,
        mock_find_model,
        mock_logger,
        mock_prepared,
        mock_video_path,
        tmp_path,
    ):
        mock_prepare.return_value = mock_prepared
        mock_pipe, mock_writer = self._setup_pipeline_mocks(mock_viz, mock_writer)
        self._setup_reader_mock(mock_reader, num_frames=1)

        mock_find_model.side_effect = FileNotFoundError("Model not found")

        output_path = tmp_path / "output.mp4"
        result = process_video_pipeline(
            video_path=mock_video_path,
            person_click=None,
            frame_skip=1,
            layer=0,
            tracking="sports2d",
            blade_3d=False,
            export=False,
            output_path=str(output_path),
            depth=True,
            optical_flow=True,
            segment=True,
            foot_track=True,
            matting=True,
            inpainting=True,
        )

        assert "video_path" in result
        assert mock_find_model.call_count == 6
        assert mock_logger.warning.call_count == 6
        mock_pipe.add_ml_layers.assert_called_once_with([])

    @patch("src.visualization.render_layers")
    @patch("src.utils.frame_buffer.AsyncFrameReader")
    @patch("src.web_helpers.H264Writer")
    @patch("src.visualization.pipeline.VizPipeline")
    @patch("src.visualization.pipeline.prepare_poses")
    @patch("src.extras.inpainting.ImageInpainter")
    @patch("src.extras.foot_tracker.FootTracker")
    @patch("src.visualization.layers.foot_tracker_layer.FootTrackerLayer")
    @patch("src.extras.video_matting.VideoMatting")
    @patch("src.visualization.layers.matting_layer.MattingLayer")
    @patch("src.extras.segment_anything.SegmentAnything")
    @patch("src.visualization.layers.segmentation_layer.SegmentationMaskLayer")
    @patch("src.extras.optical_flow.OpticalFlowEstimator")
    @patch("src.visualization.layers.optical_flow_layer.OpticalFlowLayer")
    @patch("src.extras.depth_anything.DepthEstimator")
    @patch("src.visualization.layers.depth_layer.DepthMapLayer")
    @patch("src.extras.model_registry.ModelRegistry")
    @patch("src.web_helpers._find_model")
    def test_ml_loaded_and_render_loop(  # noqa: PLR0913
        self,
        mock_find_model,
        mock_registry,
        mock_depth_layer,
        mock_depth_est,
        mock_flow_layer,
        mock_flow_est,
        mock_seg_layer,
        mock_seg_est,
        mock_matting_layer,
        mock_matting_est,
        mock_foot_layer,
        mock_foot_est,
        mock_inpaint_est,
        mock_prepare,
        mock_viz,
        mock_writer,
        mock_reader,
        mock_render_layers,
        mock_prepared,
        mock_video_path,
        tmp_path,
    ):
        mock_prepare.return_value = mock_prepared
        mock_pipe, mock_writer = self._setup_pipeline_mocks(mock_viz, mock_writer)
        self._setup_reader_mock(mock_reader, num_frames=2)

        mock_context = MagicMock()
        mock_context.custom_data = {}
        mock_pipe.render_frame.return_value = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            mock_context,
        )

        mock_find_model.return_value = "/fake/model.onnx"
        mock_registry_instance = MagicMock()
        mock_registry_instance.is_registered.return_value = True
        mock_registry.return_value = mock_registry_instance

        mock_depth_est_instance = MagicMock()
        mock_depth_est_instance.estimate.return_value = np.zeros((480, 640), dtype=np.float32)
        mock_depth_est.return_value = mock_depth_est_instance

        mock_flow_est_instance = MagicMock()
        mock_flow_est_instance.estimate_from_previous.return_value = np.zeros(
            (480, 640, 2), dtype=np.float32
        )
        mock_flow_est.return_value = mock_flow_est_instance

        mock_seg_est_instance = MagicMock()
        mock_seg_est_instance.segment.return_value = np.ones((480, 640), dtype=np.uint8)
        mock_seg_est.return_value = mock_seg_est_instance

        mock_matting_est_instance = MagicMock()
        mock_matting_est_instance.matting.return_value = np.ones((480, 640), dtype=np.float32)
        mock_matting_est.return_value = mock_matting_est_instance

        mock_foot_est_instance = MagicMock()
        mock_foot_est_instance.detect.return_value = [{"x": 100, "y": 200}]
        mock_foot_est.return_value = mock_foot_est_instance

        mock_inpaint_est_instance = MagicMock()
        mock_inpaint_est_instance.inpaint.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_inpaint_est.return_value = mock_inpaint_est_instance

        mock_render_layers.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        output_path = tmp_path / "output.mp4"
        result = process_video_pipeline(
            video_path=mock_video_path,
            person_click=None,
            frame_skip=1,
            layer=1,
            tracking="sports2d",
            blade_3d=False,
            export=False,
            output_path=str(output_path),
            depth=True,
            optical_flow=True,
            segment=True,
            foot_track=True,
            matting=True,
            inpainting=True,
        )

        assert "video_path" in result
        mock_depth_est_instance.estimate.assert_called()
        mock_flow_est_instance.estimate_from_previous.assert_called()
        mock_seg_est_instance.segment.assert_called()
        mock_matting_est_instance.matting.assert_called()
        mock_foot_est_instance.detect.assert_called()
        mock_inpaint_est_instance.inpaint.assert_called()
        mock_render_layers.assert_called()
        assert len(mock_pipe.add_ml_layers.call_args[0][0]) == 5

    @patch("src.web_helpers._run_analysis")
    @patch("src.utils.frame_buffer.AsyncFrameReader")
    @patch("src.web_helpers.H264Writer")
    @patch("src.visualization.pipeline.VizPipeline")
    @patch("src.visualization.pipeline.prepare_poses")
    def test_with_element_type_and_export(
        self,
        mock_prepare,
        mock_viz,
        mock_writer,
        mock_reader,
        mock_run_analysis,
        mock_prepared,
        mock_video_path,
        tmp_path,
    ):
        mock_prepare.return_value = mock_prepared
        mock_pipe, mock_writer = self._setup_pipeline_mocks(mock_viz, mock_writer)
        self._setup_reader_mock(mock_reader, num_frames=1)

        mock_run_analysis.return_value = (
            [{"name": "height", "value": 0.5}],
            {"takeoff": 1, "landing": 5},
            ["Bend knees"],
        )

        output_path = tmp_path / "output.mp4"
        result = process_video_pipeline(
            video_path=mock_video_path,
            person_click=None,
            frame_skip=1,
            layer=0,
            tracking="sports2d",
            blade_3d=False,
            export=True,
            output_path=str(output_path),
            element_type="waltz_jump",
        )

        assert result["metrics"] == [{"name": "height", "value": 0.5}]
        assert result["phases"] == {"takeoff": 1, "landing": 5}
        assert result["recommendations"] == ["Bend knees"]
        mock_pipe.save_exports.assert_called_once_with(output_path)
        assert result["poses_path"] == "poses.npz"
        assert result["csv_path"] == "data.csv"

    @patch("src.utils.frame_buffer.AsyncFrameReader")
    @patch("src.web_helpers.H264Writer")
    @patch("src.visualization.pipeline.VizPipeline")
    @patch("src.visualization.pipeline.prepare_poses")
    def test_progress_callback(
        self,
        mock_prepare,
        mock_viz,
        mock_writer,
        mock_reader,
        mock_prepared,
        mock_video_path,
        tmp_path,
    ):
        mock_prepare.return_value = mock_prepared
        _mock_pipe, mock_writer = self._setup_pipeline_mocks(mock_viz, mock_writer)
        mock_prepared.meta.num_frames = 50
        self._setup_reader_mock(mock_reader, num_frames=50)

        progress_cb = MagicMock()

        output_path = tmp_path / "output.mp4"
        result = process_video_pipeline(
            video_path=mock_video_path,
            person_click=None,
            frame_skip=1,
            layer=0,
            tracking="sports2d",
            blade_3d=False,
            export=False,
            output_path=str(output_path),
            progress_cb=progress_cb,
        )

        assert progress_cb.call_count >= 2
        progress_cb.assert_any_call(0.95, "Saving exports...")
