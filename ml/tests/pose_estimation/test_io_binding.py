"""Tests for IO Binding zero-copy inference in BatchRTMO.

IO Binding pre-allocates GPU tensors via OrtValue, eliminating
CPU->GPU->CPU copies per call. Effective when batch size is constant.

Since BatchRTMO imports onnxruntime lazily inside __init__ and
infer_batch_iobinding, we mock it at the sys.modules level.
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Track original onnxruntime to restore after tests
_ORT_ORIG = sys.modules.get("onnxruntime")


def _install_ort_mock():
    """Create and install a fully-featured onnxruntime mock.

    Returns (mock_ort, mock_session, mock_opts, mock_binding).
    """
    mock_ort = MagicMock()
    mock_ort.__spec__ = importlib.util.spec_from_loader("onnxruntime", loader=None)
    mock_ort.__file__ = "/fake/onnxruntime/__init__.py"
    mock_ort.__path__ = ["/fake/onnxruntime"]
    mock_ort.__package__ = "onnxruntime"

    mock_ort.GraphOptimizationLevel = MagicMock()
    mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

    mock_ort.ExecutionMode = MagicMock()
    mock_ort.ExecutionMode.ORT_SEQUENTIAL = 0

    # SessionOptions
    mock_opts = types.SimpleNamespace(
        graph_optimization_level=None,
        enable_mem_pattern=None,
        enable_mem_reuse=None,
        intra_op_num_threads=None,
        inter_op_num_threads=None,
    )
    mock_ort.SessionOptions = MagicMock(return_value=mock_opts)

    # IO Binding mock with proper output data
    mock_binding = MagicMock()
    mock_input_ortvalue = MagicMock()
    mock_binding.get_inputs.return_value = [
        MagicMock(get_ortvalue=MagicMock(return_value=mock_input_ortvalue))
    ]
    # Empty detections so postprocess_batch yields zeros without errors
    mock_output_det = MagicMock(
        get_numpy_data=MagicMock(return_value=np.zeros((1, 0, 5), dtype=np.float32))
    )
    mock_output_kp = MagicMock(
        get_numpy_data=MagicMock(return_value=np.zeros((1, 0, 17, 3), dtype=np.float32))
    )
    mock_binding.get_outputs.return_value = [mock_output_det, mock_output_kp]

    mock_session = MagicMock()
    mock_session.io_binding.return_value = mock_binding
    mock_session.get_inputs.return_value = [MagicMock(name="images", shape=[None, 3, 640, 640])]
    mock_session.get_outputs.return_value = [
        MagicMock(name="det_output"),
        MagicMock(name="kp_output"),
    ]

    mock_ort.InferenceSession.return_value = mock_session

    # OrtValue mock
    mock_ortvalue_cls = MagicMock()
    mock_ortvalue_instance = MagicMock()
    mock_ortvalue_cls.ortvalue_from_shape_and_type.return_value = mock_ortvalue_instance
    mock_ort.OrtValue = mock_ortvalue_cls

    sys.modules["onnxruntime"] = mock_ort
    return mock_ort, mock_session, mock_opts, mock_binding


def _restore_ort():
    """Restore original onnxruntime in sys.modules."""
    if _ORT_ORIG is not None:
        sys.modules["onnxruntime"] = _ORT_ORIG
    else:
        sys.modules.pop("onnxruntime", None)


def _evict_rtmo_batch():
    """Remove cached rtmo_batch module so re-import picks up mocks."""
    for key in list(sys.modules.keys()):
        if "rtmo_batch" in key:
            del sys.modules[key]


@contextmanager
def _mocked_batchrtmo(test_device: str = "cuda"):
    """Context manager that provides a BatchRTMO with mocked onnxruntime.

    The mock stays active for the entire context so infer_batch_iobinding
    (which does lazy `import onnxruntime as ort`) also sees the mock.

    Yields (mock_ort, mock_session, mock_opts, mock_binding, instance).
    """
    mock_ort, mock_session, mock_opts, mock_binding = _install_ort_mock()
    _evict_rtmo_batch()

    try:
        with (
            patch.dict(
                "src.pose_estimation.rtmo_batch.RTMO_MODELS", {"balanced": "/fake/model.onnx"}
            ),
            patch("src.pose_estimation.rtmo_batch.Path") as mock_path_cls,
        ):
            mock_path = MagicMock()
            mock_path.__str__ = lambda self: "/fake/model.onnx"
            mock_path.exists.return_value = True
            mock_path_cls.return_value = mock_path

            from src.pose_estimation.rtmo_batch import BatchRTMO

            instance = BatchRTMO(
                mode="balanced",
                device=test_device,
                score_thr=0.3,
                nms_thr=0.45,
            )

            yield mock_ort, mock_session, mock_opts, mock_binding, instance
    finally:
        _restore_ort()
        _evict_rtmo_batch()


@pytest.mark.cuda
def test_batch_rtmo_io_binding_matches_regular():
    """IO Binding results should match regular session.run results."""
    pytest.skip(
        "Requires GPU — run manually with: uv run python scripts/benchmark_batch_inference.py"
    )


class TestIOBindingMethod:
    """BatchRTMO should expose infer_batch_iobinding for zero-copy GPU inference."""

    def test_batch_rtmo_has_io_binding_method(self):
        """BatchRTMO should have infer_batch_iobinding method."""
        with _mocked_batchrtmo() as (_ort, _session, _opts, _binding, instance):
            assert hasattr(instance, "infer_batch_iobinding"), (
                "BatchRTMO is missing infer_batch_iobinding method"
            )
            assert callable(instance.infer_batch_iobinding), (
                "infer_batch_iobinding should be callable"
            )

    def test_io_binding_returns_empty_for_no_frames(self):
        """infer_batch_iobinding should return empty list for empty input."""
        with _mocked_batchrtmo() as (_ort, _session, _opts, _binding, instance):
            result = instance.infer_batch_iobinding([])
            assert result == []

    def test_io_binding_creates_ortvalue_on_first_call(self):
        """First call to infer_batch_iobinding should create OrtValue via io_binding."""
        with _mocked_batchrtmo() as (mock_ort, mock_session, _opts, _binding, instance):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            instance.infer_batch_iobinding([frame])

            mock_session.io_binding.assert_called_once()
            mock_ort.OrtValue.ortvalue_from_shape_and_type.assert_called_once()
            call_args = mock_ort.OrtValue.ortvalue_from_shape_and_type.call_args
            shape = call_args[0][0]
            assert shape == [1, 3, 640, 640], f"Expected [1, 3, 640, 640], got {shape}"

    def test_io_binding_reuses_binding_for_same_batch_size(self):
        """io_binding should be reused when batch size matches."""
        with _mocked_batchrtmo() as (_ort, mock_session, _opts, _binding, instance):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            instance.infer_batch_iobinding([frame])
            instance.infer_batch_iobinding([frame])

            # io_binding called once for initial creation, reused on second call
            assert mock_session.io_binding.call_count == 1

    def test_io_binding_recreates_for_different_batch_size(self):
        """io_binding should be recreated when batch size changes."""
        with _mocked_batchrtmo() as (_ort, mock_session, _opts, _binding, instance):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            instance.infer_batch_iobinding([frame])
            instance.infer_batch_iobinding([frame, frame])

            # io_binding called again because batch size changed
            assert mock_session.io_binding.call_count == 2

    def test_io_binding_uses_run_with_iobinding(self):
        """infer_batch_iobinding should use session.run_with_iobinding."""
        with _mocked_batchrtmo() as (_ort, mock_session, _opts, _binding, instance):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            instance.infer_batch_iobinding([frame])

            mock_session.run_with_iobinding.assert_called_once()

    def test_io_binding_binds_input_and_outputs(self):
        """IO binding should bind input OrtValue and output names to CUDA."""
        with _mocked_batchrtmo() as (_ort, _session, _opts, mock_binding, instance):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            instance.infer_batch_iobinding([frame])

            mock_binding.bind_ortvalue_input.assert_called_once()
            call_args = mock_binding.bind_ortvalue_input.call_args
            input_name = call_args[0][0]
            assert input_name == instance._input_name

            output_names = instance._output_names
            assert mock_binding.bind_output.call_count == len(output_names)
            for name in output_names:
                mock_binding.bind_output.assert_any_call(name, "cuda", 0)

    def test_io_binding_input_shape_matches_batch_size(self):
        """OrtValue shape should match actual batch size."""
        with _mocked_batchrtmo() as (mock_ort, _session, _opts, _binding, instance):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            batch = [frame, frame, frame]
            instance.infer_batch_iobinding(batch)

            call_args = mock_ort.OrtValue.ortvalue_from_shape_and_type.call_args
            shape = call_args[0][0]
            assert shape[0] == 3, f"Expected batch dim 3, got {shape[0]}"

    def test_io_binding_updates_input_inplace(self):
        """infer_batch_iobinding should update OrtValue in-place."""
        with _mocked_batchrtmo() as (_ort, _session, _opts, mock_binding, instance):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            instance.infer_batch_iobinding([frame])

            mock_input_ortvalue = mock_binding.get_inputs.return_value[0].get_ortvalue.return_value
            mock_input_ortvalue.update_inplace.assert_called_once()
            call_args = mock_input_ortvalue.update_inplace.call_args
            batch_tensor = call_args[0][0]
            assert batch_tensor.shape == (1, 3, 640, 640)
            assert batch_tensor.dtype == np.float32

    def test_io_binding_returns_empty_detections(self):
        """infer_batch_iobinding should return postprocessed results."""
        with _mocked_batchrtmo() as (_ort, _session, _opts, _binding, instance):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = instance.infer_batch_iobinding([frame])

            assert isinstance(result, list)
            assert len(result) == 1
            kp, scores = result[0]
            assert kp.shape == (0, 17, 2)
            assert scores.shape == (0, 17)

    def test_close_releases_binding_cache(self):
        """close() should clear the IO binding cache."""
        with _mocked_batchrtmo() as (_ort, _session, _opts, _binding, instance):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            instance.infer_batch_iobinding([frame])

            assert hasattr(instance, "_binding")
            assert hasattr(instance, "_binding_batch_size")

            instance.close()
            assert not hasattr(instance, "_binding")
            assert not hasattr(instance, "_binding_batch_size")

    def test_io_binding_falls_back_on_cpu(self):
        """infer_batch_iobinding should fall back to infer_batch on CPU device."""
        with _mocked_batchrtmo(test_device="cpu") as (
            _ort,
            mock_session,
            _opts,
            _binding,
            instance,
        ):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Mock session.run to return empty detections
            mock_session.run.return_value = [
                np.zeros((1, 0, 5), dtype=np.float32),
                np.zeros((1, 0, 17, 3), dtype=np.float32),
            ]

            result = instance.infer_batch_iobinding([frame])

            # Should have used regular session.run, not IO binding
            mock_session.run_with_iobinding.assert_not_called()
            mock_session.run.assert_called()
            assert isinstance(result, list)
            assert len(result) == 1
