"""NeuFlowV2 optical flow wrapper.

Uses ONNX Runtime for inference. Input: frame pair (BGR). Output: dense flow field (H, W, 2).

Model: NeuFlowV2 (mixed training)
Source: https://github.com/neufieldrobotics/NeuFlow_v2
ONNX: https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ml.model_registry import ModelRegistry  # noqa: TC001

logger = logging.getLogger(__name__)

MODEL_ID = "optical_flow"


class OpticalFlowEstimator:
    """Dense optical flow estimation via NeuFlowV2.

    Args:
        registry: ModelRegistry with "optical_flow" registered.

    Supports two usage patterns:
    - ``estimate(frame1, frame2)`` -- explicit frame pair
    - ``estimate_from_previous(frame)`` -- caches previous frame automatically
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        self._prev_frame: np.ndarray | None = None
        # NeuFlowV2 expects two separate images
        details = self._session.get_input_details()
        self._input_names = [d["name"] for d in details]

    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Estimate optical flow between two frames.

        Args:
            frame1: BGR image (H, W, 3) uint8.
            frame2: BGR image (H, W, 3) uint8, same size as frame1.

        Returns:
            Flow field (H, W, 2) float32.

        Raises:
            ValueError: If frames have different sizes.
        """
        if frame1.shape[:2] != frame2.shape[:2]:
            raise ValueError(
                f"Frames must have the same size: {frame1.shape[:2]} vs {frame2.shape[:2]}"
            )

        h, w = frame1.shape[:2]

        # Prepare inputs -- NeuFlowV2 expects two separate images
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0

        inputs = {
            self._input_names[0]: img1[np.newaxis],
            self._input_names[1]: img2[np.newaxis],
        }

        # Inference
        output = self._session.run(None, inputs)[0]

        # Output shape: (2, H, W) -> (H, W, 2)
        flow = output.transpose(1, 2, 0).astype(np.float32)

        # Resize to original frame size if needed
        if flow.shape[:2] != (h, w):
            flow_xy = np.stack([flow[:, :, 0], flow[:, :, 1]], axis=-1)
            flow = cv2.resize(flow_xy, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        return flow

    def estimate_from_previous(self, frame: np.ndarray) -> np.ndarray | None:
        """Estimate flow from previously cached frame.

        On first call, caches the frame and returns None.
        On subsequent calls, estimates flow between previous and current frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            Flow field (H, W, 2) float32, or None on first call.
        """
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return None

        flow = self.estimate(self._prev_frame, frame)
        self._prev_frame = frame.copy()
        return flow

    def reset(self) -> None:
        """Clear cached previous frame."""
        self._prev_frame = None
