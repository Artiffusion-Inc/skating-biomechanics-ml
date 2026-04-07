"""SAM 2 wrapper for image segmentation.

Uses ONNX Runtime for inference. Input: RGB image + point prompt. Output: binary mask.

Model: SAM 2 Tiny (38.9M params, ~200MB VRAM)
Source: https://github.com/facebookresearch/sam2
ONNX: https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "segment_anything"
INPUT_SIZE = 1024


class SegmentAnything:
    """Image segmentation via SAM 2.

    Args:
        registry: ModelRegistry with "segment_anything" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        self._input_size = INPUT_SIZE
        details = self._session.get_input_details()
        self._input_names = [d["name"] for d in details]

    def segment(
        self,
        frame: np.ndarray,
        point: tuple[int, int] | None = None,
        box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray | None:
        """Segment the image using a point or box prompt.

        Args:
            frame: BGR image (H, W, 3) uint8.
            point: (x, y) pixel coordinate as prompt, or None.
            box: (x1, y1, x2, y2) pixel box as prompt, or None.

        Returns:
            Binary mask (H, W) bool, or None if no prompt provided.
        """
        if point is None and box is None:
            return None

        h, w = frame.shape[:2]

        # Prepare image
        img = cv2.resize(
            frame, (self._input_size, self._input_size), interpolation=cv2.INTER_LINEAR
        )
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = (img_rgb.astype(np.float32) - [123.675, 116.28, 103.53]) / [
            58.395,
            57.12,
            57.375,
        ]
        img_tensor = img_norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        # Prepare point prompt
        point_coords = np.array([], dtype=np.float32).reshape(0, 2)
        point_labels = np.array([], dtype=np.float32)
        if point is not None:
            # Scale point to model input size
            sx = self._input_size / w
            sy = self._input_size / h
            point_coords = np.array([[point[0] * sx, point[1] * sy]], dtype=np.float32)
            point_labels = np.array([1.0], dtype=np.float32)  # foreground

        # Build inputs (SAM2 ONNX format — adapt to actual ONNX export)
        inputs = {}
        for i, name in enumerate(self._input_names):
            if i == 0:
                inputs[name] = img_tensor
            elif "point_coords" in name.lower():
                inputs[name] = point_coords
            elif "point_labels" in name.lower():
                inputs[name] = point_labels

        try:
            outputs = self._session.run(None, inputs)
            # Find mask output (typically first or second output)
            masks = None
            for out in outputs:
                if isinstance(out, np.ndarray) and out.ndim == 4:
                    masks = out
                    break

            if masks is None:
                return None

            # Take best mask (highest IoU prediction)
            mask = masks[0, 0]  # (H_in, W_in)
            mask = mask > 0.0  # threshold

            # Resize to original frame size
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            return mask
        except Exception as e:
            logger.warning("SAM2 inference failed: %s", e)
            return None
