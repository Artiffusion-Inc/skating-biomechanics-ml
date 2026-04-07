"""FootTrackNet wrapper for specialized foot detection.

Uses ONNX Runtime. Input: RGB image. Output: person + foot bounding boxes.

Model: FootTrackNet (2.53M params, ~30MB VRAM)
Source: Qualcomm AI Hub
ONNX: https://huggingface.co/qualcomm/Person-Foot-Detection
"""

from __future__ import annotations

import logging

import cv2
import numpy as np  # noqa: TC002 — used at runtime in detect()

from src.ml.model_registry import ModelRegistry  # noqa: TC001 — used at runtime

logger = logging.getLogger(__name__)

MODEL_ID = "foot_tracker"


class FootTracker:
    """Person and foot detection via FootTrackNet.

    Args:
        registry: ModelRegistry with "foot_tracker" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        details = self._session.get_input_details()
        self._input_name = details[0]["name"]
        self._input_size = (640, 480)  # FootTrackNet default

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect persons and feet in a frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            List of dicts with keys: ``bbox`` (x1,y1,x2,y2), ``class_id``
            (0=person, 1=foot), ``confidence`` (float).
        """
        h, w = frame.shape[:2]
        img = cv2.resize(frame, self._input_size, interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, swapRB=True)

        output = self._session.run(None, {self._input_name: blob})[0]

        detections: list[dict] = []
        for det in output:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls = det[:6]
                if conf > 0.3:
                    # Scale bbox back to original frame size
                    sx, sy = w / self._input_size[0], h / self._input_size[1]
                    detections.append(
                        {
                            "bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                            "class_id": int(cls),
                            "confidence": float(conf),
                        }
                    )

        return detections
