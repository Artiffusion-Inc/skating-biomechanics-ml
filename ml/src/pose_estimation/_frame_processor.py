from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .h36m import coco_to_h36m

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FrameProcessor:
    """Converts raw RTMO output (COCO keypoints) to H3.6M format per frame."""

    def __init__(self, output_format: str = "normalized") -> None:
        self.output_format = output_format

    def convert_keypoints(
        self,
        keypoints: NDArray[np.float32],  # (P, 17, 2) pixels
        scores: NDArray[np.float32],  # (P, 17)
        frame_width: int,
        frame_height: int,
    ) -> NDArray[np.float32]:  # (P, 17, 3)
        n_persons = keypoints.shape[0]
        h36m_poses = np.zeros((n_persons, 17, 3), dtype=np.float32)
        w, h = float(frame_width), float(frame_height)

        for p in range(n_persons):
            coco = np.zeros((17, 3), dtype=np.float32)
            coco[:, :2] = keypoints[p].astype(np.float32)
            coco[:, 2] = scores[p].astype(np.float32)
            coco[:, 0] /= w
            coco[:, 1] /= h

            h36m = coco_to_h36m(coco)

            if self.output_format == "pixels":
                h36m[:, 0] *= w
                h36m[:, 1] *= h

            h36m_poses[p] = h36m

        return h36m_poses
