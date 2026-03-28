"""AthletePose3D 3D pose estimator.

Monocular 3D pose estimation using fine-tuned AthletePose3D models.

Models:
- MotionAgFormer-S: 59MB, fast, suitable for RTX 3050 Ti
- TCPFormer: 422MB, more accurate

Reference: AthletePose3D: A Large-Scale 3D Sports Pose Dataset
"""

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .blazepose_to_h36m import blazepose_to_h36m


class AthletePose3DExtractor:
    """Monocular 3D pose estimation using AthletePose3D.

    Processes 2D poses (H3.6M 17-keypoint format) and outputs 3D poses.
    Uses temporal modeling with 81-frame windows.
    """

    # Temporal window size (frames)
    TEMPORAL_WINDOW = 81

    def __init__(
        self,
        model_path: Path | str,
        device: str = "auto",
        model_type: str = "motionagformer-s",
    ):
        """Initialize the 3D pose estimator.

        Args:
            model_path: Path to model checkpoint (.pth.tr file)
            device: "cuda", "cpu", or "auto" (default)
            model_type: Model architecture type
        """
        self.model_path = Path(model_path)
        self.model_type = model_type

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Temporal buffer for 81-frame window
        self.temporal_buffer: deque[np.ndarray] = deque(maxlen=self.TEMPORAL_WINDOW)

        # Load model (lazy loading on first use)
        self.model: torch.nn.Module | None = None
        self._model_loaded = False

    def _load_model(self) -> torch.nn.Module:
        """Load the AthletePose3D model."""
        if self._model_loaded:
            return self.model  # type: ignore

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        # Model architecture depends on type
        # This is a placeholder - actual model loading depends on
        # the AthletePose3D repository implementation
        if self.model_type == "motionagformer-s":
            from .models import MotionAgFormerS  # type: ignore

            self.model = MotionAgFormerS()
        elif self.model_type == "tcpformer":
            from .models import TCPFormer  # type: ignore

            self.model = TCPFormer()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        self._model_loaded = True

        return self.model

    def extract_frame(
        self,
        pose_2d: np.ndarray,
    ) -> np.ndarray | None:
        """Extract 3D pose from single 2D pose (with temporal context).

        Args:
            pose_2d: (17, 2) or (17, 3) array in H3.6M format
                - If (17, 2): x, y coordinates
                - If (17, 3): x, y, confidence

        Returns:
            pose_3d: (17, 3) array with x, y, z coordinates
                Returns None if temporal buffer not full yet
        """
        # Add to temporal buffer
        self.temporal_buffer.append(pose_2d)

        # Need full window for inference
        if len(self.temporal_buffer) < self.TEMPORAL_WINDOW:
            return None

        # Stack into temporal window
        window = np.stack(list(self.temporal_buffer))  # (81, 17, 2 or 3)

        return self._extract_window(window)

    def extract_sequence(
        self,
        poses_2d: np.ndarray,
    ) -> np.ndarray:
        """Extract 3D poses from 2D pose sequence.

        Args:
            poses_2d: (N, 17, 2) or (N, 17, 3) array in H3.6M format

        Returns:
            poses_3d: (N, 17, 3) array with x, y, z coordinates
        """
        from .blazepose_to_h36m import blazepose_to_h36m  # noqa: F401

        n_frames = poses_2d.shape[0]

        # Pad sequence to multiple of window size
        pad_size = (self.TEMPORAL_WINDOW - (n_frames % self.TEMPORAL_WINDOW)) % self.TEMPORAL_WINDOW
        if pad_size > 0:
            # Repeat last frame for padding
            padding = np.tile(poses_2d[-1:], (pad_size, 1, 1))
            poses_2d_padded = np.vstack([poses_2d, padding])
        else:
            poses_2d_padded = poses_2d

        # Process in windows
        poses_3d_list = []
        for i in range(0, len(poses_2d_padded) - self.TEMPORAL_WINDOW + 1, self.TEMPORAL_WINDOW // 2):
            window = poses_2d_padded[i:i + self.TEMPORAL_WINDOW]
            pose_3d = self._extract_window(window)
            poses_3d_list.append(pose_3d)

        # Merge overlapping windows (simple average)
        # For now, just concatenate and trim
        poses_3d = np.vstack(poses_3d_list)[:n_frames]

        return poses_3d

    def _extract_window(self, window: np.ndarray) -> np.ndarray:
        """Extract 3D pose from temporal window.

        Args:
            window: (81, 17, 2) or (81, 17, 3) array

        Returns:
            pose_3d: (17, 3) array (middle frame of window)
        """
        # Ensure correct format
        if window.shape[2] == 3:
            # Drop confidence channel
            window = window[:, :, :2]

        # Convert to tensor
        tensor = torch.from_numpy(window).float().to(self.device)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # (1, 81, 17, 2)

        # Run inference
        with torch.no_grad():
            model = self._load_model()
            output = model(tensor)  # (1, 81, 17, 3) or (1, 17, 3)

        # Extract result
        if output.dim() == 4:
            # Full sequence output - take middle frame
            pose_3d = output[0, self.TEMPORAL_WINDOW // 2].cpu().numpy()
        else:
            # Single frame output
            pose_3d = output[0].cpu().numpy()

        return pose_3d

    def reset(self):
        """Reset temporal buffer."""
        self.temporal_buffer.clear()


# Standalone function for quick inference
def extract_3d_poses(
    poses_2d: np.ndarray,
    model_path: Path | str,
    model_type: str = "motionagformer-s",
    device: str = "auto",
) -> np.ndarray:
    """Extract 3D poses from 2D pose sequence.

    Convenience function that creates extractor and runs inference.

    Args:
        poses_2d: (N, 17, 2) or (N, 33, 2) array
            - If 33 keypoints: BlazePose format (auto-converted)
            - If 17 keypoints: H3.6M format (direct)
        model_path: Path to model checkpoint
        model_type: Model architecture type
        device: "cuda", "cpu", or "auto"

    Returns:
        poses_3d: (N, 17, 3) array with x, y, z coordinates
    """
    # Convert BlazePose to H3.6M if needed
    if poses_2d.shape[1] == 33:
        from .blazepose_to_h36m import blazepose_to_h36m

        poses_2d = blazepose_to_h36m(poses_2d)

    # Create extractor and run
    extractor = AthletePose3DExtractor(model_path, device, model_type)
    return extractor.extract_sequence(poses_2d)
