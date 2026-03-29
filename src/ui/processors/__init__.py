"""Процессоры для обработки данных.

Data processing pipeline modules.
"""

from .base import Processor
from .frame_renderer import FrameRenderer
from .pose_processor import PoseProcessor
from .video_exporter import VideoExporter

__all__ = ["FrameRenderer", "PoseProcessor", "Processor", "VideoExporter"]
