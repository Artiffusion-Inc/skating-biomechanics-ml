"""Visualization layer system.

Provides modular layer-based rendering:
- Base layer class
- Skeleton overlay layer
- Velocity vectors layer
- Motion trails layer
- Blade indicator layer
- HUD layer
- Vertical axis layer
- Joint angle layer
- Timer layer
- Angle panel layer
"""

from src.visualization.layers.angle_panel_layer import AnglePanelLayer
from src.visualization.layers.base import Layer, LayerContext, render_layers
from src.visualization.layers.blade_layer import BladeLayer
from src.visualization.layers.depth_layer import DepthMapLayer
from src.visualization.layers.hud_layer import HUDLayer
from src.visualization.layers.joint_angle_layer import JointAngleLayer
from src.visualization.layers.optical_flow_layer import OpticalFlowLayer
from src.visualization.layers.segmentation_layer import SegmentationMaskLayer
from src.visualization.layers.skeleton_layer import SkeletonLayer
from src.visualization.layers.timer_layer import TimerLayer
from src.visualization.layers.trail_layer import TrailLayer
from src.visualization.layers.velocity_layer import VelocityLayer
from src.visualization.layers.vertical_axis_layer import VerticalAxisLayer

__all__ = [
    "AnglePanelLayer",
    "BladeLayer",
    "DepthMapLayer",
    "HUDLayer",
    "JointAngleLayer",
    "Layer",
    "LayerContext",
    "OpticalFlowLayer",
    "SegmentationMaskLayer",
    "SkeletonLayer",
    "TimerLayer",
    "TrailLayer",
    "VelocityLayer",
    "VerticalAxisLayer",
    "render_layers",
]
