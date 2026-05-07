"""Temporal Action Segmentation (TAS) for figure skating.

Coarse segmentation: Jump / Spin / Step / None.
Fine classifier: Random Forest on extracted segment features.
"""

from .dataset import MCFSCoarseDataset, coarse_label, normalize_poses, op25_to_coco17, pad_collate

__all__ = [
    "MCFSCoarseDataset",
    "coarse_label",
    "normalize_poses",
    "op25_to_coco17",
    "pad_collate",
]
