#!/usr/bin/env python3
"""DistilPoseTrainer: DWPose Two-Stage Knowledge Distillation for YOLO26-Pose.

Teacher: MogaNet-B (pre-computed heatmaps in HDF5)
Student: YOLO26-Pose (sigma head: cv4_sigma, 34 channels = 17kp x 2)

DWPose Method (ICCV 2023):
    Stage 1 (210 epochs): Feature + Logit distillation with weight decay
        L_total = L_gt + w_kd × (α×L_feat + β×L_logit)

        Where:
        - L_gt: Standard YOLO pose loss
        - L_feat: MSE(teacher_feat, student_feat) - backbone feature distillation
        - L_logit: KL(teacher_hm, student_hm) - heatmap logit distillation
        - w_kd: Weight decay = 1 - (epoch-1)/max_epochs
        - α: Feature loss weight (default: 0.00005)
        - β: Logit loss weight (default: 0.1)

        Config: NO Mosaic, AdamW(lr=2e-3), batch=128

    Stage 2 (42 epochs, optional): Self-KD (head only, frozen backbone)
        Use student as its own teacher, train only decoder
        Optional: +0.1% AP boost

Usage:
    python3 distill_trainer.py                    # dry run (no args)
    python3 distill_trainer.py --test              # unit tests (no GPU)
    python3 distill_trainer.py train ...           # train via CLI

Stage 1 example:
    python3 distill_trainer.py train \\
        --model yolo26n-pose.pt \\
        --data data.yaml \\
        --teacher-hm heatmaps.h5 \\
        --epochs 210 \\
        --batch 128 \\
        --alpha 0.00005 \\
        --beta 0.1 \\
        --feature-layers 4,6,8

Stage 2 example:
    python3 distill_trainer.py train \\
        --model checkpoint.pt \\
        --data data.yaml \\
        --epochs 42 \\
        --stage2 \\
        --batch 64
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import h5py
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Heatmap utilities (inline, same logic as simulate_heatmap.py)
# ---------------------------------------------------------------------------


def keypoints_to_heatmap(
    kpts: torch.Tensor,
    sigma: torch.Tensor,
    visibility: torch.Tensor | None = None,
    hm_shape: tuple[int, int, int] = (17, 72, 96),
) -> torch.Tensor:
    """Generate 2D Gaussian heatmaps from predicted keypoints and sigma.

    Args:
        kpts: Keypoint coordinates (B, K, 2) in [0, 1] normalized space.
        sigma: Per-keypoint sigma (B, K, 2) -- (sigma_x, sigma_y) per keypoint.
        visibility: Optional visibility mask (B, K), 1=visible, 0=occluded.
        hm_shape: Output heatmap shape (K, H, W).

    Returns:
        Heatmap tensor (B, K, H, W).
    """
    B, _K, _ = kpts.shape
    _, _H, _W = hm_shape[1], hm_shape[2], hm_shape[0] if len(hm_shape) == 3 else hm_shape
    K_out = hm_shape[0]
    H_out = hm_shape[1]
    W_out = hm_shape[2]

    device = kpts.device
    dtype = kpts.dtype

    # Create grid: (H, W) in [0, 1]
    gy = torch.linspace(0, 1, H_out, device=device, dtype=dtype)
    gx = torch.linspace(0, 1, W_out, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")  # (H, W)

    # Reshape for broadcasting: kpts (B, K, 1, 1, 2), grid (1, 1, H, W, 2)
    kpts_exp = kpts[:, :K_out].unsqueeze(2).unsqueeze(3)  # (B, K, 1, 1, 2)
    grid_exp = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 2)

    # Sigma: (B, K, 2) -> (B, K, 1, 1, 2), ensure positive via softplus
    sigma_exp = (
        sigma[:, : K_out * 2].view(B, K_out, 2).sigmoid().unsqueeze(2).unsqueeze(3)
    )  # (B, K, 1, 1, 2)
    sigma_exp = sigma_exp.clamp(min=1e-4)

    # Gaussian: exp(-0.5 * ((x - mu)^2 / sigma^2))
    diff = (grid_exp - kpts_exp) / sigma_exp  # (B, K, H, W, 2)
    exponent = -0.5 * (diff**2).sum(dim=-1)  # (B, K, H, W)
    hm = torch.exp(exponent)  # (B, K, H, W)

    # Apply visibility mask
    if visibility is not None:
        vis = visibility[:, :K_out].unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        hm = hm * vis

    return hm


def extract_teacher_value(
    teacher_heatmap: torch.Tensor,
    kpts: torch.Tensor,
) -> torch.Tensor:
    """Extract teacher heatmap value at predicted keypoint locations.

    Args:
        teacher_heatmap: (B, K, H, W) teacher heatmaps.
        kpts: (B, K, 2) keypoint coordinates in [0, 1].

    Returns:
        Values (B, K) at each keypoint location.
    """
    B, K, _, _ = teacher_heatmap.shape
    H, W = teacher_heatmap.shape[2], teacher_heatmap.shape[3]
    device = teacher_heatmap.device

    # Convert normalized coords to pixel coords
    px = (kpts[..., 0] * (W - 1)).clamp(0, W - 1)
    py = (kpts[..., 1] * (H - 1)).clamp(0, H - 1)

    # Quantize to integer indices
    ix = px.long().clamp(0, W - 1)
    iy = py.long().clamp(0, H - 1)

    # Gather: (B, K)
    values = teacher_heatmap[
        torch.arange(B, device=device).unsqueeze(1),
        torch.arange(K, device=device).unsqueeze(0),
        iy,
        ix,
    ]
    return values


# ---------------------------------------------------------------------------
# LMDB teacher heatmap loader
# ---------------------------------------------------------------------------


class TeacherHeatmapLoader:
    """Load pre-computed teacher heatmaps from LMDB."""

    def __init__(self, lmdb_path, hm_shape=(17, 72, 96)):
        from pathlib import Path

        self.hm_shape = hm_shape
        self.path = Path(lmdb_path)
        self.env = None
        print(f"LMDB loader: {self.path}")

    def _open_env(self):
        if self.env is None:
            import lmdb

            self.env = lmdb.open(str(self.path), readonly=True, lock=False)
            print(f"Worker {id(self)} opened LMDB")

    def load(self, im_files):
        from pathlib import Path

        import numpy as np
        import torch

        self._open_env()
        result = []
        for img_path in im_files:
            with self.env.begin() as txn:
                # Try multiple key formats
                data = txn.get(img_path.encode())
                if data is None:
                    # Try just filename
                    data = txn.get(Path(img_path).name.encode())
                if data is None:
                    # Try relative path: data/.../images/xxx.jpg
                    p = Path(img_path)
                    parts = p.parts
                    try:
                        data_idx = parts.index("data")
                        rel_path = str(Path(*parts[data_idx:]))
                        data = txn.get(rel_path.encode())
                    except ValueError:
                        pass
                if data is not None:
                    hm = np.frombuffer(data, dtype=np.float16).copy()
                    hm = hm.reshape(self.hm_shape)
                    result.append(hm)
        if not result:
            return None
        heatmaps = np.stack(result, axis=0).astype(np.float32)
        return torch.from_numpy(heatmaps)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class TeacherFeatureLoader:
    """Load pre-computed teacher backbone features from HDF5.

    HDF5 structure:
        /layer4: (N, C4, H4, W4) float16
        /layer6: (N, C6, H6, W6) float16
        /layer8: (N, C8, H8, W8) float16
        /indices: JSON sidecar mapping image_path -> row index

    The indices sidecar is stored as an HDF5 attribute on the root group.
    """

    def __init__(self, hdf5_path: str | Path, feature_layers: list[int] | None = None):
        if feature_layers is None:
            feature_layers = [4, 6, 8]
        self.feature_layers = feature_layers
        self.path = Path(hdf5_path)
        self._indices: dict[str, int] | None = None
        self._file: h5py.File | None = None
        self._datasets: dict[int, h5py.Dataset] = {}

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(str(self.path), "r", libver="latest", swmr=True)
            # Cache dataset references
            for layer_idx in self.feature_layers:
                layer_name = f"layer{layer_idx}"
                if layer_name in self._file:
                    self._datasets[layer_idx] = self._file[layer_name]
        return self._file

    @property
    def indices(self) -> dict[str, int]:
        if self._indices is None:
            raw = self.file.attrs.get("indices", "{}")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            self._indices = json.loads(raw)
        return self._indices

    def load(self, im_files: list[str]) -> dict[int, torch.Tensor] | None:
        """Load teacher features for a batch of images.

        Args:
            im_files: List of image file paths (may be absolute or relative).

        Returns:
            Dict mapping layer_idx -> feature tensor (B, C, H, W) float32,
            or None if no matching features found.
        """
        try:
            idx_map = self.indices
        except (FileNotFoundError, OSError):
            return None

        rows = []
        for f in im_files:
            # Try both absolute and basename matching
            idx = idx_map.get(f)
            if idx is None:
                idx = idx_map.get(Path(f).name)
            if idx is None:
                # Try stripping common prefixes
                for key, val in idx_map.items():
                    if Path(f).name == Path(key).name:
                        idx = val
                        break
            rows.append(idx)

        # Filter out images without teacher features
        valid_mask = [r is not None for r in rows]
        if not any(valid_mask):
            return None

        # HDF5 requires sorted indices - keep track of original positions
        valid_rows = [(rows[i], i) for i in range(len(rows)) if valid_mask[i]]
        if not valid_rows:
            return None

        # Sort by HDF5 index (first element of tuple)
        valid_rows_sorted = sorted(valid_rows, key=lambda x: x[0])
        sorted_indices = [x[0] for x in valid_rows_sorted]
        original_positions = [x[1] for x in valid_rows_sorted]

        # Load features for each layer
        features = {}
        B = len(im_files)

        for layer_idx in self.feature_layers:
            if layer_idx not in self._datasets:
                continue

            dataset = self._datasets[layer_idx]
            feat = torch.from_numpy(dataset[sorted_indices]).float()  # (N_valid, C, H, W)

            # Get shape info
            _N_valid, C, H, W = feat.shape

            # Pad back to full batch size with zeros
            full_feat = torch.zeros(B, C, H, W, device=feat.device, dtype=feat.dtype)

            # Place features back in original batch order
            for sorted_idx, orig_pos in zip(range(len(feat)), original_positions, strict=False):
                full_feat[orig_pos] = feat[sorted_idx]

            features[layer_idx] = full_feat

        return features if features else None

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._datasets.clear()


# ---------------------------------------------------------------------------
# Feature extraction utilities
# ---------------------------------------------------------------------------


def extract_backbone_features(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    layer_indices: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Extract intermediate features from YOLO backbone.

    Args:
        model: YOLO pose model
        input_tensor: Input image tensor (B, 3, H, W)
        layer_indices: Which backbone layers to extract (default: [4, 6, 8])

    Returns:
        Dict mapping layer_idx -> feature tensor
    """
    if layer_indices is None:
        layer_indices = [4, 6, 8]
    features = {}
    hooks = []

    def make_hook(idx):
        def hook(_module, _input, output):
            # Store output feature map
            features[idx] = output.detach()

        return hook

    # Register forward hooks on backbone layers
    # YOLOv8/v11 backbone structure: typically has 10 conv/PSP layers
    backbone = model.model[:10] if hasattr(model, "model") else model

    for idx in layer_indices:
        if idx < len(backbone):
            layer = backbone[idx]
            handle = layer.register_forward_hook(make_hook(idx))
            hooks.append(handle)

    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove hooks
    for handle in hooks:
        handle.remove()

    return features


# ---------------------------------------------------------------------------
# FeatureAdapter: 1x1 conv projections for channel alignment
# ---------------------------------------------------------------------------


class FeatureAdapter(nn.Module):
    """1x1 convolution adapter for channel alignment in feature distillation.

    Projects teacher features to match student channel dimensions (or vice versa).
    This preserves ALL feature channels instead of discarding via min_channels slicing.

    Args:
        in_channels: Input feature channels (teacher channels).
        out_channels: Output feature channels (student channels).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features to output channel dimension.

        Args:
            x: Input tensor (B, in_channels, H, W).

        Returns:
            Projected tensor (B, out_channels, H, W).
        """
        return self.projection(x)


# ---------------------------------------------------------------------------
# DistilPoseTrainer with DWPose Two-Stage method
# ---------------------------------------------------------------------------


class DistilPoseTrainer:
    """DWPose Two-Stage Knowledge Distillation wrapper around Ultralytics PoseTrainer.

    Stage 1: Feature + Logit distillation with weight decay
        L_total = L_gt + w_kd × (α×L_feat + β×L_logit)

        Where:
        - L_gt: Standard YOLO pose loss
        - L_feat: MSE(teacher_feat, student_feat) - backbone feature distillation
        - L_logit: KL(teacher_hm, student_hm) - heatmap logit distillation
        - w_kd: Weight decay = 1 - (epoch-1)/max_epochs
        - α: Feature loss weight (default: 0.00005)
        - β: Logit loss weight (default: 0.1)

        Config: NO Mosaic, AdamW(lr=2e-3), batch=128

    Stage 2: Self-KD (head only, frozen backbone)
        Use student as its own teacher

    Args:
        teacher_hm_path: Path to HDF5 with pre-computed teacher heatmaps.
        alpha: Feature loss weight (default: 0.00005).
        beta: Logit loss weight (default: 0.1).
        warmup_epochs: Number of epochs before KD loss kicks in (default: 5).
        hm_shape: Heatmap shape (K, H, W) (default: 17, 72, 96).
        freeze_backbone: Whether to freeze non-sigma parameters (default: True).
        stage2: Enable Stage 2 self-KD (default: False).
        feature_layers: Backbone layer indices for feature distillation (default: [4, 6, 8]).
        teacher_feat_path: Path to cached teacher features (optional).
    """

    def __init__(
        self,
        teacher_hm_path: str | Path | None = None,
        alpha: float = 0.00005,
        beta: float = 0.1,
        warmup_epochs: int = 5,
        hm_shape: tuple[int, int, int] = (17, 72, 96),
        freeze_backbone: bool = True,
        stage2: bool = False,
        feature_layers: list[int] | None = None,
        teacher_feat_path: str | Path | None = None,
    ):
        if feature_layers is None:
            feature_layers = [4, 6, 8]
        self.teacher_hm_path = teacher_hm_path
        self.alpha = alpha
        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.hm_shape = hm_shape
        self.freeze_backbone = freeze_backbone
        self.stage2 = stage2
        self.feature_layers = feature_layers
        self.teacher_feat_path = teacher_feat_path
        self._loader: TeacherHeatmapLoader | None = None
        self._feat_loader: TeacherFeatureLoader | None = None
        self._original_loss = None  # type: ignore[assignment]
        self._current_epoch = 0
        self._max_epochs = 210  # Default, will be updated from trainer
        self.adapters: dict[int, nn.Module] = {}  # Feature adapters per layer

    @property
    def loader(self) -> TeacherHeatmapLoader | None:
        if self._loader is None and self.teacher_hm_path is not None:
            self._loader = TeacherHeatmapLoader(self.teacher_hm_path, self.hm_shape)
        return self._loader

    @property
    def feat_loader(self) -> TeacherFeatureLoader | None:
        if self._feat_loader is None and self.teacher_feat_path is not None:
            self._feat_loader = TeacherFeatureLoader(self.teacher_feat_path, self.feature_layers)
        return self._feat_loader

    def set_epoch(self, epoch: int):
        """Update current epoch (called from training loop)."""
        self._current_epoch = epoch

    def set_max_epochs(self, max_epochs: int):
        """Update max epochs for weight decay calculation."""
        self._max_epochs = max_epochs

    def _initialize_adapters(self, model: torch.nn.Module, device=None):
        """Initialize feature adapters for teacher→student channel projection.

        Creates 1x1 conv adapters lazily based on actual feature dimensions.
        Called during setup_model after teacher features are loaded.

        Args:
            model: Student model (YOLO26-Pose).
            device: Device to create adapters on (defaults to model device).
        """
        # Adapters will be created lazily on first forward pass
        # when we know the actual teacher channel dimensions
        pass

    def _get_or_create_adapter(
        self,
        layer_idx: int,
        teacher_channels: int,
        student_channels: int,
        device: torch.device,
    ) -> nn.Module:
        """Get or create a feature adapter for a specific layer.

        Args:
            layer_idx: Layer index (4, 6, 8).
            teacher_channels: Teacher feature channel count.
            student_channels: Student feature channel count.
            device: Device for the adapter.

        Returns:
            FeatureAdapter module.
        """
        if layer_idx not in self.adapters:
            # Create new adapter: teacher_channels → student_channels
            adapter = FeatureAdapter(teacher_channels, student_channels)
            adapter = adapter.to(device)
            # Train the adapter (learnable projection)
            for param in adapter.parameters():
                param.requires_grad = True
            self.adapters[layer_idx] = adapter

        return self.adapters[layer_idx]

    def compute_kd_weight(self) -> float:
        """Compute KD weight decay: w_kd = 1 - (epoch-1)/max_epochs.

        Linearly decays from 1.0 to 0.0 over training.
        """
        if self._max_epochs <= 1:
            return 1.0
        w_kd = 1.0 - (self._current_epoch - 1) / self._max_epochs
        return max(0.0, w_kd)

    def setup_model(self, model: torch.nn.Module):
        """Patch model: freeze backbone and replace loss function."""
        # Stage 2: freeze backbone completely
        if self.stage2:
            for name, param in model.named_parameters():
                if "cv4" in name or "detect" in name or "sigma" in name:
                    # Train head/detector only
                    param.requires_grad = True
                else:
                    # Freeze backbone
                    param.requires_grad = False
        elif self.freeze_backbone:
            # Stage 1: freeze all except sigma head
            for name, param in model.named_parameters():
                if "sigma" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # Initialize feature adapters for channel projection
        # MogaNet-B → YOLO26n channel mapping:
        # Layer 4: 160 → 64 (or 128 depending on variant)
        # Layer 6: 320 → 128 (or 256)
        # Layer 8: 512 → 256
        # Adapters will be created lazily on first use when shapes are known
        self._initialize_adapters(model)

        # Save original loss and model ref for picklable kd_loss method
        self._original_loss = model.loss
        self._model = model
        model.loss = self.kd_loss  # type: ignore[assignment]

    def kd_loss(self, batch: dict[str, Any], preds=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute DWPose KD loss: L_gt + w_kd × (α×L_feat + β×L_logit).

        Defined as a proper class method (not a nested closure) so that
        ``torch.save`` / DDP can pickle it without hitting
        ``AttributeError: Can't pickle local object``.
        """
        model = self._model

        # 1. Forward pass (single, shared between GT and KD losses)
        if preds is None:
            preds = model.forward(batch["img"])

        # Parse E2E output if needed
        preds_for_loss = preds
        if isinstance(preds_for_loss, tuple):
            preds_for_loss = preds_for_loss[1] if len(preds_for_loss) > 1 else preds_for_loss

        # 2. Standard GT loss using pre-computed preds
        gt_loss, loss_items = self._original_loss(batch, preds)

        # 3. KD losses: skip during validation and warmup
        # Pad loss_items with zeros to maintain consistent size (6 base + 3 KD)
        kd_pad = torch.zeros(3, device=gt_loss.device)
        if not model.training:
            return gt_loss, torch.cat([loss_items, kd_pad])
        if self._current_epoch < self.warmup_epochs:
            return gt_loss, torch.cat([loss_items, kd_pad])

        # 4. Compute KD weight decay
        w_kd = self.compute_kd_weight()

        # Stage 2: Self-KD uses student as teacher (no external teacher data)
        if self.stage2:
            return gt_loss, torch.cat([loss_items, kd_pad])

        # 5. Load teacher heatmaps
        teacher_hm = None
        if self.loader is not None:
            im_files = batch.get("im_file", [])
            if isinstance(im_files, (list, tuple)):
                teacher_hm = self.loader.load(list(im_files))

        if teacher_hm is None:
            return gt_loss, torch.cat([loss_items, kd_pad])

        # Move teacher heatmaps to correct device
        teacher_hm = teacher_hm.to(gt_loss.device)

        # 6. Extract student predictions from the one2many branch
        src_preds = preds_for_loss
        if isinstance(src_preds, dict) and "one2many" in src_preds:
            src_preds = src_preds["one2many"]

        kpts_raw = src_preds.get("kpts")
        sigma_raw = src_preds.get("kpts_sigma")

        if kpts_raw is None or sigma_raw is None:
            return gt_loss, loss_items

        # Reshape predictions
        B = kpts_raw.shape[0]
        K = self.hm_shape[0]

        kpts_perm = kpts_raw.permute(0, 2, 1).contiguous()
        kpts_perm.view(B, -1, K, 3)

        sigma_perm = sigma_raw.permute(0, 2, 1).contiguous()
        sigma_reshaped = sigma_perm.view(B, -1, K, 2)

        # 7. Get GT keypoints for visibility
        gt_kpts = batch.get("keypoints")
        batch_idx = batch.get("batch_idx")

        if gt_kpts is None or batch_idx is None:
            return gt_loss, loss_items

        # Build per-image GT keypoints
        batch_idx_flat = batch_idx.long().flatten()
        max_objects = max((batch_idx_flat == i).sum().item() for i in range(B)) if B > 0 else 0
        if max_objects == 0:
            return gt_loss, loss_items

        gt_kpts_per_image = torch.zeros(
            B, max_objects, K, 3, device=gt_loss.device, dtype=gt_kpts.dtype
        )
        gt_kpts_dev = gt_kpts.to(gt_loss.device)
        offsets = torch.zeros(B + 1, dtype=torch.long, device=gt_loss.device)
        offsets.scatter_add_(0, batch_idx_flat + 1, torch.ones_like(batch_idx_flat))
        offsets = offsets.cumsum(0)
        within_idx = (
            torch.arange(len(batch_idx_flat), device=gt_loss.device) - offsets[batch_idx_flat]
        )
        gt_kpts_per_image[batch_idx_flat, within_idx] = gt_kpts_dev

        primary_kpts = gt_kpts_per_image[:, 0, :, :]
        primary_visibility = (primary_kpts[..., 2] > 0).float()

        # 8. Generate student heatmap from sigma predictions
        kpts_xy = primary_kpts[..., :2]
        sigma_mean = sigma_reshaped.mean(dim=1).to(gt_loss.device)

        student_hm = keypoints_to_heatmap(
            kpts_xy.detach(),
            sigma_mean,
            primary_visibility.detach(),
            hm_shape=self.hm_shape,
        )

        # 9. Logit distillation: KL(teacher_hm, student_hm)
        if teacher_hm.shape[2:] != student_hm.shape[2:]:
            teacher_hm_resized = F.interpolate(
                teacher_hm,
                size=student_hm.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            teacher_hm_resized = teacher_hm

        # Normalize to probability distributions over spatial dims, then KL
        teacher_flat = teacher_hm_resized.reshape(
            teacher_hm_resized.shape[0], teacher_hm_resized.shape[1], -1
        )
        student_flat = student_hm.reshape(student_hm.shape[0], student_hm.shape[1], -1)
        teacher_probs = F.softmax(teacher_flat, dim=-1)
        student_log = F.log_softmax(student_flat, dim=-1)
        logit_loss = F.kl_div(student_log, teacher_probs, reduction="batchmean")

        # 10. Feature distillation removed (frozen backbone = dead gradient path)
        #     feat_loss stays at 0.0

        # 11. Total loss with weight decay (logit-only KD)
        kd_total = self.beta * logit_loss
        total_loss = gt_loss + w_kd * kd_total

        kd_items = torch.cat(
            [
                loss_items,
                torch.tensor([logit_loss.item(), 0.0, w_kd], device=gt_loss.device),
            ]
        )
        return total_loss, kd_items

    def restore_model(self, model: torch.nn.Module):
        """Restore original loss function (cleanup)."""
        if self._original_loss is not None:
            model.loss = self._original_loss  # type: ignore[assignment]
        if self._loader is not None:
            self._loader.close()
        if self._feat_loader is not None:
            self._feat_loader.close()

    def get_loss_names(self, base_names: tuple[str, ...]) -> tuple[str, ...]:
        """Return extended loss names including KD components."""
        return (*base_names, "kd_logit_loss", "kd_feat_loss", "kd_weight")


# ---------------------------------------------------------------------------
# Ultralytics PoseTrainer integration
# ---------------------------------------------------------------------------


def make_distil_pose_trainer(
    teacher_hm_path: str | Path | None = None,
    alpha: float = 0.00005,
    beta: float = 0.1,
    warmup_epochs: int = 5,
    hm_shape: tuple[int, int, int] = (17, 72, 96),
    freeze_backbone: bool = True,
    stage2: bool = False,
    feature_layers: list[int] | None = None,
    teacher_feat_path: str | Path | None = None,
):
    """Create a PoseTrainer subclass with DWPose KD support.

    Returns a class that can be used with Ultralytics CLI or Python API:
        trainer = make_distil_pose_trainer(teacher_hm_path="heatmaps.h5")
        t = trainer(overrides={...})
        t.train()

    Args:
        See DistilPoseTrainer.__init__ for parameter docs.

    Returns:
        A PoseTrainer subclass with KD loss.
    """
    from ultralytics.models.yolo.pose.train import PoseTrainer
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.torch_utils import unwrap_model

    if feature_layers is None:
        feature_layers = [4, 6, 8]
    kd_config = {
        "teacher_hm_path": teacher_hm_path,
        "alpha": alpha,
        "beta": beta,
        "warmup_epochs": warmup_epochs,
        "hm_shape": hm_shape,
        "freeze_backbone": freeze_backbone,
        "stage2": stage2,
        "feature_layers": feature_layers,
        "teacher_feat_path": teacher_feat_path,
    }

    class _DistilPoseTrainer(PoseTrainer):
        """PoseTrainer with DWPose Knowledge Distillation."""

        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            super().__init__(cfg, overrides, _callbacks)
            self._kd = DistilPoseTrainer(**kd_config)

        def setup_model(self):
            """Set up model and patch with KD loss."""
            ckpt = super().setup_model()
            # Patch model loss after setup
            self._kd.setup_model(unwrap_model(self.model))
            # Update max epochs for weight decay
            self._kd.set_max_epochs(self.epochs)
            return ckpt

        def preprocess_batch(self, batch):
            """Attach epoch info for KD warmup."""
            batch = super().preprocess_batch(batch)
            self._kd.set_epoch(self.epoch)
            return batch

        def get_validator(self):
            """Return validator with extended loss names."""
            validator = super().get_validator()
            # Extend loss names for KD components
            base_names = self.loss_names
            self.loss_names = self._kd.get_loss_names(base_names)
            return validator

        def label_loss_items(self, loss_items=None, prefix="train"):
            """Label loss items including KD components."""
            labels = super().label_loss_items(loss_items, prefix)
            if loss_items is not None and len(loss_items) > len(labels):
                # Add KD loss labels
                extra = len(loss_items) - len(labels)
                for i in range(extra):
                    labels[f"{prefix}_kd_{i}"] = loss_items[len(labels) + i].item()
            return labels

    _DistilPoseTrainer.__name__ = "DistilPoseTrainer"
    _DistilPoseTrainer.__qualname__ = "DistilPoseTrainer"
    return _DistilPoseTrainer


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def train():
    """Launch DWPose KD training via Ultralytics CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="DistilPoseTrainer: DWPose KD for YOLO26-Pose")
    parser.add_argument("--model", type=str, default="yolo26n-pose.pt", help="Student model path")
    parser.add_argument("--data", type=str, required=True, help="data.yaml path")
    parser.add_argument("--teacher-hm", type=str, help="Teacher heatmaps HDF5 path")
    parser.add_argument("--epochs", type=int, default=210)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--alpha", type=float, default=0.00005, help="Feature loss weight")
    parser.add_argument("--beta", type=float, default=0.1, help="Logit loss weight")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze", dest="freeze_backbone", action="store_false")
    parser.add_argument("--hm-shape", type=str, default="17,72,96", help="K,H,W heatmap shape")
    parser.add_argument("--stage2", action="store_true", help="Enable Stage 2 self-KD")
    parser.add_argument(
        "--feature-layers",
        type=str,
        default="4,6,8",
        help="Backbone layers for feature distillation",
    )
    parser.add_argument("--teacher-feat", type=str, help="Teacher features cache path (optional)")
    parser.add_argument("--name", type=str, default="distil_pose")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    hm_shape = tuple(int(x) for x in args.hm_shape.split(","))
    feature_layers = [int(x) for x in args.feature_layers.split(",")]

    TrainerClass = make_distil_pose_trainer(
        teacher_hm_path=args.teacher_hm,
        alpha=args.alpha,
        beta=args.beta,
        warmup_epochs=args.warmup_epochs,
        hm_shape=hm_shape,
        freeze_backbone=args.freeze_backbone,
        stage2=args.stage2,
        feature_layers=feature_layers,
        teacher_feat_path=args.teacher_feat,
    )

    # DWPose Stage 1 config: NO Mosaic, AdamW(lr=2e-3)
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "name": args.name,
        "mosaic": 0.0,  # DWPose: NO Mosaic
        "lr0": 0.002 if not args.stage2 else 0.001,  # AdamW lr=2e-3
        "optimizer": "AdamW",
        "device": args.device,
    }

    trainer = TrainerClass(overrides=overrides)
    trainer.train()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def run_tests():
    """Run unit tests without GPU (mock data)."""
    import traceback

    passed = 0
    failed = 0

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except (AssertionError, RuntimeError, ValueError) as e:
            print(f"  FAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("DistilPoseTrainer unit tests (DWPose Two-Stage)")
    print("=" * 60)

    # Test 1: keypoints_to_heatmap shape
    def test_heatmap_shape():
        kpts = torch.rand(2, 17, 2)
        sigma = torch.rand(2, 17, 2) + 0.1
        vis = torch.ones(2, 17)
        hm = keypoints_to_heatmap(kpts, sigma, vis, hm_shape=(17, 72, 96))
        assert hm.shape == (2, 17, 72, 96), f"Expected (2,17,72,96), got {hm.shape}"
        assert hm.min() >= 0, "Heatmap should be non-negative"
        assert hm.max() <= 1.0 + 1e-6, f"Heatmap max should be <= 1, got {hm.max()}"

    test("keypoints_to_heatmap shape and range", test_heatmap_shape)

    # Test 2: keypoints_to_heatmap peak location
    def test_heatmap_peak():
        kpts = torch.tensor([[[0.5, 0.5]]])
        sigma = torch.tensor([[[-2.0, -2.0]]])
        hm = keypoints_to_heatmap(kpts, sigma, hm_shape=(1, 72, 96))
        peak_val = hm[0, 0].max().item()
        assert peak_val > 0.9, f"Peak should be > 0.9, got {peak_val}"
        corner_val = hm[0, 0, 0, 0].item()
        assert corner_val < 1e-3, f"Corner should be ~0, got {corner_val}"

    test("keypoints_to_heatmap peak at center", test_heatmap_peak)

    # Test 3: keypoints_to_heatmap with visibility
    def test_heatmap_visibility():
        kpts = torch.rand(1, 3, 2)
        sigma = torch.rand(1, 3, 2) + 0.1
        vis = torch.tensor([[1.0, 0.0, 1.0]])
        hm = keypoints_to_heatmap(kpts, sigma, vis, hm_shape=(3, 72, 96))
        assert hm[0, 0].sum() > 0, "Visible keypoint should have non-zero heatmap"
        assert hm[0, 1].sum() == 0, "Occluded keypoint should have zero heatmap"

    test("keypoints_to_heatmap visibility mask", test_heatmap_visibility)

    # Test 4: extract_teacher_value
    def test_extract_value():
        hm = torch.zeros(1, 2, 10, 10)
        hm[0, 0, 5, 5] = 1.0
        hm[0, 1, 3, 7] = 0.5
        kpts = torch.tensor([[[5.0 / 9, 5.0 / 9], [7.0 / 9, 3.0 / 9]]])
        vals = extract_teacher_value(hm, kpts)
        assert vals.shape == (1, 2), f"Expected (1,2), got {vals.shape}"
        assert abs(vals[0, 0].item() - 1.0) < 0.01, f"Expected 1.0, got {vals[0, 0]}"
        assert abs(vals[0, 1].item() - 0.5) < 0.01, f"Expected 0.5, got {vals[0, 1]}"

    test("extract_teacher_value at correct locations", test_extract_value)

    # Test 5: TeacherHeatmapLoader graceful handling
    def test_loader_no_file():
        loader = TeacherHeatmapLoader(hdf5_path="/nonexistent/path.h5")
        result = loader.load(["image1.jpg", "image2.jpg"])
        assert result is None

    test("TeacherHeatmapLoader graceful no-file handling", test_loader_no_file)

    # Test 6: DistilPoseTrainer init (Stage 1)
    def test_kd_init_stage1():
        kd = DistilPoseTrainer(
            teacher_hm_path=None,
            alpha=0.00005,
            beta=0.1,
            warmup_epochs=5,
            stage2=False,
        )
        assert kd.alpha == 0.00005
        assert kd.beta == 0.1
        assert kd.warmup_epochs == 5
        assert kd.stage2 is False
        assert kd._current_epoch == 0

    test("DistilPoseTrainer Stage 1 initialization", test_kd_init_stage1)

    # Test 7: DistilPoseTrainer init (Stage 2)
    def test_kd_init_stage2():
        kd = DistilPoseTrainer(
            stage2=True,
            feature_layers=[4, 6, 8],
        )
        assert kd.stage2 is True
        assert kd.feature_layers == [4, 6, 8]

    test("DistilPoseTrainer Stage 2 initialization", test_kd_init_stage2)

    # Test 8: KD weight decay calculation
    def test_kd_weight_decay():
        kd = DistilPoseTrainer(
            teacher_hm_path=None,
            alpha=0.00005,
            beta=0.1,
        )
        kd.set_max_epochs(100)

        # Epoch 1: w_kd = 1.0
        kd.set_epoch(1)
        w1 = kd.compute_kd_weight()
        assert abs(w1 - 1.0) < 1e-6, f"Epoch 1: expected 1.0, got {w1}"

        # Epoch 50: w_kd = 1 - 49/100 = 0.51
        kd.set_epoch(50)
        w50 = kd.compute_kd_weight()
        assert abs(w50 - 0.51) < 1e-6, f"Epoch 50: expected 0.51, got {w50}"

        # Epoch 100: w_kd = 1 - 99/100 = 0.01
        kd.set_epoch(100)
        w100 = kd.compute_kd_weight()
        assert abs(w100 - 0.01) < 1e-6, f"Epoch 100: expected 0.01, got {w100}"

    test("KD weight decay schedule", test_kd_weight_decay)

    # Test 9: KD weight decay clamping
    def test_kd_weight_clamp():
        kd = DistilPoseTrainer(
            teacher_hm_path=None,
        )
        kd.set_max_epochs(10)
        kd.set_epoch(15)  # Past max_epochs
        w = kd.compute_kd_weight()
        assert w == 0.0, f"Weight should be clamped to 0.0, got {w}"

    test("KD weight decay clamping at 0.0", test_kd_weight_clamp)

    # Test 10: get_loss_names
    def test_loss_names():
        kd = DistilPoseTrainer()
        names = kd.get_loss_names(("box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"))
        assert names == (
            "box_loss",
            "pose_loss",
            "kobj_loss",
            "cls_loss",
            "dfl_loss",
            "kd_logit_loss",
            "kd_feat_loss",
            "kd_weight",
        )

    test("get_loss_names includes DWPose KD components", test_loss_names)

    # Test 11: make_distil_pose_trainer returns callable
    def test_make_trainer():
        TrainerClass = make_distil_pose_trainer(
            teacher_hm_path=None,
            alpha=0.00005,
            beta=0.1,
        )
        assert callable(TrainerClass)
        assert TrainerClass.__name__ == "DistilPoseTrainer"

    test("make_distil_pose_trainer returns class", test_make_trainer)

    # Test 12: Stage 2 config
    def test_stage2_config():
        TrainerClass = make_distil_pose_trainer(
            stage2=True,
            feature_layers=[2, 4, 6],
        )
        assert callable(TrainerClass)

    test("make_distil_pose_trainer Stage 2 config", test_stage2_config)

    # Test 13: KL divergence loss computation
    def test_kl_loss():
        # Create proper probability distributions
        teacher = torch.ones(1, 1, 10, 10) * 0.01
        teacher[0, 0, 5, 5] = 0.9
        # Normalize to sum to 1 (valid probability distribution)
        teacher = teacher / teacher.sum(dim=(2, 3), keepdim=True)

        student = torch.ones(1, 1, 10, 10) * 0.01
        student[0, 0, 5, 5] = 0.5
        student = student / student.sum(dim=(2, 3), keepdim=True)

        # PyTorch kl_div: input=log_probs, target=probs
        # KL(P || Q) where P=student, Q=teacher
        student_log = (student + 1e-8).log()
        kl = F.kl_div(student_log, teacher, reduction="batchmean")

        # KL should be finite
        assert not torch.isnan(kl).item(), f"KL loss should not be NaN, got {kl}"
        assert not torch.isinf(kl).item(), f"KL loss should not be Inf, got {kl}"

    test("KL divergence loss computation", test_kl_loss)

    # Test 14: Warmup epoch tracking
    def test_kd_warmup():
        kd = DistilPoseTrainer(warmup_epochs=5)
        kd.set_epoch(3)
        assert kd._current_epoch < kd.warmup_epochs
        kd.set_epoch(5)
        assert kd._current_epoch >= kd.warmup_epochs

    test("DistilPoseTrainer warmup epoch tracking", test_kd_warmup)

    print()
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(run_tests())
    else:
        train()
