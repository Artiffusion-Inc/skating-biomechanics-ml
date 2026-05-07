"""TAS training with Hydra configs + MLflow tracking.

Usage:
    uv run python experiments/train_tas_hydra.py
    uv run python experiments/train_tas_hydra.py model=default training=default dataset=default
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

SRC = Path(__file__).parent.parent / "ml" / "src"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load stdlib types to prevent shadowing
import types as _stdlib_types
sys.modules["types"] = _stdlib_types

# Build fake package hierarchy so relative imports resolve
sys.modules["ml"] = _stdlib_types.ModuleType("ml")
sys.modules["ml.src"] = _stdlib_types.ModuleType("ml.src")
sys.modules["ml.src.pose_estimation"] = _stdlib_types.ModuleType("ml.src.pose_estimation")
sys.modules["ml.src.tas"] = _stdlib_types.ModuleType("ml.src.tas")

# Load modules in dependency order
h36m = _load_module("ml.src.pose_estimation.h36m", SRC / "pose_estimation" / "h36m.py")

# Also expose for any absolute imports
sys.modules["pose_estimation.h36m"] = h36m

# Load TAS modules
dataset_mod = _load_module("ml.src.tas.dataset", SRC / "tas" / "dataset.py")
model_mod = _load_module("ml.src.tas.model", SRC / "tas" / "model.py")
metrics_mod = _load_module("ml.src.tas.metrics", SRC / "tas" / "metrics.py")

MCFSCoarseDataset = dataset_mod.MCFSCoarseDataset
pad_collate = dataset_mod.pad_collate
BiGRUTAS = model_mod.BiGRUTAS
OverlapF1 = metrics_mod.OverlapF1


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for poses, labels, lengths in loader:
        poses, labels = poses.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(poses, lengths)
        loss = criterion(logits.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_fold(model, loader, device):
    model.eval()
    metric = OverlapF1(iou_threshold=0.5)
    all_preds = []
    all_true = []
    with torch.no_grad():
        for poses, labels, lengths in loader:
            poses = poses.to(device)
            logits = model(poses, lengths)
            preds = logits.argmax(dim=-1).cpu().numpy()
            for i, le in enumerate(lengths):
                all_preds.append(preds[i, :le])
                all_true.append(labels[i, :le].numpy())

    f1s = []
    for p, t in zip(all_preds, all_true):
        result = metric.compute(p, t)
        f1s.append(result["f1"])
    return {"f1": float(np.mean(f1s)), "precision": 0.0, "recall": 0.0}


@hydra.main(config_path="../ml/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # MLflow setup (SQLite backend to avoid file-store deprecation)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)

    ds = MCFSCoarseDataset(
        Path(cfg.dataset.features_dir), Path(cfg.dataset.labels_dir)
    )
    print(f"Dataset size: {len(ds)}")

    kf = KFold(
        n_splits=cfg.training.n_splits, shuffle=True, random_state=cfg.training.random_state
    )
    fold_results = []

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=cfg.experiment.run_name):
        mlflow.log_params({"model": dict(cfg.model), "training": dict(cfg.training)})
        mlflow.set_tag("git_commit", _get_git_commit())

        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(ds)))):
            print(f"\n--- Fold {fold + 1}/{cfg.training.n_splits} ---")

            with mlflow.start_run(run_name=f"fold-{fold+1}", nested=True):
                mlflow.log_param("fold", fold)

                train_ds = Subset(ds, train_idx)
                val_ds = Subset(ds, val_idx)

                train_loader = DataLoader(
                    train_ds,
                    batch_size=cfg.training.batch_size,
                    shuffle=True,
                    collate_fn=pad_collate,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    collate_fn=pad_collate,
                )

                model = BiGRUTAS(
                    hidden_dim=cfg.model.hidden_dim,
                    num_layers=cfg.model.num_layers,
                    dropout=cfg.model.dropout,
                ).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=cfg.training.lr
                )
                criterion = nn.CrossEntropyLoss(ignore_index=-1)

                best_f1 = 0.0
                for epoch in range(cfg.training.epochs):
                    train_loss = train_epoch(
                        model, train_loader, optimizer, criterion, device
                    )
                    val_result = eval_fold(model, val_loader, device)
                    print(
                        f"  Epoch {epoch + 1}: loss={train_loss:.4f}, val_f1={val_result['f1']:.4f}"
                    )

                    mlflow.log_metrics(
                        {"train_loss": train_loss, "val_f1": val_result["f1"]},
                        step=epoch,
                    )

                    if val_result["f1"] > best_f1:
                        best_f1 = val_result["f1"]
                        ckpt_path = ckpt_dir / f"fold_{fold}_best.pt"
                        torch.save(
                            {
                                "fold": fold,
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "best_f1": best_f1,
                                "config": dict(cfg.model),
                            },
                            ckpt_path,
                        )
                        mlflow.log_artifact(
                            str(ckpt_path), artifact_path=f"fold_{fold}/checkpoints"
                        )

                fold_results.append(best_f1)
                mlflow.log_metric("best_f1", best_f1)
                print(f"  Fold {fold + 1} best F1: {best_f1:.4f}")

        print(f"\n=== {cfg.training.n_splits}-Fold CV Results ===")
        print(f"Mean F1@50: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")

        mlflow.log_metric("cv_mean_f1", float(np.mean(fold_results)))
        mlflow.log_metric("cv_std_f1", float(np.std(fold_results)))

        with open(ckpt_dir / "cv_results.json", "w") as f:
            json.dump(
                {
                    "fold_f1s": fold_results,
                    "mean": float(np.mean(fold_results)),
                    "std": float(np.std(fold_results)),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
