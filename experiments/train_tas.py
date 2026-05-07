"""Experiment: TAS BiGRU coarse segmentation.

Hypothesis: BiGRU on normalized H3.6M poses achieves >0.70 OverlapF1@50 on MCFS 4-class segmentation
Status: PENDING

Usage:
    uv run python experiments/train_tas.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from ml.src.tas.dataset import MCFSCoarseDataset, pad_collate
from ml.src.tas.metrics import OverlapF1
from ml.src.tas.model import BiGRUTAS

BASE = Path("data/datasets/mcfs")
CHECKPOINT_DIR = Path("experiments/checkpoints/tas_bigr")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for poses, labels, lengths in loader:
        poses, labels = poses.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(poses, lengths)
        # Masked cross-entropy (ignore -1 padding)
        loss = criterion(logits.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_fold(model, loader, device) -> dict:
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

    # Compute per-sample F1 and average
    f1s = []
    for p, t in zip(all_preds, all_true):
        result = metric.compute(p, t)
        f1s.append(result["f1"])
    return {"f1": float(np.mean(f1s)), "precision": 0.0, "recall": 0.0}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = MCFSCoarseDataset(BASE / "features", BASE / "groundTruth")
    print(f"Dataset size: {len(ds)}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(ds)))):
        print(f"\n--- Fold {fold + 1}/5 ---")
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=pad_collate)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=pad_collate)

        model = BiGRUTAS(hidden_dim=128, num_layers=2, dropout=0.3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        best_f1 = 0.0
        for epoch in range(50):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_result = eval_fold(model, val_loader, device)
            print(f"  Epoch {epoch + 1}: loss={train_loss:.4f}, val_f1={val_result['f1']:.4f}")
            if val_result["f1"] > best_f1:
                best_f1 = val_result["f1"]
                torch.save({
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "config": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
                }, CHECKPOINT_DIR / f"fold_{fold}_best.pt")

        fold_results.append(best_f1)
        print(f"  Fold {fold + 1} best F1: {best_f1:.4f}")

    print(f"\n=== 5-Fold CV Results ===")
    print(f"Mean F1@50: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")

    # Save summary
    with open(CHECKPOINT_DIR / "cv_results.json", "w") as f:
        json.dump({"fold_f1s": fold_results, "mean": float(np.mean(fold_results)), "std": float(np.std(fold_results))}, f, indent=2)


if __name__ == "__main__":
    main()
