"""Evaluate TAS model on held-out MCFS fold.

Produces: per-class accuracy, confusion matrix, OverlapF1@50, qualitative examples.

Usage:
    uv run python experiments/evaluate_tas.py --checkpoint experiments/checkpoints/tas_bigr/fold_0_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ml.src.tas.dataset import MCFSCoarseDataset, pad_collate
from ml.src.tas.metrics import OverlapF1
from ml.src.tas.model import BiGRUTAS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MCFSCoarseDataset(Path("data/datasets/mcfs/features"), Path("data/datasets/mcfs/groundTruth"))

    # Use last 20% as test set (not in CV folds)
    n = len(ds)
    test_idx = list(range(int(n * 0.8), n))
    test_ds = Subset(ds, test_idx)
    loader = DataLoader(test_ds, batch_size=8, collate_fn=pad_collate)

    model = BiGRUTAS().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metric = OverlapF1(iou_threshold=0.5)
    all_f1s = []
    all_frame_acc = []

    with torch.no_grad():
        for poses, labels, lengths in loader:
            poses = poses.to(device)
            logits = model(poses, lengths)
            preds = logits.argmax(dim=-1).cpu().numpy()
            for i, le in enumerate(lengths):
                p = preds[i, :le]
                t = labels[i, :le].numpy()
                f1_result = metric.compute(p, t)
                all_f1s.append(f1_result["f1"])
                all_frame_acc.append((p == t).mean())

    print(f"Mean Frame Acc: {np.mean(all_frame_acc):.4f}")
    print(f"Mean OverlapF1@50: {np.mean(all_f1s):.4f}")
    print(f"Median OverlapF1@50: {np.median(all_f1s):.4f}")


if __name__ == "__main__":
    main()
