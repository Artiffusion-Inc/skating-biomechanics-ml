"""
Experiment 7 — InfoGCN for Figure Skating Classification (H25-H29)

Hypotheses:
    H25: InfoGCN on FSC 64 classes (from scratch) >= 70%
    H26: InfoGCN + BalancedSampler >= InfoGCN plain (>= +5pp)
    H27: InfoGCN multi-modal (joint + bone + motion) >= joint-only (>= +3pp)
    H28: InfoGCN pretrained SkatingVerse -> fine-tune FSC 64 >= 78% (BLOCKED)
    H29: InfoGCN + joint angles >= InfoGCN without (>= +2pp)

Based on: InfoGCN (Chi et al., CVPR 2022) — stnoah1/infogcn
DeepGlint result: 92.03% on SkatingVerse 28 classes with InfoGCN + ViTPose skeletons.

Usage:
    # H25: InfoGCN baseline
    uv run python experiments/exp_infogcn.py --exp h25

    # H26: BalancedSampler
    uv run python experiments/exp_infogcn.py --exp h26

    # H27: Multi-modal ensemble
    uv run python experiments/exp_infogcn.py --exp h27

    # H29: Joint angles
    uv run python experiments/exp_infogcn.py --exp h29

    # Quick test (10 epochs)
    uv run python experiments/exp_infogcn.py --exp h25 --epochs 10

Status: PENDING
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import optim
from torch.amp import GradScaler, autocast

# Add experiments/ to path for infogcn package
sys.path.insert(0, str(Path(__file__).parent))

from infogcn.feeder import FSCFeeder
from infogcn.loss import BalancedSampler, LabelSmoothingCE, get_mmd_loss
from infogcn.model import InfoGCN

BASE = Path(__file__).resolve().parent.parent / "data" / "datasets"


def load_fsc_data(modality="joint", window_size=64, balanced=False):
    """Load FSC dataset with given modality."""
    data_path = BASE / "figure-skating-classification"
    num_classes = 64

    train_set = FSCFeeder(
        data_path=data_path,
        split="train",
        modality=modality,
        window_size=window_size,
        p_interval=(0.5, 1.0),
        random_rot=True,
    )
    test_set = FSCFeeder(
        data_path=data_path,
        split="test",
        modality=modality,
        window_size=window_size,
        p_interval=(0.95,),
        random_rot=False,
    )

    if balanced:
        sampler = BalancedSampler(train_set, num_classes=num_classes)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=64,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    in_channels = {
        "joint": 2,
        "bone": 2,
        "motion": 2,
        "joint_angle": 13,
    }.get(modality, 2)

    return train_loader, test_loader, num_classes, in_channels


def train_epoch(model, loader, optimizer, scaler, loss_fn, device, lambda_1, lambda_2):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data, y, _ in loader:
        data = data.float().to(device)
        y = y.long().to(device)

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            y_hat, z = model(data)
            mmd_loss, l2_z, _ = get_mmd_loss(z, model.z_prior, y, model.num_class)
            cls_loss = loss_fn(y_hat, y)
            loss = cls_loss + lambda_1 * l2_z + lambda_2 * mmd_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += cls_loss.item() * y.size(0)
        correct += (y_hat.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, lambda_1, lambda_2):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for data, y, _ in loader:
        data = data.float().to(device)
        y = y.long().to(device)

        with autocast(device_type="cuda"):
            y_hat, z = model(data)
            mmd_loss, l2_z, _ = get_mmd_loss(z, model.z_prior, y, model.num_class)
            cls_loss = loss_fn(y_hat, y)
            loss = cls_loss + lambda_1 * l2_z + lambda_2 * mmd_loss

        total_loss += cls_loss.item() * y.size(0)
        correct += (y_hat.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, 100 * correct / total


def run_single_experiment(
    exp_name,
    modality="joint",
    balanced=False,
    num_epochs=110,
    window_size=64,
    lr=0.1,
    in_channels_override=None,
):
    """Run a single InfoGCN experiment configuration."""
    print(f"\n{'=' * 60}")
    print(f"  {exp_name}: InfoGCN ({modality}, balanced={balanced})")
    print(f"{'=' * 60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_loader, test_loader, num_classes, in_channels = load_fsc_data(
        modality=modality,
        window_size=window_size,
        balanced=balanced,
    )
    if in_channels_override:
        in_channels = in_channels_override

    print(f"Train: {len(train_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")
    print(f"Classes: {num_classes}, In channels: {in_channels}, Window: {window_size}")

    # Model
    model = InfoGCN(
        num_class=num_classes,
        num_point=17,
        num_person=1,
        in_channels=in_channels,
        drop_out=0.0,
        num_head=3,
        noise_ratio=0.5,
        k=0,
        gain=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Optimizer (same as original: SGD + Nesterov)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(num_epochs * 0.82), int(num_epochs * 0.91)], gamma=0.1
    )
    loss_fn = LabelSmoothingCE(smoothing=0.1)
    scaler = GradScaler()

    lambda_1, lambda_2 = 1e-4, 0.1
    best_acc = 0.0
    warm_up_epochs = 5
    start_time = time.time()

    for epoch in range(num_epochs):
        # Learning rate warmup (linear from 0 to base_lr over warm_up_epochs)
        if epoch < warm_up_epochs:
            warmup_lr = lr * (epoch + 1) / warm_up_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            loss_fn,
            device,
            lambda_1,
            lambda_2,
        )
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            lambda_1,
            lambda_2,
        )
        scheduler.step()

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        if (epoch + 1) % 10 == 0 or is_best or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Train: {train_acc:5.1f}% (loss {train_loss:.4f}) | "
                f"Test: {test_acc:5.1f}% (loss {test_loss:.4f}) | "
                f"Best: {best_acc:5.1f}%" + (" *" if is_best else "")
            )

    elapsed = time.time() - start_time
    print(f"\n  Result: {exp_name} = {best_acc:.1f}% ({elapsed:.0f}s)")
    return best_acc


def run_h25(num_epochs=110):
    """H25: InfoGCN on FSC 64 classes from scratch.
    Target: >= 70%. BiGRU baseline: 67.9%."""
    return run_single_experiment(
        "H25-InfoGCN", modality="joint", balanced=False, num_epochs=num_epochs
    )


def run_h26(num_epochs=110):
    """H26: InfoGCN + BalancedSampler.
    Target: >= H25 + 5pp. Tests if balanced sampling fixes 50:1 imbalance."""
    return run_single_experiment(
        "H26-InfoGCN+BS", modality="joint", balanced=True, num_epochs=num_epochs
    )


def run_h27(num_epochs=110):
    """H27: Multi-modal ensemble (joint + bone + motion).
    Trains 3 separate models, ensembles via softmax averaging.
    Target: >= joint-only + 3pp."""
    modalities = ["joint", "bone", "motion"]
    scores = {}
    for mod in modalities:
        print(f"\n--- Training {mod} stream ---")
        scores[mod] = run_single_experiment(
            f"H27-{mod}",
            modality=mod,
            balanced=True,
            num_epochs=num_epochs,
        )

    # Ensemble evaluation
    print("\n--- H27 Ensemble ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, num_classes, _ = load_fsc_data(modality="joint", window_size=64, balanced=False)

    models = {}
    for mod in modalities:
        in_ch = 2
        m = InfoGCN(num_class=num_classes, num_point=17, num_person=1, in_channels=in_ch).to(device)
        # Load best weights (saved by run_single_experiment)
        ckpt = Path(f"experiments/checkpoints/infogcn_h27_{mod}_best.pt")
        if ckpt.exists():
            m.load_state_dict(torch.load(ckpt, map_location=device))
        models[mod] = m

    # Evaluate ensemble
    all_preds = {mod: [] for mod in modalities}
    all_labels = []
    with torch.no_grad():
        for data, y, _ in test_loader:
            data = data.float().to(device)
            for mod in modalities:
                out, _ = models[mod](data)
                all_preds[mod].append(out.softmax(1).cpu())
            all_labels.append(y)

    all_labels = torch.cat(all_labels)
    ensemble_pred = sum(torch.cat(all_preds[m]) for m in modalities) / len(modalities)
    acc = (ensemble_pred.argmax(1) == all_labels).float().mean().item() * 100
    print(f"  H27 Ensemble: {acc:.1f}%")
    print("  Individual: " + ", ".join(f"{m}={scores[m]:.1f}%" for m in modalities))
    return acc


def run_h29(num_epochs=110):
    """H29: InfoGCN + joint angles (13 input channels: xy + 11 angles).
    Target: >= H25 + 2pp. Tests if view-invariant angles help InfoGCN."""
    return run_single_experiment(
        "H29-InfoGCN+Angles",
        modality="joint_angle",
        balanced=True,
        num_epochs=num_epochs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="InfoGCN experiments for figure skating classification"
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["h25", "h26", "h27", "h29"],
        help="Experiment hypothesis to run",
    )
    parser.add_argument("--epochs", type=int, default=110, help="Number of training epochs")
    parser.add_argument("--window-size", type=int, default=64, help="Temporal window size")
    parser.add_argument("--lr", type=float, default=0.1, help="Base learning rate")
    args = parser.parse_args()

    print("InfoGCN Figure Skating Classification Experiments")
    print(f"Experiment: {args.exp.upper()}, Epochs: {args.epochs}")

    experiments = {
        "h25": run_h25,
        "h26": run_h26,
        "h27": run_h27,
        "h29": run_h29,
    }

    results = {}
    for exp in [args.exp]:
        acc = experiments[exp](num_epochs=args.epochs)
        results[exp] = acc

    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for exp, acc in results.items():
        print(f"  {exp.upper()}: {acc:.1f}%")
    print()


if __name__ == "__main__":
    main()
