"""Train YOLO26-Pose on AthletePose3D + COCO mix.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 train_yolo26_pose.py --config configs/hp_lr001_640.yaml
    CUDA_VISIBLE_DEVICES=0 python3 train_yolo26_pose.py --name hp_test --lr 0.001 --freeze 5 --epochs 50 --imgsz 640
"""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26-Pose")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--name", type=str, default="run", help="Run name")
    parser.add_argument("--data", type=str, default="/root/data/datasets/yolo26_ap3d/data.yaml", help="Dataset YAML")
    parser.add_argument("--model", type=str, default="yolo26s-pose.pt", help="Model to fine-tune")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR factor")
    parser.add_argument("--freeze", type=int, default=5, help="Freeze backbone layers")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1=auto)")
    parser.add_argument("--device", type=int, default=0, help="GPU device")
    parser.add_argument("--val", action="store_true", default=False, help="Enable validation")
    parser.add_argument("--no-val", action="store_true", dest="no_val", help="Disable validation")
    parser.add_argument("--save-period", type=int, default=0, help="Save checkpoint every N epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation")
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup augmentation")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip prob")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste augmentation")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")

    args = parser.parse_args()

    # Override from YAML config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.get("hyperparameters", {}).items():
            if hasattr(args, k) and v is not None:
                setattr(args, k, v)
        if "name" in cfg:
            args.name = cfg["name"]

    if args.no_val:
        args.val = False

    print(f"=== YOLO26-Pose Training: {args.name} ===")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"LR: {args.lr}, freeze: {args.freeze}, epochs: {args.epochs}")
    print(f"imgsz: {args.imgsz}, batch: {args.batch}, device: {args.device}")
    print(f"val: {args.val}, save_period: {args.save_period}")
    print(f"augment: mosaic={args.mosaic}, mixup={args.mixup}, fliplr={args.fliplr}")

    from ultralytics import YOLO

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        lr0=args.lr,
        lrf=args.lrf,
        freeze=args.freeze,
        mosaic=args.mosaic,
        mixup=args.mixup,
        fliplr=args.fliplr,
        copy_paste=args.copy_paste,
        workers=args.workers,
        device=args.device,
        project="runs/pose",
        name=args.name,
        exist_ok=True,
        val=args.val,
        patience=args.patience if args.val else 0,
        save_period=args.save_period,
        verbose=True,
    )

    if args.batch > 0:
        train_kwargs["batch"] = args.batch

    t0 = time.time()
    model.train(**train_kwargs)
    elapsed = time.time() - t0

    print(f"\n=== Training complete: {args.name} ===")
    print(f"Time: {elapsed/60:.1f} min")

    # Report best result
    results_path = Path(f"runs/pose/{args.name}/results.csv")
    if results_path.exists():
        import csv
        rows = list(csv.DictReader(results_path.open()))
        if rows:
            best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))
            print(f"Best mAP50: {best.get('metrics/mAP50(B)', 'N/A')} @ epoch {best.get('epoch', '?')}")
            print(f"Best mAP50-95: {best.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == "__main__":
    main()
