#!/usr/bin/env python3
"""Set up isolated mmpose environment for RTMPose-s SimCC distillation.

Usage:
    python scripts/setup_mmpose_env.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "rtmpose-simcc-kd"
VENV_DIR = EXPERIMENT_DIR / ".venv"
CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"
REQUIREMENTS_FILE = EXPERIMENT_DIR / "requirements.txt"

# Official mmpose URLs
RTMPOSE_CONFIG_URL = (
    "https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
    "configs/body_2d_keypoint/rtmpose/coco/"
    "rtmpose-s_8xb256-420e_coco-256x192.py"
)
RTMPOSE_WEIGHTS_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
    "rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-d8df01c0_20230127.pth"
)


def run(cmd: list[str] | str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a shell command and stream output."""
    if isinstance(cmd, list):
        cmd_str = " ".join(cmd)
    else:
        cmd_str = cmd
        cmd = ["bash", "-c", cmd]

    print(f"\n[RUN] {cmd_str}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def setup_venv() -> Path:
    """Create isolated venv with Python 3.11."""
    print("=== Step 1: Create isolated venv ===")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    if VENV_DIR.exists():
        print(f"Venv already exists at {VENV_DIR}")
    else:
        run(
            ["uv", "venv", str(VENV_DIR), "--python", "3.11"],
            cwd=PROJECT_ROOT,
        )

    python_exe = VENV_DIR / "bin" / "python"
    if not python_exe.exists():
        print(f"[ERROR] Python not found at {python_exe}")
        sys.exit(1)

    print(f"Python: {python_exe}")
    return python_exe


def install_core_deps(python_exe: Path) -> None:
    """Install torch, openmim, mmengine, mmcv, mmpose."""
    print("\n=== Step 2: Install core dependencies ===")

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(VENV_DIR)
    env["PATH"] = f"{VENV_DIR / 'bin'}:{env.get('PATH', '')}"
    env["UV_PYTHON"] = str(python_exe)

    # 2a. PyTorch CUDA 12.1
    run(
        [
            "uv",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        cwd=PROJECT_ROOT,
        env=env,
    )

    # 2b. openmim
    run(
        ["uv", "pip", "install", "-U", "openmim"],
        cwd=PROJECT_ROOT,
        env=env,
    )

    # 2c. mmengine, mmcv, mmpose via mim
    # mim installs packages into the active virtualenv using pip underneath
    mim = str(VENV_DIR / "bin" / "mim")
    run(
        [mim, "install", "mmengine"],
        cwd=PROJECT_ROOT,
        env=env,
    )
    run(
        [mim, "install", "mmcv>=2.0.1"],
        cwd=PROJECT_ROOT,
        env=env,
    )
    run(
        [mim, "install", "mmpose>=1.3.0"],
        cwd=PROJECT_ROOT,
        env=env,
    )

    # Verify imports
    print("\n--- Verifying imports ---")
    verify_script = "; ".join(
        [
            "import mmpose",
            "import mmcv",
            "import mmengine",
            "print('mmpose:', mmpose.__version__)",
            "print('mmcv:', mmcv.__version__)",
            "print('mmengine:', mmengine.__version__)",
        ]
    )
    run([str(python_exe), "-c", verify_script], cwd=PROJECT_ROOT, env=env)


def pin_requirements() -> None:
    """Pin installed versions to requirements.txt."""
    print("\n=== Step 3: Pin requirements.txt ===")

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(VENV_DIR)
    env["PATH"] = f"{VENV_DIR / 'bin'}:{env.get('PATH', '')}"

    # uv pip freeze outputs pinned requirements
    result = subprocess.run(
        ["uv", "pip", "freeze"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Fallback: use pip freeze via venv
        result = subprocess.run(
            [str(VENV_DIR / "bin" / "python"), "-m", "pip", "freeze"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

    pinned = result.stdout.strip().splitlines()

    # Append our extra constraints if not already present
    extras = [
        "h5py>=3.10.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "onnxruntime-gpu>=1.16.0",
    ]

    existing_packages = {line.split("==")[0].lower() for line in pinned if "==" in line}
    for extra in extras:
        pkg = extra.split(">=")[0].lower()
        if pkg not in existing_packages:
            pinned.append(extra)

    REQUIREMENTS_FILE.write_text("\n".join(pinned) + "\n")
    print(f"Wrote {len(pinned)} lines to {REQUIREMENTS_FILE}")


def download_assets() -> None:
    """Download RTMPose-s config and pretrained weights."""
    print("\n=== Step 4: Download RTMPose-s config and weights ===")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = CHECKPOINTS_DIR / "rtmpose-s_8xb256-420e_coco-256x192.py"
    weights_path = (
        CHECKPOINTS_DIR / "rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-d8df01c0_20230127.pth"
    )

    if config_path.exists():
        print(f"Config already exists: {config_path} ({config_path.stat().st_size} bytes)")
    else:
        run(["wget", "-q", "-O", str(config_path), RTMPOSE_CONFIG_URL], cwd=PROJECT_ROOT)
        print(f"Downloaded config: {config_path} ({config_path.stat().st_size} bytes)")

    if weights_path.exists():
        print(f"Weights already exist: {weights_path} ({weights_path.stat().st_size} bytes)")
    else:
        run(["wget", "-q", "-O", str(weights_path), RTMPOSE_WEIGHTS_URL], cwd=PROJECT_ROOT)
        print(f"Downloaded weights: {weights_path} ({weights_path.stat().st_size} bytes)")

    # Sanity-check sizes
    config_size = config_path.stat().st_size
    weights_size = weights_path.stat().st_size
    assert 1000 < config_size < 100_000, f"Config size unexpected: {config_size}"
    assert 10_000_000 < weights_size < 50_000_000, f"Weights size unexpected: {weights_size}"
    print("Size checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up mmpose environment")
    parser.add_argument("--skip-venv", action="store_true", help="Skip venv creation if it exists")
    parser.add_argument("--skip-install", action="store_true", help="Skip package installation")
    parser.add_argument("--skip-download", action="store_true", help="Skip asset download")
    args = parser.parse_args()

    python_exe = setup_venv()

    if not args.skip_install:
        install_core_deps(python_exe)
        pin_requirements()
    else:
        print("Skipping package installation (--skip-install)")

    if not args.skip_download:
        download_assets()
    else:
        print("Skipping asset download (--skip-download)")

    print("\n=== Setup complete ===")
    print(f"Venv: {VENV_DIR}")
    print(f"Requirements: {REQUIREMENTS_FILE}")
    print(f"Checkpoints: {CHECKPOINTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
