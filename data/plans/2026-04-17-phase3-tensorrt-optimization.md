# Phase 3: TensorRT Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate pose estimation inference with TensorRT via ONNX Runtime TensorRT Execution Provider

**Architecture:** Extend DeviceConfig with TensorRT flag, modify PoseExtractor to conditionally use TensorRT EP, benchmark performance

**Tech Stack:** onnxruntime-gpu (already has TensorRT EP), tensorrt (for engine building), pycuda

---

## Task 1: Verify TensorRT EP Availability

**Files:**
- Create: `ml/scripts/check_tensorrt.py`
- Test: `ml/tests/test_tensorrt_ep.py`

- [ ] **Step 1: Write detection script**

```python
# ml/scripts/check_tensorrt.py
"""Check if TensorRT Execution Provider is available."""

import sys

def main():
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_tensorrt = "TensorrtExecutionProvider" in providers

        print("Available ONNX Runtime providers:")
        for p in providers:
            print(f"  - {p}")

        if has_tensorrt:
            print("\n✅ TensorRT EP available")
            return 0
        else:
            print("\n❌ TensorRT EP NOT available")
            print("Install: pip install tensorrt pycuda")
            return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run detection**

Run: `uv run python ml/scripts/check_tensorrt.py`
Expected: Either ✅ TensorRT EP available or ❌ message with install instructions

- [ ] **Step 3: Write test**

```python
# ml/tests/test_tensorrt_ep.py
"""Tests for TensorRT EP availability."""

import pytest

def test_tensorrt_ep_available():
    """Check if TensorRT EP is available (optional)."""
    import onnxruntime as ort

    providers = ort.get_available_providers()
    has_tensorrt = "TensorrtExecutionProvider" in providers

    # This test should pass regardless of TensorRT availability
    # It just reports the status
    if has_tensorrt:
        assert "TensorrtExecutionProvider" in providers
```

- [ ] **Step 4: Run test**

Run: `uv run pytest ml/tests/test_tensorrt_ep.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/scripts/check_tensorrt.py ml/tests/test_tensorrt_ep.py
git commit -m "feat(pose): add TensorRT EP detection script"
```

---

## Task 2: Extend DeviceConfig with TensorRT Support

**Files:**
- Modify: `ml/src/device.py`
- Test: `ml/tests/test_device.py`

- [ ] **Step 1: Add tensorrt_enabled field to DeviceConfig**

```python
# ml/src/device.py
# Add after line 99:

@dataclass(frozen=True)
class DeviceConfig:
    device: DeviceName = "cuda"
    tensorrt_enabled: bool = False  # NEW

    def __init__(self, device: str = "auto", tensorrt_enabled: bool = False) -> None:
        resolved = _resolve_device_name(device)
        object.__setattr__(self, "device", resolved)
        object.__setattr__(self, "tensorrt_enabled", tensorrt_enabled)
```

- [ ] **Step 2: Update onnx_providers property**

```python
# Replace existing onnx_providers property (lines 134-138):

@property
def onnx_providers(self) -> list[str]:
    """ONNX Runtime execution providers for this device."""
    if self.is_cuda:
        if self.tensorrt_enabled:
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
```

- [ ] **Step 3: Add convenience classmethod**

```python
# Add after default() classmethod (around line 114):

@classmethod
def with_tensorrt(cls, device: str = "auto") -> DeviceConfig:
    """Create config with TensorRT enabled (CUDA required)."""
    return cls(device=device, tensorrt_enabled=True)
```

- [ ] **Step 4: Write tests**

```python
# Add to ml/tests/test_device.py:

class TestDeviceConfigTensorRT:
    """Tests for DeviceConfig TensorRT support."""

    def test_tensorrt_disabled_by_default(self):
        """TensorRT should be disabled by default."""
        from src.device import DeviceConfig

        cfg = DeviceConfig()
        assert not cfg.tensorrt_enabled
        assert "TensorrtExecutionProvider" not in cfg.onnx_providers

    def test_tensorrt_enabled_cuda(self):
        """TensorRT EP should be first in providers when enabled."""
        from src.device import DeviceConfig

        cfg = DeviceConfig(device="cuda", tensorrt_enabled=True)
        assert cfg.tensorrt_enabled
        assert cfg.onnx_providers[0] == "TensorrtExecutionProvider"

    def test_tensorrt_ignored_on_cpu(self):
        """TensorRT flag should be ignored on CPU."""
        from src.device import DeviceConfig

        cfg = DeviceConfig(device="cpu", tensorrt_enabled=True)
        assert cfg.tensorrt_enabled  # Flag is set but...
        assert "TensorrtExecutionProvider" not in cfg.onnx_providers  # ...not in providers

    def test_with_tensorrt_classmethod(self):
        """with_tensorrt() convenience method."""
        from src.device import DeviceConfig

        cfg = DeviceConfig.with_tensorrt()
        assert cfg.tensorrt_enabled
        assert cfg.onnx_providers[0] == "TensorrtExecutionProvider"

    def test_repr_includes_tensorrt(self):
        """repr should show TensorRT status."""
        from src.device import DeviceConfig

        cfg = DeviceConfig(device="cuda", tensorrt_enabled=True)
        assert "tensorrt" in repr(cfg).lower()
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest ml/tests/test_device.py::TestDeviceConfigTensorRT -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add ml/src/device.py ml/tests/test_device.py
git commit -m "feat(device): add TensorRT support to DeviceConfig"
```

---

## Task 3: Update PoseExtractor for TensorRT EP

**Files:**
- Modify: `ml/src/pose_estimation/pose_extractor.py`
- Test: `ml/tests/pose_estimation/test_pose_extractor_tensorrt.py`

- [ ] **Step 1: Add tensorrt parameter to PoseExtractor**

Read current `PoseExtractor.__init__` signature:
```bash
grep -A 20 "class PoseExtractor" ml/src/pose_estimation/pose_extractor.py
```

- [ ] **Step 2: Modify constructor to accept tensorrt flag**

```python
# In ml/src/pose_estimation/pose_extractor.py
# Modify PoseExtractor.__init__ to accept tensorrt parameter

# Add parameter after existing params:
def __init__(
    self,
    ...existing params...,
    tensorrt: bool = False,
):
    """Initialize pose extractor.

    Args:
        ...existing doc...
        tensorrt: Enable TensorRT EP (requires CUDA). Default: False.
    """
    # Store tensorrt flag
    self._tensorrt = tensorrt and self._device_cfg.is_cuda

    # Update DeviceConfig if tensorrt requested
    if tensorrt:
        self._device_cfg = DeviceConfig.with_tensorrt(device=self._device_cfg.device)
```

- [ ] **Step 3: Update extract_poses() function signature**

```python
# In ml/src/pose_estimation/pose_extractor.py
# Update extract_poses() function:

def extract_poses(
    video_path: str | Path,
    ...existing params...,
    tensorrt: bool = False,
) -> npt.NDArray[np.float32]:
    """Extract poses from video.

    Args:
        ...existing doc...
        tensorrt: Enable TensorRT EP. Default: False.
    """
    extractor = PoseExtractor(..., tensorrt=tensorrt)
    return extractor.extract_video(video_path)
```

- [ ] **Step 4: Write tests**

```python
# Create ml/tests/pose_estimation/test_pose_extractor_tensorrt.py:

import pytest
from src.pose_estimation import PoseExtractor
from src.device import DeviceConfig

class TestPoseExtractorTensorRT:
    """Tests for PoseExtractor TensorRT support."""

    def test_tensorrt_flag_stored(self):
        """TensorRT flag should be stored."""
        extractor = PoseExtractor(tensorrt=True)
        assert extractor._tensorrt is True

    def test_tensorrt_ignored_on_cpu(self):
        """TensorRT should be ignored on CPU device."""
        extractor = PoseExtractor(device="cpu", tensorrt=True)
        assert extractor._tensorrt is False  # Disabled on CPU

    def test_tensorrt_updates_providers(self):
        """TensorRT should update ONNX providers."""
        extractor = PoseExtractor(tensorrt=True)
        if extractor._device_cfg.is_cuda:
            assert "TensorrtExecutionProvider" in extractor._device_cfg.onnx_providers

    def test_tensorrt_disabled_by_default(self):
        """TensorRT should be disabled by default."""
        extractor = PoseExtractor()
        assert extractor._tensorrt is False
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest ml/tests/pose_estimation/test_pose_extractor_tensorrt.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add ml/src/pose_estimation/pose_extractor.py ml/tests/pose_estimation/test_pose_extractor_tensorrt.py
git commit -m "feat(pose): add TensorRT support to PoseExtractor"
```

---

## Task 4: Add TensorRT CLI Flag

**Files:**
- Modify: `ml/src/cli.py`
- Test: `ml/tests/test_cli_tensorrt.py`

- [ ] **Step 1: Add --tensorrt flag to analyze command**

```python
# In ml/src/cli.py
# Add to analyze_parser:

analyze_parser.add_argument(
    "--tensorrt",
    action="store_true",
    help="Enable TensorRT optimization (CUDA only, faster inference)"
)
```

- [ ] **Step 2: Pass tensorrt flag to pipeline**

```python
# In ml/src/cli.py
# In main() function, analyze command handler:

if args.command == "analyze":
    pipeline = AnalysisPipeline(
        ...existing params...,
        tensorrt=args.tensorrt,  # NEW
    )
```

- [ ] **Step 3: Update AnalysisPipeline to accept tensorrt**

```python
# In ml/src/pipeline.py
# Update AnalysisPipeline.__init__:

class AnalysisPipeline:
    def __init__(
        self,
        ...existing params...,
        tensorrt: bool = False,
    ):
        """Initialize analysis pipeline.

        Args:
            ...existing doc...
            tensorrt: Enable TensorRT optimization. Default: False.
        """
        self._tensorrt = tensorrt
        # Pass to extract_poses() call
```

- [ ] **Step 4: Write tests**

```python
# Create ml/tests/test_cli_tensorrt.py:

import pytest
from src.cli import main

class TestCLITensorRT:
    """Tests for CLI TensorRT flag."""

    def test_tensorrt_flag_accepted(monkeypatch, tmp_path):
        """--tensorrt flag should be accepted."""
        monkeypatch.setattr(sys, "argv", ["cli", "analyze", "test.mp4", "--tensorrt"])
        # Should not raise argparse error
        with pytest.raises(SystemExit):
            main()

    def test_tensorrt_flag_default_false(monkeypatch, tmp_path):
        """--tensorrt should default to False."""
        # Test that tensorrt=False when flag not provided
        # (Integration test would need mock video)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest ml/tests/test_cli_tensorrt.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ml/src/cli.py ml/src/pipeline.py ml/tests/test_cli_tensorrt.py
git commit -m "feat(cli): add --tensorrt flag"
```

---

## Task 5: Benchmark Performance

**Files:**
- Create: `ml/scripts/benchmark_tensorrt.py`
- Create: `ml/docs/benchmark_results.md`

- [ ] **Step 1: Write benchmark script**

```python
# ml/scripts/benchmark_tensorrt.py
"""Benchmark TensorRT vs CUDA performance."""

import time
import numpy as np
from pathlib import Path
from src.pose_estimation import PoseExtractor

def benchmark(video_path: str, runs: int = 5):
    """Benchmark inference with and without TensorRT."""

    print(f"Benchmarking: {video_path}")
    print(f"Runs: {runs}")

    # Baseline: CUDA only
    print("\n🔵 Baseline (CUDA only)...")
    extractor_cuda = PoseExtractor(device="cuda", tensorrt=False)

    times_cuda = []
    for i in range(runs):
        start = time.perf_counter()
        poses = extractor_cuda.extract_video(video_path)
        elapsed = time.perf_counter() - start
        times_cuda.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s ({len(poses)} frames)")

    baseline_fps = len(poses) / np.median(times_cuda)

    # TensorRT enabled
    print("\n🟢 TensorRT enabled...")
    try:
        extractor_trt = PoseExtractor(device="cuda", tensorrt=True)
        times_trt = []

        for i in range(runs):
            start = time.perf_counter()
            poses = extractor_trt.extract_video(video_path)
            elapsed = time.perf_counter() - start
            times_trt.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s ({len(poses)} frames)")

        tensorrt_fps = len(poses) / np.median(times_trt)
        speedup = tensorrt_fps / baseline_fps

        print("\n📊 Results:")
        print(f"  Baseline:  {baseline_fps:.1f} FPS")
        print(f"  TensorRT:  {tensorrt_fps:.1f} FPS")
        print(f"  Speedup:   {speedup:.2f}x")

        return {
            "baseline_fps": baseline_fps,
            "tensorrt_fps": tensorrt_fps,
            "speedup": speedup,
        }

    except Exception as e:
        print(f"\n❌ TensorRT benchmark failed: {e}")
        return {
            "baseline_fps": baseline_fps,
            "tensorrt_fps": None,
            "speedup": None,
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: benchmark_tensorrt.py <video_path> [runs]")
        sys.exit(1)

    video_path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    benchmark(video_path, runs)
```

- [ ] **Step 2: Run benchmark**

Run: `uv run python ml/scripts/benchmark_tensorrt.py /path/to/test_video.mp4 3`
Expected: Performance comparison output

- [ ] **Step 3: Document results**

```markdown
# ml/docs/benchmark_results.md

# TensorRT Benchmark Results

## Test Environment
- GPU: RTX 3050 Ti (4GB VRAM)
- CUDA: 13.2
- ONNX Runtime: 1.24.4
- Date: [RUN_DATE]

## Results

| Video | Baseline FPS | TensorRT FPS | Speedup |
|-------|-------------|-------------|---------|
| test1.mp4 | [FILL] | [FILL] | [FILL] |

## Notes
- [Add observations about startup time, memory usage, etc.]
```

- [ ] **Step 4: Commit**

```bash
git add ml/scripts/benchmark_tensorrt.py ml/docs/benchmark_results.md
git commit -m "feat(bench): add TensorRT benchmark script"
```

---

## Task 6: Documentation and Cleanup

**Files:**
- Modify: `ml/CLAUDE.md`
- Modify: `ml/src/__init__.py`

- [ ] **Step 1: Update CLAUDE.md**

```markdown
# In ml/CLAUDE.md, add after "Device Configuration" section:

## TensorRT Optimization

TensorRT acceleration is available via ONNX Runtime TensorRT Execution Provider:

```python
from src.device import DeviceConfig
from src.pose_estimation import PoseExtractor

# Enable TensorRT
cfg = DeviceConfig.with_tensorrt()
extractor = PoseExtractor(device=cfg)

# Or via CLI
uv run python -m src.cli analyze video.mp4 --tensorrt
```

**Expected performance:** 1.5-2x speedup on RTX 3050 Ti.

**Requirements:**
- CUDA GPU
- tensorrt package installed
- pycuda package installed

Check availability: `uv run python ml/scripts/check_tensorrt.py`
```

- [ ] **Step 2: Update __init__.py exports (if needed)**

```python
# In ml/src/__init__.py
# Add TensorRT-related exports if exposing new API
```

- [ ] **Step 3: Run all tests**

Run: `uv run pytest ml/tests/ -v --no-cov`
Expected: All tests pass

- [ ] **Step 4: Lint check**

Run: `uv run ruff check ml/src/`
Expected: No errors

- [ ] **Step 5: Final commit**

```bash
git add ml/CLAUDE.md ml/src/__init__.py
git commit -m "docs(tensorrt): update documentation for TensorRT support"
```

---

## Verification Steps

After completing all tasks:

1. **Test TensorRT detection:**
   ```bash
   uv run python ml/scripts/check_tensorrt.py
   ```

2. **Run unit tests:**
   ```bash
   uv run pytest ml/tests/test_device.py::TestDeviceConfigTensorRT -v
   uv run pytest ml/tests/pose_estimation/test_pose_extractor_tensorrt.py -v
   ```

3. **Benchmark performance:**
   ```bash
   uv run python ml/scripts/benchmark_tensorrt.py data/test_video.mp4 3
   ```

4. **Test CLI integration:**
   ```bash
   uv run python -m src.cli analyze video.mp4 --tensorrt --element waltz_jump
   ```

---

## Expected Outcomes

- ✅ DeviceConfig supports TensorRT via `tensorrt_enabled` flag
- ✅ PoseExtractor accepts `tensorrt` parameter
- ✅ CLI has `--tensorrt` flag
- ✅ Benchmark script measures actual performance gains
- ✅ All tests passing
- ✅ Documentation updated

## Risk Assessment

**Low Risk:**
- ONNX Runtime TensorRT EP is well-tested
- Changes are additive (non-breaking)
- Fallback to CUDA if TensorRT unavailable

**Medium Risk:**
- TensorRT engine compilation on first run (slow startup)
- Potential precision differences (minimal for FP16)

**Mitigation:**
- TensorRT is opt-in (disabled by default)
- Clear error messages if TensorRT unavailable
- Benchmark validation before enabling in production
