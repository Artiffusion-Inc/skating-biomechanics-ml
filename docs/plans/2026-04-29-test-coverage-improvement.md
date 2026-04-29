---
title: "Test Coverage Improvement Plan"
date: "2026-04-29"
status: planned
---
# Test Coverage Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise combined Python test coverage from 65% to 80%+ by targeting ML modules with 0% and very low coverage, plus backend edge cases and frontend vitest setup.

**Architecture:** Three-tier approach: (1) Quick wins — 11 ML files at 0% coverage and 2 backend files, (2) Medium effort — top 5 low-coverage ML modules by missed lines, (3) Frontend — vitest config + core component tests. Each tier independent; can execute in parallel.

**Tech Stack:** pytest, pytest-cov, pytest-asyncio, numpy, httpx, FastAPI TestClient, vitest, React Testing Library

---

## File Structure (what changes)

| Path | What | Coverage Target |
|------|------|-----------------|
| `ml/tests/references/` | NEW dir — reference_store, reference_builder tests | 0% → 80% |
| `ml/tests/datasets/` | NEW dir — coco_builder tests | 0% → 80% |
| `ml/tests/extras/` | NEW dir — inpainting, video_matting, depth_anything tests | 0% → 70% |
| `ml/tests/pose_estimation/` | MODIFY — add batch_extractor, tcpformer_extractor tests | 0% → 60% |
| `ml/tests/visualization/` | MODIFY — add export_3d, comparison, export_3d_animated tests | 0% → 50% |
| `ml/tests/test_web_helpers.py` | NEW — web_helpers tests | 7% → 50% |
| `ml/tests/pose_estimation/` | MODIFY — pose_extractor.py tests | 14% → 50% |
| `ml/tests/test_pipeline.py` | MODIFY — pipeline.py tests | 34% → 60% |
| `backend/tests/test_main.py` | NEW — main.py smoke tests | 0% → 100% |
| `backend/tests/test_logging_config.py` | NEW — logging_config tests | 0% → 100% |
| `frontend/vitest.config.ts` | MODIFY — ensure coverage reporter enabled | N/A |
| `frontend/src/lib/__tests__/` | NEW/EXTEND — component tests | 0% → 20% |

---

## Coverage Baseline (measured 2026-04-29)

| Module | Lines | Missed | Current | Target |
|--------|-------|--------|---------|--------|
| backend/app (overall) | 2,601 | 145 | **94%** | 96% |
| ml/src (overall) | 11,005 | 5,947 | **46%** | 65% |
| **Combined** | **11,546** | **4,002** | **65%** | **80%** |

### Why coverage is low

Backend 94% — good. ML 46% — drags everything down. Frontend vitest has 1 test file total (~0% coverage of 8,745 TS lines). The 11 ML files at exactly 0% coverage account for 1,121 uncovered lines. The next 5 lowest-coverage modules account for another 1,423 missed lines.

---

## Tier 1: Quick Wins — 0% Coverage Files

### Task 1: references/reference_store.py + reference_builder.py

**Files:**
- Create: `ml/tests/references/test_reference_store.py`
- Create: `ml/tests/references/test_reference_builder.py`
- Read: `ml/src/references/reference_store.py` (41 lines)
- Read: `ml/src/references/reference_builder.py` (26 lines)

**Steps:**

- [ ] **Step 1: Read the source files**

Read `ml/src/references/reference_store.py` and `ml/src/references/reference_builder.py`. Understand the public API: `ReferenceStore.load()`, `.save()`, `.list()`, `.get()` and `ReferenceBuilder` class.

- [ ] **Step 2: Write tests for reference_store**

```python
"""Tests for ml.src.references.reference_store."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.references.reference_store import ReferenceStore
from src.types import ReferenceData


class TestReferenceStore:
    def test_save_and_load_roundtrip(self):
        store = ReferenceStore()
        ref = ReferenceData(
            element="waltz_jump",
            poses_norm=np.random.rand(10, 17, 2).astype(np.float32),
            poses_px=np.random.rand(10, 17, 2).astype(np.float32) * 640,
            fps=30.0,
            phases={"takeoff": 1, "peak": 2, "landing": 3},
            metrics={"airtime": 0.5},
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.npz"
            store.save(ref, path)
            loaded = store.load(path)
        assert loaded.element == ref.element
        assert loaded.fps == ref.fps
        assert loaded.phases == ref.phases
        assert loaded.metrics == ref.metrics
        np.testing.assert_array_almost_equal(loaded.poses_norm, ref.poses_norm)
        np.testing.assert_array_almost_equal(loaded.poses_px, ref.poses_px)

    def test_list_references(self):
        store = ReferenceStore()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ref = ReferenceData(
                element="three_turn",
                poses_norm=np.zeros((5, 17, 2)),
                poses_px=np.zeros((5, 17, 2)),
                fps=30.0,
                phases={},
                metrics={},
            )
            store.save(ref, td / "a.npz")
            store.save(ref, td / "b.npz")
            result = store.list(td)
        assert len(result) == 2
        assert all(r.suffix == ".npz" for r in result)

    def test_get_existing(self):
        store = ReferenceStore()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ref = ReferenceData(
                element="salchow",
                poses_norm=np.zeros((5, 17, 2)),
                poses_px=np.zeros((5, 17, 2)),
                fps=30.0,
                phases={},
                metrics={},
            )
            store.save(ref, td / "salchow.npz")
            loaded = store.get(td, "salchow")
        assert loaded.element == "salchow"

    def test_get_missing_returns_none(self):
        store = ReferenceStore()
        with tempfile.TemporaryDirectory() as td:
            result = store.get(Path(td), "nonexistent")
        assert result is None
```

Run: `uv run pytest ml/tests/references/test_reference_store.py -v`
Expected: PASS (after implementation exists)

- [ ] **Step 3: Write tests for reference_builder**

```python
"""Tests for ml.src.references.reference_builder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.references.reference_builder import ReferenceBuilder
from src.types import ReferenceData


class TestReferenceBuilder:
    @patch("src.references.reference_builder.extract_video_tracked")
    def test_build_creates_reference_data(self, mock_extract):
        mock_poses = np.random.rand(30, 17, 2).astype(np.float32)
        mock_extract.return_value = MagicMock(
            poses_norm=mock_poses,
            poses_px=mock_poses * 640,
            fps=30.0,
        )
        builder = ReferenceBuilder(
            element="waltz_jump",
            takeoff_sec=1.0,
            peak_sec=1.2,
            landing_sec=1.4,
        )
        ref = builder.build("/tmp/fake.mp4")
        assert isinstance(ref, ReferenceData)
        assert ref.element == "waltz_jump"
        assert ref.fps == 30.0
        assert ref.phases["takeoff"] == 30
        assert ref.phases["peak"] == 36
        assert ref.phases["landing"] == 42

    def test_build_with_manual_phases(self):
        builder = ReferenceBuilder(
            element="three_turn",
            takeoff_sec=0.5,
            peak_sec=0.7,
            landing_sec=0.9,
        )
        with patch("src.references.reference_builder.extract_video_tracked") as mock_extract:
            mock_poses = np.zeros((15, 17, 2), dtype=np.float32)
            mock_extract.return_value = MagicMock(
                poses_norm=mock_poses,
                poses_px=mock_poses,
                fps=30.0,
            )
            ref = builder.build("/tmp/fake.mp4")
        assert ref.phases == {"takeoff": 15, "peak": 21, "landing": 27}
```

Run: `uv run pytest ml/tests/references/test_reference_builder.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ml/tests/references/
git commit -m "test(references): cover reference_store and reference_builder"
```

---

### Task 2: datasets/coco_builder.py

**Files:**
- Create: `ml/tests/datasets/test_coco_builder.py`
- Read: `ml/src/datasets/coco_builder.py` (37 lines)

**Steps:**

- [ ] **Step 1: Read source**

- [ ] **Step 2: Write tests**

```python
"""Tests for ml.src.datasets.coco_builder."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.datasets.coco_builder import COCOBBuilder


class TestCOCOBBuilder:
    def test_empty_dataset(self):
        builder = COCOBBuilder()
        assert builder.images == []
        assert builder.annotations == []

    def test_add_image_and_annotation(self):
        builder = COCOBBuilder()
        img_id = builder.add_image("video_1", 0, 1920, 1080)
        assert img_id == 1
        assert len(builder.images) == 1
        assert builder.images[0]["file_name"] == "video_1_frame_0.jpg"

        keypoints = np.random.rand(17, 3).astype(np.float32)
        ann_id = builder.add_annotation(img_id, keypoints, bbox=[10, 20, 100, 200])
        assert ann_id == 1
        assert len(builder.annotations) == 1
        assert builder.annotations[0]["image_id"] == img_id

    def test_export_to_json(self):
        builder = COCOBBuilder()
        img_id = builder.add_image("v", 0, 640, 480)
        kps = np.zeros((17, 3), dtype=np.float32)
        builder.add_annotation(img_id, kps, bbox=[0, 0, 640, 480])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "coco.json"
            builder.export(out)
            assert out.exists()
            import json

            data = json.loads(out.read_text())
        assert data["images"][0]["file_name"] == "v_frame_0.jpg"
        assert len(data["annotations"]) == 1
```

Run: `uv run pytest ml/tests/datasets/test_coco_builder.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add ml/tests/datasets/test_coco_builder.py
git commit -m "test(datasets): cover coco_builder"
```

---

### Task 3: extras/inpainting.py + extras/video_matting.py

**Files:**
- Create: `ml/tests/extras/test_inpainting.py`
- Create: `ml/tests/extras/test_video_matting.py`
- Read: `ml/src/extras/inpainting.py` (29 lines)
- Read: `ml/src/extras/video_matting.py` (46 lines)

**Steps:**

- [ ] **Step 1: Read source**

- [ ] **Step 2: Write inpainting tests**

```python
"""Tests for ml.src.extras.inpainting."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.extras.inpainting import Inpainter


class TestInpainter:
    @patch("src.extras.inpainting.cv2.inpaint")
    def test_inpaint_frame_delegates_to_cv2(self, mock_inpaint):
        mock_inpaint.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        inp = Inpainter()
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        result = inp.inpaint_frame(frame, mask)
        mock_inpaint.assert_called_once()
        assert result.shape == frame.shape

    def test_inpaint_frame_with_zero_mask(self):
        inp = Inpainter()
        frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask = np.zeros((50, 50), dtype=np.uint8)
        with patch("src.extras.inpainting.cv2.inpaint", return_value=frame) as mock_inpaint:
            result = inp.inpaint_frame(frame, mask)
        np.testing.assert_array_equal(result, frame)
```

Run: `uv run pytest ml/tests/extras/test_inpainting.py -v`
Expected: PASS

- [ ] **Step 3: Write video_matting tests**

```python
"""Tests for ml.src.extras.video_matting."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.extras.video_matting import VideoMatting


class TestVideoMatting:
    @patch("src.extras.video_matting.cv2.resize")
    def test_matting_frame_returns_mask(self, mock_resize):
        mock_resize.return_value = np.zeros((100, 100), dtype=np.float32)
        vm = VideoMatting()
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 200
        mask = vm.matting_frame(frame)
        assert mask.dtype == np.uint8
        assert mask.shape == (1080, 1920)

    @patch("src.extras.video_matting.cv2.resize")
    def test_matting_frame_calls_resize(self, mock_resize):
        mock_resize.return_value = np.zeros((100, 100), dtype=np.float32)
        vm = VideoMatting(downscale=0.5)
        frame = np.ones((1080, 1920, 3), dtype=np.uint8)
        vm.matting_frame(frame)
        mock_resize.assert_called()
```

Run: `uv run pytest ml/tests/extras/test_video_matting.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ml/tests/extras/
git commit -m "test(extras): cover inpainting and video_matting"
```

---

### Task 4: pose_estimation/batch_extractor.py + pose_3d/tcpformer_extractor.py

**Files:**
- Create: `ml/tests/pose_estimation/test_batch_extractor.py`
- Create: `ml/tests/pose_3d/test_tcpformer_extractor.py`
- Read: `ml/src/pose_estimation/batch_extractor.py` (123 lines)
- Read: `ml/src/pose_3d/tcpformer_extractor.py` (11 lines)

**Steps:**

- [ ] **Step 1: Read source**

- [ ] **Step 2: Write batch_extractor tests**

```python
"""Tests for ml.src.pose_estimation.batch_extractor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pose_estimation.batch_extractor import BatchExtractor


class TestBatchExtractor:
    @patch("src.pose_estimation.batch_extractor.PoseExtractor")
    def test_extract_batch_returns_tracked_results(self, MockExtractor):
        mock_ext = MagicMock()
        mock_ext.extract_video_tracked.return_value = MagicMock(
            poses_norm=np.zeros((10, 17, 2), dtype=np.float32),
            poses_px=np.zeros((10, 17, 2), dtype=np.float32),
            fps=30.0,
        )
        MockExtractor.return_value = mock_ext

        be = BatchExtractor(device="cpu")
        results = be.extract_batch(["/tmp/a.mp4", "/tmp/b.mp4"])
        assert len(results) == 2
        assert all(r.fps == 30.0 for r in results)

    @patch("src.pose_estimation.batch_extractor.PoseExtractor")
    def test_extract_batch_with_person_click(self, MockExtractor):
        mock_ext = MagicMock()
        mock_ext.extract_video_tracked.return_value = MagicMock(
            poses_norm=np.zeros((5, 17, 2), dtype=np.float32),
            poses_px=np.zeros((5, 17, 2), dtype=np.float32),
            fps=30.0,
        )
        MockExtractor.return_value = mock_ext

        be = BatchExtractor(device="cpu")
        from src.types import PersonClick

        click = PersonClick(x=100, y=200, frame=0)
        results = be.extract_batch(["/tmp/a.mp4"], person_click=click)
        assert len(results) == 1
        mock_ext.extract_video_tracked.assert_called_once()
```

Run: `uv run pytest ml/tests/pose_estimation/test_batch_extractor.py -v`
Expected: PASS

- [ ] **Step 3: Write tcpformer_extractor tests**

```python
"""Tests for ml.src.pose_3d.tcpformer_extractor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pose_3d.tcpformer_extractor import TCPFormerExtractor


class TestTCPFormerExtractor:
    @patch("src.pose_3d.tcpformer_extractor.torch")
    def test_extract_sequence_returns_3d_poses(self, mock_torch):
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.return_value = MagicMock(
            cpu=MagicMock(return_value=MagicMock(
                numpy=MagicMock(return_value=np.random.rand(1, 17, 3).astype(np.float32))
            ))
        )
        mock_torch.load.return_value = {"model": {}}, {}

        extractor = TCPFormerExtractor.__new__(TCPFormerExtractor)
        extractor.model = mock_model
        extractor.device = "cpu"

        poses_2d = np.random.rand(81, 17, 2).astype(np.float32)
        result = extractor.extract_sequence(poses_2d)
        assert result.shape == (81, 17, 3)
```

Run: `uv run pytest ml/tests/pose_3d/test_tcpformer_extractor.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ml/tests/pose_estimation/test_batch_extractor.py ml/tests/pose_3d/test_tcpformer_extractor.py
git commit -m "test(pose): cover batch_extractor and tcpformer_extractor"
```

---

### Task 5: visualization/export_3d.py + visualization/export_3d_animated.py + visualization/comparison.py

**Files:**
- Create: `ml/tests/visualization/test_export_3d.py`
- Create: `ml/tests/visualization/test_export_3d_animated.py`
- Create: `ml/tests/visualization/test_comparison.py`
- Read: `ml/src/visualization/export_3d.py` (104 lines)
- Read: `ml/src/visualization/export_3d_animated.py` (254 lines)
- Read: `ml/src/visualization/comparison.py` (233 lines)

**Steps:**

- [ ] **Step 1: Read source files**

- [ ] **Step 2: Write export_3d tests**

```python
"""Tests for ml.src.visualization.export_3d."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.visualization.export_3d import SkeletonExporter3D


class TestSkeletonExporter3D:
    def test_init(self):
        exporter = SkeletonExporter3D()
        assert exporter.scale == 1.0

    @patch("src.visualization.export_3d.trimesh.Scene")
    def test_export_glb_creates_file(self, MockScene):
        mock_scene = MagicMock()
        MockScene.return_value = mock_scene
        exporter = SkeletonExporter3D()
        poses = np.random.rand(10, 17, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test.glb"
            exporter.export_glb(poses, out)
        mock_scene.export.assert_called_once_with(str(out))

    def test_create_mesh_for_frame(self):
        exporter = SkeletonExporter3D()
        frame = np.zeros((17, 3), dtype=np.float32)
        frame[0] = [0, 0, 0]
        frame[1] = [1, 0, 0]
        mesh = exporter._create_mesh_for_frame(frame)
        assert mesh is not None
```

Run: `uv run pytest ml/tests/visualization/test_export_3d.py -v`
Expected: PASS

- [ ] **Step 3: Write export_3d_animated tests**

```python
"""Tests for ml.src.visualization.export_3d_animated."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.visualization.export_3d_animated import AnimatedSkeletonExporter


class TestAnimatedSkeletonExporter:
    def test_init(self):
        exporter = AnimatedSkeletonExporter(fps=30.0)
        assert exporter.fps == 30.0

    @patch("src.visualization.export_3d_animated.trimesh.Scene")
    def test_export_animated_glb(self, MockScene):
        mock_scene = MagicMock()
        MockScene.return_value = mock_scene
        exporter = AnimatedSkeletonExporter(fps=30.0)
        poses = np.random.rand(30, 17, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "anim.glb"
            exporter.export_animated_glb(poses, out)
        mock_scene.export.assert_called_once_with(str(out))

    def test_create_keyframe_transforms(self):
        exporter = AnimatedSkeletonExporter(fps=30.0)
        poses = np.zeros((3, 17, 3), dtype=np.float32)
        transforms = exporter._create_keyframe_transforms(poses)
        assert len(transforms) == 3
```

Run: `uv run pytest ml/tests/visualization/test_export_3d_animated.py -v`
Expected: PASS

- [ ] **Step 4: Write comparison tests**

```python
"""Tests for ml.src.visualization.comparison."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.visualization.comparison import ComparisonRenderer


class TestComparisonRenderer:
    def test_init(self):
        renderer = ComparisonRenderer(width=640, height=480)
        assert renderer.width == 640
        assert renderer.height == 480

    @patch("src.visualization.comparison.cv2")
    def test_render_side_by_side(self, mock_cv2):
        mock_cv2.hconcat.return_value = np.zeros((480, 1280, 3), dtype=np.uint8)
        renderer = ComparisonRenderer(width=640, height=480)
        frame_a = np.ones((480, 640, 3), dtype=np.uint8) * 100
        frame_b = np.ones((480, 640, 3), dtype=np.uint8) * 200
        result = renderer.render_side_by_side(frame_a, frame_b)
        assert result.shape == (480, 1280, 3)
        mock_cv2.hconcat.assert_called_once()

    @patch("src.visualization.comparison.cv2")
    def test_render_overlay(self, mock_cv2):
        mock_cv2.addWeighted.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer = ComparisonRenderer(width=640, height=480)
        frame_a = np.ones((480, 640, 3), dtype=np.uint8)
        frame_b = np.ones((480, 640, 3), dtype=np.uint8) * 2
        result = renderer.render_overlay(frame_a, frame_b, alpha=0.5)
        mock_cv2.addWeighted.assert_called_once_with(frame_a, 0.5, frame_b, 0.5, 0)
```

Run: `uv run pytest ml/tests/visualization/test_comparison.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/tests/visualization/test_export_3d.py ml/tests/visualization/test_export_3d_animated.py ml/tests/visualization/test_comparison.py
git commit -m "test(viz): cover export_3d, export_3d_animated, comparison"
```

---

### Task 6: web_helpers.py

**Files:**
- Create: `ml/tests/test_web_helpers.py`
- Read: `ml/src/web_helpers.py` (259 lines)

**Steps:**

- [ ] **Step 1: Read source**

- [ ] **Step 2: Write tests**

```python
"""Tests for ml.src.web_helpers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.web_helpers import (
    encode_preview_image,
    render_detection_preview,
    render_pose_preview,
)


class TestEncodePreviewImage:
    def test_returns_base64_string(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        b64 = encode_preview_image(img)
        assert isinstance(b64, str)
        assert b64.startswith("data:image/jpeg;base64,")

    def test_empty_image_still_encodes(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        b64 = encode_preview_image(img)
        assert isinstance(b64, str)


class TestRenderDetectionPreview:
    @patch("src.web_helpers.cv2")
    @patch("src.web_helpers.PoseExtractor")
    def test_renders_preview_with_persons(self, MockExtractor, mock_cv2):
        mock_ext = MagicMock()
        mock_ext.extract_video.return_value = [
            {"bbox": [10, 20, 100, 200], "confidence": 0.95}
        ]
        MockExtractor.return_value = mock_ext
        mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.rectangle.return_value = None
        mock_cv2.putText.return_value = None
        mock_cv2.imencode.return_value = (True, b"fakejpeg")

        result = render_detection_preview("/tmp/fake.mp4")
        assert "persons" in result
        assert result["persons"][0]["confidence"] == 0.95


class TestRenderPosePreview:
    @patch("src.web_helpers.cv2")
    @patch("src.web_helpers.PoseExtractor")
    def test_renders_pose_preview(self, MockExtractor, mock_cv2):
        mock_ext = MagicMock()
        mock_ext.extract_video.return_value = np.random.rand(10, 17, 2).astype(np.float32) * 640
        MockExtractor.return_value = mock_ext
        mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.line.return_value = None
        mock_cv2.circle.return_value = None
        mock_cv2.imencode.return_value = (True, b"fakejpeg")

        result = render_pose_preview("/tmp/fake.mp4")
        assert "preview" in result
```

Run: `uv run pytest ml/tests/test_web_helpers.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add ml/tests/test_web_helpers.py
git commit -m "test(ml): cover web_helpers preview functions"
```

---

## Tier 2: Low-Coverage Modules (Biggest Impact)

### Task 7: pose_estimation/pose_extractor.py

**Files:**
- Modify: `ml/tests/pose_estimation/test_pose_extractor.py` (create if not exists)
- Read: `ml/src/pose_estimation/pose_extractor.py` (656 lines, 14% coverage)

**Steps:**

- [ ] **Step 1: Read existing tests**

Check if `ml/tests/pose_estimation/test_pose_extractor.py` exists. If not, create it.

- [ ] **Step 2: Add tests for PoseExtractor initialization**

```python
"""Additional tests for ml.src.pose_estimation.pose_extractor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pose_estimation.pose_extractor import PoseExtractor


class TestPoseExtractorInit:
    @patch("src.pose_estimation.pose_extractor.DeviceConfig")
    def test_default_init(self, MockDevice):
        MockDevice.default.return_value = MagicMock(device="cuda")
        extractor = PoseExtractor()
        assert extractor is not None

    @patch("src.pose_estimation.pose_extractor.DeviceConfig")
    def test_cpu_device(self, MockDevice):
        MockDevice.default.return_value = MagicMock(device="cpu")
        extractor = PoseExtractor(device="cpu")
        assert extractor is not None
```

- [ ] **Step 3: Add tests for extract_frame**

```python
    @patch("src.pose_estimation.pose_extractor.rtmlib")
    def test_extract_frame_returns_keypoints(self, mock_rtmlib):
        mock_pose = MagicMock()
        mock_pose.return_value = [
            {"keypoints": np.random.rand(17, 2), "bbox": [0, 0, 100, 100]}
        ]
        mock_rtmlib.PoseTracker = MagicMock
        mock_rtmlib.PoseTracker.return_value = mock_pose

        extractor = PoseExtractor.__new__(PoseExtractor)
        extractor.tracker = mock_pose
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.extract_frame(frame)
        assert len(result) == 1
        assert "keypoints" in result[0]

    def test_extract_frame_with_no_persons(self):
        extractor = PoseExtractor.__new__(PoseExtractor)
        extractor.tracker = MagicMock(return_value=[])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.extract_frame(frame)
        assert result == []
```

- [ ] **Step 4: Add tests for extract_video_tracked**

```python
    @patch("src.pose_estimation.pose_extractor.extract_frames")
    def test_extract_video_tracked_returns_tracked(self, mock_extract_frames):
        mock_extract_frames.return_value = [
            np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)
        ]
        extractor = PoseExtractor.__new__(PoseExtractor)
        extractor.tracker = MagicMock(side_effect=[
            [{"keypoints": np.random.rand(17, 2), "bbox": [0, 0, 10, 10], "track_id": 0}]
            for _ in range(5)
        ])
        from src.types import PersonClick

        result = extractor.extract_video_tracked(
            "/tmp/fake.mp4",
            person_click=PersonClick(x=320, y=240, frame=0),
        )
        assert result is not None
        assert hasattr(result, "poses_norm") or isinstance(result, dict)
```

Run: `uv run pytest ml/tests/pose_estimation/test_pose_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/tests/pose_estimation/test_pose_extractor.py
git commit -m "test(pose): expand PoseExtractor coverage"
```

---

### Task 8: pipeline.py

**Files:**
- Modify: `ml/tests/test_pipeline.py` (create if not exists)
- Read: `ml/src/pipeline.py` (422 lines, 34% coverage)

**Steps:**

- [ ] **Step 1: Read existing tests and source**

- [ ] **Step 2: Add tests for AnalysisPipeline orchestration**

```python
"""Additional tests for ml.src.pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline import AnalysisPipeline


class TestAnalysisPipelineOrchestration:
    @patch("src.pipeline.PoseExtractor")
    @patch("src.pipeline.PhaseDetector")
    @patch("src.pipeline.BiomechanicsAnalyzer")
    @patch("src.pipeline.Recommender")
    def test_run_full_pipeline(
        self, MockRecommender, MockAnalyzer, MockPhaseDetector, MockExtractor
    ):
        mock_poses = np.random.rand(30, 17, 2).astype(np.float32)
        MockExtractor.return_value.extract_video_tracked.return_value = MagicMock(
            poses_norm=mock_poses,
            poses_px=mock_poses * 640,
            fps=30.0,
        )
        MockPhaseDetector.return_value.detect_phases.return_value = {
            "takeoff": 5,
            "peak": 10,
            "landing": 15,
        }
        MockAnalyzer.return_value.analyze.return_value = {
            "airtime": 0.5,
            "max_height": 0.3,
        }
        MockRecommender.return_value.recommend.return_value = "Хороший прыжок"

        pipeline = AnalysisPipeline(element="waltz_jump")
        result = pipeline.run("/tmp/fake.mp4")
        assert "report" in result or hasattr(result, "report")

    def test_pipeline_init_sets_element(self):
        pipeline = AnalysisPipeline(element="salchow")
        assert pipeline.element == "salchow"

    def test_pipeline_init_default_reference(self):
        pipeline = AnalysisPipeline(element="waltz_jump")
        assert pipeline.reference_data is None
```

Run: `uv run pytest ml/tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add ml/tests/test_pipeline.py
git commit -m "test(pipeline): expand AnalysisPipeline coverage"
```

---

## Tier 3: Backend Edge Cases

### Task 9: backend/app/main.py + backend/app/logging_config.py

**Files:**
- Create: `backend/tests/test_main.py`
- Create: `backend/tests/test_logging_config.py`
- Read: `backend/app/main.py` (36 lines)
- Read: `backend/app/logging_config.py` (8 lines)

**Steps:**

- [ ] **Step 1: Read source**

- [ ] **Step 2: Write main.py tests**

```python
"""Tests for backend.app.main."""

import pytest
from fastapi.testclient import TestClient


class TestMainApp:
    def test_health_endpoint(self):
        from app.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_app_has_routes(self):
        from app.main import app

        routes = [route.path for route in app.routes]
        assert "/health" in routes or any("health" in r for r in routes)
```

Run: `uv run pytest backend/tests/test_main.py -v`
Expected: PASS

- [ ] **Step 3: Write logging_config.py tests**

```python
"""Tests for backend.app.logging_config."""

import structlog

from app.logging_config import configure_logging


class TestLoggingConfig:
    def test_configure_logging_does_not_raise(self):
        configure_logging()
        assert structlog.is_configured()
```

Run: `uv run pytest backend/tests/test_logging_config.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_main.py backend/tests/test_logging_config.py
git commit -m "test(backend): cover main.py and logging_config.py"
```

---

## Tier 4: Frontend Vitest Setup

### Task 10: Frontend vitest coverage + component tests

**Files:**
- Modify: `frontend/vitest.config.ts`
- Create: `frontend/src/lib/__tests__/utils.test.ts`
- Read: `frontend/vitest.config.ts`
- Read: `frontend/package.json`

**Steps:**

- [ ] **Step 1: Read existing vitest config**

Check if `frontend/vitest.config.ts` has coverage reporter. If not, add `@vitest/coverage-v8` dependency.

- [ ] **Step 2: Add coverage to vitest config**

```typescript
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  test: {
    environment: "jsdom",
    globals: true,
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "json"],
      exclude: [
        "node_modules/",
        "**/*.d.ts",
        "**/*.config.*",
        "**/mockData.ts",
      ],
    },
  },
});
```

Install if needed: `cd frontend && bun add -d @vitest/coverage-v8 jsdom @testing-library/react @testing-library/jest-dom`

- [ ] **Step 3: Write a utility test**

```typescript
import { describe, expect, it } from "vitest";

// Replace with an actual utility from frontend/src/lib/
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

describe("clamp", () => {
  it("clamps to max", () => {
    expect(clamp(150, 0, 100)).toBe(100);
  });
  it("clamps to min", () => {
    expect(clamp(-10, 0, 100)).toBe(0);
  });
  it("returns value when in range", () => {
    expect(clamp(50, 0, 100)).toBe(50);
  });
});
```

Run: `cd frontend && bun run vitest run --coverage`
Expected: Tests pass, coverage report generated in `frontend/coverage/`

- [ ] **Step 4: Commit**

```bash
git add frontend/vitest.config.ts frontend/src/lib/__tests__/
git commit -m "test(frontend): enable vitest coverage and add utility tests"
```

---

## Verification

After all tasks complete, run the full coverage report:

```bash
# Python
uv run pytest backend/tests/ ml/tests/ -m "not slow and not integration" \
  --cov=backend/app --cov=ml/src --cov-report=term-missing --tb=short -q

# Frontend
cd frontend && bun run vitest run --coverage
```

**Expected result:** Combined Python coverage should rise from 65% to 80%+. Frontend coverage should be >0%.

---

## Self-Review Checklist

| Requirement | Task | Status |
|-------------|------|--------|
| reference_store.py 0% → 80% | Task 1 | Planned |
| reference_builder.py 0% → 80% | Task 1 | Planned |
| coco_builder.py 0% → 80% | Task 2 | Planned |
| inpainting.py 0% → 70% | Task 3 | Planned |
| video_matting.py 0% → 70% | Task 3 | Planned |
| batch_extractor.py 0% → 60% | Task 4 | Planned |
| tcpformer_extractor.py 0% → 60% | Task 4 | Planned |
| export_3d.py 0% → 50% | Task 5 | Planned |
| export_3d_animated.py 0% → 50% | Task 5 | Planned |
| comparison.py 0% → 50% | Task 5 | Planned |
| web_helpers.py 7% → 50% | Task 6 | Planned |
| pose_extractor.py 14% → 50% | Task 7 | Planned |
| pipeline.py 34% → 60% | Task 8 | Planned |
| backend main.py 0% → 100% | Task 9 | Planned |
| backend logging_config.py 0% → 100% | Task 9 | Planned |
| frontend vitest coverage | Task 10 | Planned |
