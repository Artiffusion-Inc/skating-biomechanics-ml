"""Tests for unified visualization pipeline."""

import numpy as np

from src.visualization.pipeline import VizPipeline


def _fake_meta(w=640, h=480, fps=30, num_frames=10):
    from types import SimpleNamespace

    return SimpleNamespace(width=w, height=h, fps=fps, num_frames=num_frames)


class TestVizPipelineInit:
    def test_minimal_init(self):
        meta = _fake_meta()
        poses = np.random.rand(10, 17, 2).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses)
        assert pipe.layer == 0
        assert len(pipe.layers) == 0

    def test_layer_1_builds_velocity_and_trail(self):
        meta = _fake_meta()
        poses = np.random.rand(10, 17, 2).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses, layer=1)
        assert len(pipe.layers) >= 2

    def test_layer_2_adds_axis(self):
        meta = _fake_meta()
        poses = np.random.rand(10, 17, 2).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses, layer=1)
        l1_count = len(pipe.layers)
        pipe2 = VizPipeline(meta=meta, poses_norm=poses, layer=2)
        assert len(pipe2.layers) > l1_count

    def test_with_poses_3d(self):
        meta = _fake_meta()
        poses = np.random.rand(10, 17, 2).astype(np.float32)
        poses_3d = np.random.rand(10, 17, 3).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses, poses_3d=poses_3d)
        assert pipe.poses_3d is not None


class TestVizPipelineBuildLayers:
    def test_rebuild_layers_changes_count(self):
        meta = _fake_meta()
        poses = np.random.rand(10, 17, 2).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses, layer=0)
        assert len(pipe.layers) == 0
        pipe.layer = 1
        pipe.build_layers()
        assert len(pipe.layers) >= 2
