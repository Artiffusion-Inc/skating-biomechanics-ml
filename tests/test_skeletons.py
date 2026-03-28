"""Tests for skeleton hierarchy module."""


from src.skeletons import (
    BIOMECHANICS_BONE_PAIRS,
    BLAZEPOSE_BONE_PAIRS,
    get_bone_names,
    get_bone_pairs,
    get_children_of,
    get_skeleton_tree,
    render_tree,
)
from src.types import BKey


class TestSkeletonTree:
    """Test skeleton tree structure and utilities."""

    def test_get_skeleton_tree(self):
        """Test that skeleton tree can be retrieved."""
        tree = get_skeleton_tree()
        assert tree is not None
        assert tree.name == "root"
        assert tree.children is not None
        assert len(tree.children) > 0

    def test_tree_has_all_blazepose_keypoints(self):
        """Test that all BlazePose keypoints are in the tree."""
        tree = get_skeleton_tree()
        keypoint_indices = set()

        for node in tree.descendants:
            if hasattr(node, "id") and node.id is not None:
                keypoint_indices.add(int(node.id))

        # Should have all 33 BlazePose keypoints (0-32)
        expected_indices = set(range(33))
        assert keypoint_indices == expected_indices

    def test_get_bone_pairs(self):
        """Test bone pair extraction."""
        pairs = get_bone_pairs()

        assert len(pairs) > 0
        # All pairs should be tuples of (parent_idx, child_idx)
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(pair[0], int)
            assert isinstance(pair[1], int)
            # Indices should be valid BlazePose keypoints
            assert 0 <= pair[0] < 33
            assert 0 <= pair[1] < 33

    def test_bone_pairs_are_unique(self):
        """Test that bone pairs are unique."""
        pairs = get_bone_pairs()
        assert len(pairs) == len(set(pairs))

    def test_get_bone_names(self):
        """Test bone name mapping."""
        names = get_bone_names()

        assert len(names) == 33  # All BlazePose keypoints
        assert all(isinstance(k, int) for k in names)
        assert all(isinstance(v, str) for v in names.values())

        # Check specific keypoints
        assert names[BKey.NOSE] == "nose"
        assert names[BKey.LEFT_SHOULDER] == "left_shoulder"
        assert names[BKey.RIGHT_SHOULDER] == "right_shoulder"
        assert names[BKey.LEFT_HIP] == "left_hip"
        assert names[BKey.RIGHT_HIP] == "right_hip"

    def test_get_children_of(self):
        """Test getting children of a joint."""
        # Left shoulder should have left elbow as child
        left_shoulder_children = get_children_of(BKey.LEFT_SHOULDER)
        assert BKey.LEFT_ELBOW in left_shoulder_children

        # Left elbow should have left wrist as child
        left_elbow_children = get_children_of(BKey.LEFT_ELBOW)
        assert BKey.LEFT_WRIST in left_elbow_children

        # Left wrist should have fingers as children
        left_wrist_children = get_children_of(BKey.LEFT_WRIST)
        assert BKey.LEFT_PINKY in left_wrist_children
        assert BKey.LEFT_INDEX in left_wrist_children
        assert BKey.LEFT_THUMB in left_wrist_children

    def test_get_children_of_invalid_joint(self):
        """Test getting children of invalid joint returns empty list."""
        children = get_children_of(9999)
        assert children == []

    def test_render_tree(self):
        """Test tree rendering."""
        tree_str = render_tree()
        assert isinstance(tree_str, str)
        assert len(tree_str) > 0
        assert "root" in tree_str
        assert "nose" in tree_str
        assert "left_shoulder" in tree_str
        assert "right_shoulder" in tree_str

    def test_render_tree_format(self):
        """Test that tree rendering has proper structure."""
        tree_str = render_tree()
        lines = tree_str.split("\n")

        # Should have multiple lines
        assert len(lines) > 10

        # Root should be first line
        assert lines[0].startswith("root")

    def test_blazepose_bone_pairs_constant(self):
        """Test BLAZEPOSE_BONE_PAIRS constant."""
        assert len(BLAZEPOSE_BONE_PAIRS) > 0

        # All should be valid keypoint pairs
        for pair in BLAZEPOSE_BONE_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert 0 <= pair[0] < 33
            assert 0 <= pair[1] < 33

    def test_biomechanics_bone_pairs_constant(self):
        """Test BIOMECHANICS_BONE_PAIRS constant."""
        assert len(BIOMECHANICS_BONE_PAIRS) > 0
        assert len(BIOMECHANICS_BONE_PAIRS) < len(BLAZEPOSE_BONE_PAIRS)  # Subset

        # Should contain major limb segments
        expected_pairs = [
            (BKey.LEFT_SHOULDER, BKey.LEFT_ELBOW),
            (BKey.LEFT_ELBOW, BKey.LEFT_WRIST),
            (BKey.RIGHT_SHOULDER, BKey.RIGHT_ELBOW),
            (BKey.RIGHT_ELBOW, BKey.RIGHT_WRIST),
        ]

        for pair in expected_pairs:
            assert pair in BIOMECHANICS_BONE_PAIRS

    def test_tree_hierarchy_structure(self):
        """Test that tree hierarchy is anatomically correct."""
        tree = get_skeleton_tree()

        # Find left shoulder node
        left_shoulder = None
        for node in tree.descendants:
            if hasattr(node, "id") and node.id == BKey.LEFT_SHOULDER:
                left_shoulder = node
                break

        assert left_shoulder is not None
        assert left_shoulder.name == "left_shoulder"

        # Should have left elbow as child
        left_elbow = None
        for child in left_shoulder.children:
            if hasattr(child, "id") and child.id == BKey.LEFT_ELBOW:
                left_elbow = child
                break

        assert left_elbow is not None
        assert left_elbow.name == "left_elbow"

    def test_finger_keypoints_in_tree(self):
        """Test that all finger keypoints are in the tree."""
        tree = get_skeleton_tree()
        keypoint_indices = set()

        for node in tree.descendants:
            if hasattr(node, "id") and node.id is not None:
                keypoint_indices.add(int(node.id))

        # Check specific finger keypoints
        finger_keypoints = [
            BKey.LEFT_PINKY,
            BKey.LEFT_INDEX,
            BKey.LEFT_THUMB,
            BKey.RIGHT_PINKY,
            BKey.RIGHT_INDEX,
            BKey.RIGHT_THUMB,
        ]

        for kp in finger_keypoints:
            assert kp in keypoint_indices

    def test_foot_keypoints_in_tree(self):
        """Test that all foot keypoints are in the tree."""
        tree = get_skeleton_tree()
        keypoint_indices = set()

        for node in tree.descendants:
            if hasattr(node, "id") and node.id is not None:
                keypoint_indices.add(int(node.id))

        # Check specific foot keypoints
        foot_keypoints = [
            BKey.LEFT_HEEL,
            BKey.LEFT_FOOT_INDEX,
            BKey.RIGHT_HEEL,
            BKey.RIGHT_FOOT_INDEX,
        ]

        for kp in foot_keypoints:
            assert kp in keypoint_indices
