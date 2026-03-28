"""BlazePose 33-keypoint skeleton hierarchy tree.

Provides hierarchical structure for skeleton operations:
- Bone pair extraction for angle calculations
- Parent-child relationships for constraints
- Tree traversal for biomechanics analysis

Based on:
- BlazePose 33kp topology
- Pose2Sim skeleton structure
"""

from dataclasses import dataclass

from anytree import Node, RenderTree

from .types import BKey


@dataclass
class BoneNode:
    """Node in skeleton hierarchy.

    Attributes:
        name: Joint/bone name.
        keypoint_idx: BlazePose keypoint index (None for intermediate nodes).
        children: Child bones/joints.
    """

    name: str
    keypoint_idx: int | None = None
    children: list["BoneNode"] | None = None

    def to_anytree(self, parent: Node | None = None) -> Node:
        """Convert to anytree Node.

        Args:
            parent: Parent node in tree.

        Returns:
            anytree Node with this node's data.
        """
        node = Node(self.name, id=self.keypoint_idx, parent=parent)
        if self.children:
            for child in self.children:
                child.to_anytree(node)
        return node


# Define BlazePose 33kp hierarchy
# Based on: https://google.github.io/mediapipe/solutions/pose.html
BLAZEPOSE_HIERARCHY = BoneNode(
    name="root",
    children=[
        # Body center (spine)
        BoneNode(name="nose", keypoint_idx=BKey.NOSE),
        BoneNode(name="left_eye_inner", keypoint_idx=BKey.LEFT_EYE_INNER),
        BoneNode(name="left_eye", keypoint_idx=BKey.LEFT_EYE),
        BoneNode(name="left_eye_outer", keypoint_idx=BKey.LEFT_EYE_OUTER),
        BoneNode(name="right_eye_inner", keypoint_idx=BKey.RIGHT_EYE_INNER),
        BoneNode(name="right_eye", keypoint_idx=BKey.RIGHT_EYE),
        BoneNode(name="right_eye_outer", keypoint_idx=BKey.RIGHT_EYE_OUTER),
        BoneNode(name="left_ear", keypoint_idx=BKey.LEFT_EAR),
        BoneNode(name="right_ear", keypoint_idx=BKey.RIGHT_EAR),
        BoneNode(name="mouth_left", keypoint_idx=BKey.MOUTH_LEFT),
        BoneNode(name="mouth_right", keypoint_idx=BKey.MOUTH_RIGHT),
        # Left shoulder and arm
        BoneNode(
            name="left_shoulder",
            keypoint_idx=BKey.LEFT_SHOULDER,
            children=[
                BoneNode(
                    name="left_elbow",
                    keypoint_idx=BKey.LEFT_ELBOW,
                    children=[
                        BoneNode(
                            name="left_wrist",
                            keypoint_idx=BKey.LEFT_WRIST,
                            children=[
                                BoneNode(name="left_pinky", keypoint_idx=BKey.LEFT_PINKY),
                                BoneNode(name="left_index", keypoint_idx=BKey.LEFT_INDEX),
                                BoneNode(name="left_thumb", keypoint_idx=BKey.LEFT_THUMB),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # Right shoulder and arm
        BoneNode(
            name="right_shoulder",
            keypoint_idx=BKey.RIGHT_SHOULDER,
            children=[
                BoneNode(
                    name="right_elbow",
                    keypoint_idx=BKey.RIGHT_ELBOW,
                    children=[
                        BoneNode(
                            name="right_wrist",
                            keypoint_idx=BKey.RIGHT_WRIST,
                            children=[
                                BoneNode(name="right_pinky", keypoint_idx=BKey.RIGHT_PINKY),
                                BoneNode(name="right_index", keypoint_idx=BKey.RIGHT_INDEX),
                                BoneNode(name="right_thumb", keypoint_idx=BKey.RIGHT_THUMB),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # Left hip and leg
        BoneNode(
            name="left_hip",
            keypoint_idx=BKey.LEFT_HIP,
            children=[
                BoneNode(
                    name="left_knee",
                    keypoint_idx=BKey.LEFT_KNEE,
                    children=[
                        BoneNode(
                            name="left_ankle",
                            keypoint_idx=BKey.LEFT_ANKLE,
                            children=[
                                BoneNode(name="left_heel", keypoint_idx=BKey.LEFT_HEEL),
                                BoneNode(name="left_foot_index", keypoint_idx=BKey.LEFT_FOOT_INDEX),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # Right hip and leg
        BoneNode(
            name="right_hip",
            keypoint_idx=BKey.RIGHT_HIP,
            children=[
                BoneNode(
                    name="right_knee",
                    keypoint_idx=BKey.RIGHT_KNEE,
                    children=[
                        BoneNode(
                            name="right_ankle",
                            keypoint_idx=BKey.RIGHT_ANKLE,
                            children=[
                                BoneNode(name="right_heel", keypoint_idx=BKey.RIGHT_HEEL),
                                BoneNode(name="right_foot_index", keypoint_idx=BKey.RIGHT_FOOT_INDEX),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# Build the tree once at import time
_ROOT_NODE = BLAZEPOSE_HIERARCHY.to_anytree()


def get_skeleton_tree() -> Node:
    """Get the BlazePose skeleton hierarchy tree.

    Returns:
        Root node of the skeleton tree (anytree Node).
    """
    return _ROOT_NODE


def get_bone_pairs(tree: Node | None = None) -> list[tuple[int, int]]:
    """Extract all bone pairs (parent-child) from skeleton hierarchy.

    Useful for:
    - Drawing skeleton connections
    - Computing bone lengths
    - Applying kinematic constraints

    Args:
        tree: Skeleton tree (uses default BlazePose tree if None).

    Returns:
        List of (parent_idx, child_idx) tuples for all bones.
        Excludes nodes with None keypoint indices.
    """
    if tree is None:
        tree = _ROOT_NODE

    pairs = []
    for parent in tree.descendants:
        for child in parent.children:
            parent_idx = getattr(parent, "id", None)
            child_idx = getattr(child, "id", None)
            if parent_idx is not None and child_idx is not None:
                pairs.append((int(parent_idx), int(child_idx)))

    return pairs


def get_bone_names(tree: Node | None = None) -> dict[int, str]:
    """Get mapping from BlazePose keypoint indices to joint names.

    Args:
        tree: Skeleton tree (uses default BlazePose tree if None).

    Returns:
        Dictionary mapping keypoint_idx -> joint_name.
    """
    if tree is None:
        tree = _ROOT_NODE

    return {
        int(node.id): node.name
        for node in tree.descendants
        if hasattr(node, "id") and node.id is not None
    }


def get_children_of(parent_idx: int, tree: Node | None = None) -> list[int]:
    """Get child keypoint indices for a given parent joint.

    Args:
        parent_idx: BlazePose keypoint index.
        tree: Skeleton tree (uses default BlazePose tree if None).

    Returns:
        List of child keypoint indices.
    """
    if tree is None:
        tree = _ROOT_NODE

    for node in tree.descendants:
        if getattr(node, "id", None) == parent_idx:
            return [
                int(child.id)
                for child in node.children
                if hasattr(child, "id") and child.id is not None
            ]

    return []


def render_tree(tree: Node | None = None) -> str:
    """Render skeleton tree as ASCII string.

    Useful for debugging and documentation.

    Args:
        tree: Skeleton tree (uses default BlazePose tree if None).

    Returns:
        ASCII representation of the tree.
    """
    if tree is None:
        tree = _ROOT_NODE

    lines = []
    for pre, _, node in RenderTree(tree):
        keypoint_info = f" (kp={node.id})" if hasattr(node, "id") and node.id is not None else ""
        lines.append(f"{pre}{node.name}{keypoint_info}")

    return "\n".join(lines)


# Standard bone pairs for drawing (BlazePose topology)
# These match MediaPipe's recommended skeleton connections
BLAZEPOSE_BONE_PAIRS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye/ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye/ear
    (9, 10),  # Mouth
    # Torso
    (11, 12),  # Shoulders
    (11, 23), (12, 24),  # Shoulders to hips
    (23, 24),  # Hips
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Shoulder-elbow-wrist-fingers
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Shoulder-elbow-wrist-fingers
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31),  # Hip-knee-ankle-foot
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # Hip-knee-ankle-foot
]

# Subset for biomechanics analysis (major joints only)
BIOMECHANICS_BONE_PAIRS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (23, 25), (25, 27), (27, 29),  # Left leg
    (24, 26), (26, 28), (28, 30),  # Right leg
    (11, 23), (12, 24),  # Sides
    (11, 12), (23, 24),  # Across
]
