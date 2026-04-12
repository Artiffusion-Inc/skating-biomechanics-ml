"""COCO 17-keypoint skeleton graph for InfoGCN.

COCO 17kp layout:
  0:nose 1:Leye 2:Reye 3:Lear 4:Rear
  5:Lshoulder 6:Rshoulder 7:Lelbow 8:Relbow 9:Lwrist 10:Rwrist
  11:Lhip 12:Rhip 13:Lknee 14:Rknee 15:Lankle 16:Rankle
"""

import numpy as np

NUM_NODE = 17

# Edges: (parent, child) — "inward" means toward root (mid-torso)
INWARD_ORI_INDEX = [
    # head
    (1, 0),
    (2, 0),
    (3, 1),
    (4, 2),
    # torso horizontal
    (6, 5),  # Rshoulder → Lshoulder
    (12, 11),  # Rhip → Lhip
    # torso vertical
    (11, 5),  # Lhip → Lshoulder
    (12, 6),  # Rhip → Rshoulder
    # left arm
    (7, 5),  # Lelbow → Lshoulder
    (9, 7),  # Lwrist → Lelbow
    # right arm
    (8, 6),  # Relbow → Rshoulder
    (10, 8),  # Rwrist → Relbow
    # left leg
    (13, 11),  # Lknee → Lhip
    (15, 13),  # Lankle → Lknee
    # right leg
    (14, 12),  # Rknee → Rhip
    (16, 14),  # Rankle → Rknee
]

INWARD = [(i - 1, j - 1) for (i, j) in INWARD_ORI_INDEX]
OUTWARD = [(j, i) for (i, j) in INWARD]
SELF_LINK = [(i, i) for i in range(NUM_NODE)]
NEIGHBOR = INWARD + OUTWARD


def get_adjacency_matrix(edges, num_node):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1
    return A


def normalize_adjacency_matrix(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(Dn, A)
    return AD


class Graph:
    def __init__(self, labeling_mode="spatial"):
        self.num_node = NUM_NODE
        self.self_link = SELF_LINK
        self.inward = INWARD
        self.outward = OUTWARD
        self.neighbor = NEIGHBOR

        self.A_outward_binary = get_adjacency_matrix(self.outward, self.num_node)
        self.A_binary = get_adjacency_matrix(self.neighbor, self.num_node)
        self.A_norm = normalize_adjacency_matrix(self.A_binary + 2 * np.eye(self.num_node))
