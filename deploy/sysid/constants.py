"""G1 constants for the mjlab sysid port.

Mirrors `deploy/real/g1_robot_constants.py` (the pinned mjlab revision) plus the
KNEES_BENT default pose, so the sysid tooling can run without importing mjlab.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
NUM_JOINTS = len(JOINT_NAMES)
LEG_JOINTS = tuple(range(12))

S5020 = 14.25062309787429
D5020 = 0.907222843292423
S7520_14 = 40.17923863450712
D7520_14 = 2.557889775413375
S7520_22 = 99.09842777666111
D7520_22 = 6.308801853496639
S4010 = 16.77832748089279
D4010 = 1.06814150219

E7520_14, E7520_22, E5020, E4010 = 88.0, 139.0, 25.0, 5.0

_KP = np.array(
    [
        S7520_14, S7520_22, S7520_14, S7520_22, 2 * S5020, 2 * S5020,
        S7520_14, S7520_22, S7520_14, S7520_22, 2 * S5020, 2 * S5020,
        S7520_14, 2 * S5020, 2 * S5020,
        S5020, S5020, S5020, S5020, S5020, S4010, S4010,
        S5020, S5020, S5020, S5020, S5020, S4010, S4010,
    ],
    dtype=np.float64,
)
_KD = np.array(
    [
        D7520_14, D7520_22, D7520_14, D7520_22, 2 * D5020, 2 * D5020,
        D7520_14, D7520_22, D7520_14, D7520_22, 2 * D5020, 2 * D5020,
        D7520_14, 2 * D5020, 2 * D5020,
        D5020, D5020, D5020, D5020, D5020, D4010, D4010,
        D5020, D5020, D5020, D5020, D5020, D4010, D4010,
    ],
    dtype=np.float64,
)
_EFFORT = np.array(
    [
        E7520_14, E7520_22, E7520_14, E7520_22, 2 * E5020, 2 * E5020,
        E7520_14, E7520_22, E7520_14, E7520_22, 2 * E5020, 2 * E5020,
        E7520_14, 2 * E5020, 2 * E5020,
        E5020, E5020, E5020, E5020, E5020, E4010, E4010,
        E5020, E5020, E5020, E5020, E5020, E4010, E4010,
    ],
    dtype=np.float64,
)

KP = _KP
KD = _KD
EFFORT_LIMIT = _EFFORT
ACTION_SCALE = 0.25 * _EFFORT / _KP

DEFAULT_POS = np.zeros(NUM_JOINTS, dtype=np.float64)
DEFAULT_POS[[0, 6]] = -0.312
DEFAULT_POS[[3, 9]] = 0.669
DEFAULT_POS[[4, 10]] = -0.363
DEFAULT_POS[[18, 25]] = 0.6
DEFAULT_POS[15] = 0.2
DEFAULT_POS[16] = 0.2
DEFAULT_POS[22] = 0.2
DEFAULT_POS[23] = -0.2
DEFAULT_HEIGHT = 0.76

G1_JOINT_RANGES = np.array(
    [
        [-2.5307, 2.8798], [-0.5236, 2.9671], [-2.7576, 2.7576], [-0.087267, 2.8798],
        [-0.87267, 0.5236], [-0.2618, 0.2618],
        [-2.5307, 2.8798], [-2.9671, 0.5236], [-2.7576, 2.7576], [-0.087267, 2.8798],
        [-0.87267, 0.5236], [-0.2618, 0.2618],
        [-2.618, 2.618], [-0.52, 0.52], [-0.52, 0.52],
        [-3.0892, 2.6704], [-1.5882, 2.2515], [-2.618, 2.618], [-1.0472, 2.0944],
        [-1.97222, 1.97222], [-1.61443, 1.61443], [-1.61443, 1.61443],
        [-3.0892, 2.6704], [-2.2515, 1.5882], [-2.618, 2.618], [-1.0472, 2.0944],
        [-1.97222, 1.97222], [-1.61443, 1.61443], [-1.61443, 1.61443],
    ],
    dtype=np.float64,
)

FOOT_GEOM_PATTERN = r"^(left|right)_foot[1-7]_collision$"
