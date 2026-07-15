"""Global configuration constants for the video analysis module.

Toyota Smarthome refined skeleton V1.2 uses 13 joints.
Joint indices follow the dataset convention.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Toyota Smarthome Refined Skeleton v1.2 — 13-joint layout
# ---------------------------------------------------------------------------
# These 13 joints cover upper body + lower body in a compact representation.
# 0:  Pelvis        (hip centre)
# 1:  Spine         (mid-torso)
# 2:  Neck
# 3:  Head
# 4:  Left Shoulder
# 5:  Left Elbow
# 6:  Left Wrist
# 7:  Right Shoulder
# 8:  Right Elbow
# 9:  Right Wrist
# 10: Left Hip
# 11: Left Knee
# 12: Left Ankle
# ---------------------------------------------------------------------------

JOINT_NAMES: Final[tuple[str, ...]] = (
    "pelvis",
    "spine",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
)

NUM_JOINTS: Final[int] = len(JOINT_NAMES)  # 13

# Commonly used joint subsets
UPPER_BODY_JOINTS: Final[tuple[int, ...]] = (2, 3, 4, 5, 6, 7, 8, 9)  # neck → wrists
LOWER_BODY_JOINTS: Final[tuple[int, ...]] = (0, 10, 11, 12)  # pelvis + left leg
CORE_JOINTS: Final[tuple[int, ...]] = (0, 1, 2)  # pelvis, spine, neck
HEAD_JOINT: Final[int] = 3
PELVIS_JOINT: Final[int] = 0

# ---------------------------------------------------------------------------
# Sliding-window defaults
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_SIZE: Final[int] = 30  # frames (~1 s at 30 fps)
DEFAULT_WINDOW_STRIDE: Final[int] = 15  # frames (50 % overlap)

# ---------------------------------------------------------------------------
# Night-time window (22:00 – 06:00)
# ---------------------------------------------------------------------------
NIGHT_START_HOUR: Final[int] = 22
NIGHT_END_HOUR: Final[int] = 6

# ---------------------------------------------------------------------------
# Dataset paths (relative to project root)
# ---------------------------------------------------------------------------
DATASET_ROOT: Final[str] = "dataset"
SKELETON_ZIP: Final[str] = "dataset/toyota_smarthome_skeleton_v1.2.zip"
ANNOTATION_TAR: Final[str] = "dataset/Annotation_v1.0.tar.gz"
RGB_TAR: Final[str] = "dataset/toyota_smarthome_mp4.tar.gz"
