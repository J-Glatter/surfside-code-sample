"""Canonical action skeletons -> OpenPose-style conditioning images.

Hand-authored parametric cycles (v1, per PLAN.md: deterministic and reusable
across every character; tune the tunables below visually on real outputs, or
swap in extraction-from-reference later). Side view, character facing right,
normalised [0,1] coordinates with y pointing down.

Joint set and limb colours follow the OpenPose COCO-18 convention that the
`lllyasviel/sd-controlnet-openpose` ControlNet was trained on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

JOINTS = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

# OpenPose limb sequence (joint-index pairs) and the standard colour ramp.
_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]
_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 170), (255, 0, 85), (255, 0, 0),
]

# ---- figure proportions & motion tunables (normalised units / radians) --------
HIP_Y = 0.58
NECK_Y = 0.34
THIGH, SHIN = 0.13, 0.13
UPPER_ARM, FOREARM = 0.11, 0.10
CENTER_X = 0.5
DEPTH_OFFSET = 0.015          # left-side limbs nudged back so they stay legible

WALK_SWING = math.radians(30)
WALK_BEND = math.radians(25)
WALK_ARM = math.radians(20)
WALK_BOB = 0.012

RUN_SWING = math.radians(45)
RUN_BEND = math.radians(60)
RUN_ARM = math.radians(35)
RUN_ELBOW = math.radians(90)
RUN_BOB = 0.025
RUN_LEAN = 0.03               # forward lean of the torso when running


@dataclass
class Pose:
    """Named keypoints in normalised [0,1] image coordinates (None = absent)."""

    keypoints: dict[str, tuple[float, float] | None]

    def as_list(self) -> list[tuple[float, float] | None]:
        return [self.keypoints.get(j) for j in JOINTS]


def _polar(origin: tuple[float, float], angle: float, length: float) -> tuple[float, float]:
    """angle 0 = straight down; positive swings toward +x (facing direction)."""
    return origin[0] + length * math.sin(angle), origin[1] + length * math.cos(angle)


def _figure(
    bob: float = 0.0,
    lean: float = 0.0,
    r_thigh: float = 0.0, r_bend: float = 0.0,
    l_thigh: float = 0.0, l_bend: float = 0.0,
    r_arm: float = 0.0, r_elbow: float = 0.0,
    l_arm: float = 0.0, l_elbow: float = 0.0,
) -> Pose:
    """Assemble the 18-joint figure from limb angles (0 = hanging straight down).

    `*_bend` folds the shin/forearm back relative to its thigh/upper arm.
    """
    hip = (CENTER_X, HIP_Y + bob)
    neck = (CENTER_X + lean, NECK_Y + bob)
    nose = (neck[0] + 0.045, neck[1] - 0.075)

    kp: dict[str, tuple[float, float] | None] = {"neck": neck, "nose": nose}
    kp["r_eye"] = (nose[0] + 0.012, nose[1] - 0.012)
    kp["l_eye"] = (nose[0] - 0.006, nose[1] - 0.014)
    kp["r_ear"] = (neck[0] + 0.01, nose[1] + 0.005)
    kp["l_ear"] = (neck[0] - 0.012, nose[1] + 0.005)

    for side, thigh_a, bend_a, arm_a, elbow_a, depth in (
        ("r", r_thigh, r_bend, r_arm, r_elbow, 0.0),
        ("l", l_thigh, l_bend, l_arm, l_elbow, -DEPTH_OFFSET),
    ):
        s_hip = (hip[0] + depth, hip[1])
        knee = _polar(s_hip, thigh_a, THIGH)
        ankle = _polar(knee, thigh_a - bend_a, SHIN)
        kp[f"{side}_hip"], kp[f"{side}_knee"], kp[f"{side}_ankle"] = s_hip, knee, ankle

        shoulder = (neck[0] + depth, neck[1] + 0.01)
        elbow = _polar(shoulder, arm_a, UPPER_ARM)
        wrist = _polar(elbow, arm_a + elbow_a, FOREARM)
        kp[f"{side}_shoulder"], kp[f"{side}_elbow"], kp[f"{side}_wrist"] = \
            shoulder, elbow, wrist

    return Pose(kp)


# ---- action cycles -------------------------------------------------------------

def walk_cycle(frames: int = 8) -> list[Pose]:
    poses = []
    for t in range(frames):
        p = 2 * math.pi * t / frames
        poses.append(_figure(
            bob=WALK_BOB * math.cos(2 * p),
            r_thigh=WALK_SWING * math.sin(p),
            r_bend=WALK_BEND * max(0.0, math.cos(p)),
            l_thigh=WALK_SWING * math.sin(p + math.pi),
            l_bend=WALK_BEND * max(0.0, -math.cos(p)),
            # arms counter-swing their same-side leg, small constant elbow bend
            r_arm=WALK_ARM * math.sin(p + math.pi),
            r_elbow=math.radians(15),
            l_arm=WALK_ARM * math.sin(p),
            l_elbow=math.radians(15),
        ))
    return poses


def run_cycle(frames: int = 8) -> list[Pose]:
    poses = []
    for t in range(frames):
        p = 2 * math.pi * t / frames
        poses.append(_figure(
            bob=RUN_BOB * math.cos(2 * p),
            lean=RUN_LEAN,
            r_thigh=RUN_SWING * math.sin(p),
            r_bend=RUN_BEND * max(0.0, math.cos(p)) + math.radians(15),
            l_thigh=RUN_SWING * math.sin(p + math.pi),
            l_bend=RUN_BEND * max(0.0, -math.cos(p)) + math.radians(15),
            r_arm=RUN_ARM * math.sin(p + math.pi),
            r_elbow=RUN_ELBOW,
            l_arm=RUN_ARM * math.sin(p),
            l_elbow=RUN_ELBOW,
        ))
    return poses


def jump_cycle(frames: int = 6) -> list[Pose]:
    """Anticipate -> crouch -> take off -> tuck -> extend -> land."""
    d = math.radians
    keyframes = [
        # bob, lean, thigh, bend, arm, elbow
        (0.010, 0.00, d(10), d(20), d(-10), d(15)),   # slight anticipation
        (0.060, 0.02, d(40), d(80), d(-30), d(20)),   # deep crouch, arms back
        (-0.040, 0.01, d(-10), d(5), d(150), d(10)),  # take-off, extended, arms up
        (-0.080, 0.00, d(55), d(95), d(140), d(30)),  # airborne tuck
        (-0.030, 0.00, d(20), d(30), d(90), d(20)),   # extending for landing
        (0.050, 0.02, d(35), d(70), d(30), d(15)),    # landing crouch
    ]
    if frames != len(keyframes):
        raise ValueError(f"jump cycle is authored at {len(keyframes)} frames")
    poses = []
    for bob, lean, thigh, bend, arm, elbow in keyframes:
        poses.append(_figure(
            bob=bob, lean=lean,
            r_thigh=thigh, r_bend=bend, l_thigh=thigh, l_bend=bend,
            r_arm=arm, r_elbow=elbow, l_arm=arm, l_elbow=elbow,
        ))
    return poses


ACTIONS = {
    "walk": walk_cycle,
    "run": run_cycle,
    "jump": jump_cycle,
}
DEFAULT_FRAMES = {"walk": 8, "run": 8, "jump": 6}


# ---- rendering & I/O -------------------------------------------------------------

def render_openpose(pose: Pose, size: int = 512, line_width: int = 8,
                    joint_radius: int = 5) -> Image.Image:
    """Draw the pose as an OpenPose-style conditioning image (black background)."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    pts = [None if p is None else (p[0] * size, p[1] * size) for p in pose.as_list()]
    for (a, b), color in zip(_LIMBS, _COLORS, strict=False):
        if pts[a] is not None and pts[b] is not None:
            draw.line([pts[a], pts[b]], fill=color, width=line_width)
    for idx, pt in enumerate(pts):
        if pt is not None:
            color = _COLORS[idx % len(_COLORS)]
            draw.ellipse([pt[0] - joint_radius, pt[1] - joint_radius,
                          pt[0] + joint_radius, pt[1] + joint_radius], fill=color)
    return img


def save_poses(poses: list[Pose], path: str | Path) -> None:
    data = [{j: list(p.keypoints[j]) if p.keypoints.get(j) else None for j in JOINTS}
            for p in poses]
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_poses(path: str | Path) -> list[Pose]:
    data = json.loads(Path(path).read_text())
    return [Pose({j: tuple(v) if v else None for j, v in frame.items()})
            for frame in data]
