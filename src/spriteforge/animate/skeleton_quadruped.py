"""Quadruped skeletons — AP-10K keypoints -> conditioning images.

Same philosophy as the humanoid rig (skeleton.py): hand-authored parametric
gait cycles, deterministic and reusable across every four-legged character;
tune the tunables visually on real outputs. Side view, animal facing right,
normalised [0,1] coordinates with y down.

The 17-joint set follows the AP-10K convention that community animal-openpose
ControlNets are trained on. Gait timing is textbook (Muybridge): the walk is a
four-beat lateral sequence with each leg a quarter-cycle apart, the trot moves
diagonal pairs together, the gallop is asymmetric with front/hind pairs half a
cycle apart and a pitching, bobbing body. The jump is keyframed: crouch ->
launch -> stretch -> gather -> land -> recover.
"""

from __future__ import annotations

import math

from PIL import Image

from .skeleton import Pose, render_openpose

QUAD_JOINTS = [
    "l_eye", "r_eye", "nose", "neck", "tail_root",
    "l_shoulder", "l_elbow", "l_front_paw",
    "r_shoulder", "r_elbow", "r_front_paw",
    "l_hip", "l_knee", "l_back_paw",
    "r_hip", "r_knee", "r_back_paw",
]

_QUAD_LIMBS = [
    (0, 2), (1, 2), (2, 3),                    # eyes -> nose -> neck
    (3, 4),                                    # spine: neck -> tail root
    (3, 5), (5, 6), (6, 7),                    # left front leg
    (3, 8), (8, 9), (9, 10),                   # right front leg
    (4, 11), (11, 12), (12, 13),               # left hind leg
    (4, 14), (14, 15), (15, 16),               # right hind leg
]
_QUAD_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 170), (255, 0, 85),
]

# ---- body proportions & motion tunables (normalised units / radians) --------
BODY_Y = 0.50            # spine height
SPINE_HALF = 0.16        # neck / tail-root distance from body centre
UPPER_LEG, LOWER_LEG = 0.10, 0.11
CENTER_X = 0.5
DEPTH_OFFSET = 0.015     # left-side legs nudged back so they stay legible

WALK_SWING = math.radians(20)
WALK_BEND = math.radians(25)
WALK_BOB = 0.006

TROT_SWING = math.radians(28)
TROT_BEND = math.radians(35)
TROT_BOB = 0.012

GALLOP_SWING = math.radians(40)
GALLOP_BEND = math.radians(50)
GALLOP_BOB = 0.025
GALLOP_PITCH = math.radians(7)

# per-leg phase offsets (fraction of cycle), keyed LF / RF / LH / RH
_WALK_PHASES = {"lf": 0.25, "rf": 0.75, "lh": 0.0, "rh": 0.5}   # 4-beat lateral
_TROT_PHASES = {"lf": 0.0, "rf": 0.5, "lh": 0.5, "rh": 0.0}     # diagonal pairs
_GALLOP_PHASES = {"lf": 0.65, "rf": 0.5, "lh": 0.15, "rh": 0.0}  # transverse


def _polar(origin: tuple[float, float], angle: float, length: float) -> tuple[float, float]:
    """angle 0 = straight down; positive swings toward +x (facing direction)."""
    return origin[0] + length * math.sin(angle), origin[1] + length * math.cos(angle)


def _figure(
    bob: float = 0.0,
    pitch: float = 0.0,          # positive = front end up
    lf: tuple[float, float] = (0.0, 0.0),   # (swing, bend) per leg
    rf: tuple[float, float] = (0.0, 0.0),
    lh: tuple[float, float] = (0.0, 0.0),
    rh: tuple[float, float] = (0.0, 0.0),
) -> Pose:
    """Assemble the 17-joint quadruped from body attitude and leg angles.

    Front legs bend their elbow backward, hind legs bend their knee forward —
    the two-segment approximation of real quadruped anatomy that reads
    correctly in silhouette.
    """
    cx, cy = CENTER_X, BODY_Y + bob
    dx, dy = SPINE_HALF * math.cos(pitch), SPINE_HALF * math.sin(pitch)
    neck = (cx + dx, cy - dy)
    tail = (cx - dx, cy + dy)
    nose = (neck[0] + 0.095, neck[1] - 0.055)

    kp: dict[str, tuple[float, float] | None] = {
        "neck": neck, "tail_root": tail, "nose": nose,
        "r_eye": (nose[0] - 0.03, nose[1] - 0.03),
        "l_eye": (nose[0] - 0.045, nose[1] - 0.035),
    }

    front_anchor = (neck[0] - 0.02, neck[1] + 0.03)
    hind_anchor = (tail[0] + 0.02, tail[1] + 0.02)
    for name, anchor, (swing, bend), bend_sign, depth in (
        ("l_front", front_anchor, lf, -1, -DEPTH_OFFSET),
        ("r_front", front_anchor, rf, -1, 0.0),
        ("l_back", hind_anchor, lh, +1, -DEPTH_OFFSET),
        ("r_back", hind_anchor, rh, +1, 0.0),
    ):
        top = (anchor[0] + depth, anchor[1])
        mid = _polar(top, swing, UPPER_LEG)
        paw = _polar(mid, swing + bend_sign * bend, LOWER_LEG)
        side = name[0]                       # "l" / "r"
        if "front" in name:
            kp[f"{side}_shoulder"], kp[f"{side}_elbow"], kp[f"{side}_front_paw"] = \
                top, mid, paw
        else:
            kp[f"{side}_hip"], kp[f"{side}_knee"], kp[f"{side}_back_paw"] = \
                top, mid, paw
    return Pose(kp)


def _gait(frames: int, phases: dict[str, float], swing: float, bend: float,
          bob_amount: float, pitch_amount: float = 0.0) -> list[Pose]:
    poses = []
    for t in range(frames):
        u = t / frames

        def leg(key: str, u: float = u) -> tuple[float, float]:
            p = 2 * math.pi * (u + phases[key])
            return swing * math.sin(p), bend * max(0.0, math.cos(p))

        poses.append(_figure(
            bob=bob_amount * math.sin(4 * math.pi * u),
            pitch=pitch_amount * math.sin(2 * math.pi * u),
            lf=leg("lf"), rf=leg("rf"), lh=leg("lh"), rh=leg("rh"),
        ))
    return poses


def walk_cycle(frames: int = 8) -> list[Pose]:
    return _gait(frames, _WALK_PHASES, WALK_SWING, WALK_BEND, WALK_BOB)


def trot_cycle(frames: int = 8) -> list[Pose]:
    return _gait(frames, _TROT_PHASES, TROT_SWING, TROT_BEND, TROT_BOB)


def gallop_cycle(frames: int = 8) -> list[Pose]:
    return _gait(frames, _GALLOP_PHASES, GALLOP_SWING, GALLOP_BEND,
                 GALLOP_BOB, GALLOP_PITCH)


def jump_cycle(frames: int = 6) -> list[Pose]:
    """Crouch -> launch -> stretch -> gather -> land (front first) -> recover."""
    d = math.radians
    keyframes = [
        # bob, pitch, front (swing, bend), hind (swing, bend)
        (0.045, d(-4), (d(5), d(30)), (d(15), d(55))),     # coiled crouch
        (-0.020, d(12), (d(-25), d(45)), (d(-30), d(10))),  # launch: hind extended
        (-0.085, d(4), (d(35), d(15)), (d(-35), d(15))),    # airborne stretch
        (-0.085, d(-6), (d(-15), d(50)), (d(30), d(50))),   # airborne gather
        (-0.015, d(-12), (d(20), d(10)), (d(25), d(45))),   # landing, front first
        (0.035, d(-2), (d(5), d(30)), (d(10), d(45))),      # recover crouch
    ]
    if frames != len(keyframes):
        raise ValueError(f"jump cycle is authored at {len(keyframes)} frames")
    poses = []
    for bob, pitch, front, hind in keyframes:
        poses.append(_figure(bob=bob, pitch=pitch,
                             lf=front, rf=front, lh=hind, rh=hind))
    return poses


ACTIONS = {
    "walk": walk_cycle,
    "trot": trot_cycle,
    "gallop": gallop_cycle,
    "jump": jump_cycle,
}
DEFAULT_FRAMES = {"walk": 8, "trot": 8, "gallop": 8, "jump": 6}


def render_quadruped(pose: Pose, size: int = 512, line_width: int = 8,
                     joint_radius: int = 5) -> Image.Image:
    """Draw the pose in the AP-10K topology (for animal-openpose ControlNets)."""
    return render_openpose(pose, size=size, line_width=line_width,
                           joint_radius=joint_radius, joints=QUAD_JOINTS,
                           limbs=_QUAD_LIMBS, colors=_QUAD_COLORS)
