from __future__ import annotations

import numpy as np
import pytest

from spriteforge.animate.pipeline import rig_for
from spriteforge.animate.skeleton import load_poses, save_poses
from spriteforge.animate.skeleton_quadruped import (
    ACTIONS,
    DEFAULT_FRAMES,
    QUAD_JOINTS,
    gallop_cycle,
    render_quadruped,
    trot_cycle,
    walk_cycle,
)


def test_actions_produce_default_frame_counts():
    for name, fn in ACTIONS.items():
        poses = fn(DEFAULT_FRAMES[name])
        assert len(poses) == DEFAULT_FRAMES[name]
        for pose in poses:
            assert set(pose.keypoints) == set(QUAD_JOINTS)


def test_keypoints_stay_in_frame():
    for name, fn in ACTIONS.items():
        for pose in fn(DEFAULT_FRAMES[name]):
            for joint, pt in pose.keypoints.items():
                assert pt is not None
                assert 0.0 <= pt[0] <= 1.0, f"{name}/{joint} x={pt[0]}"
                assert 0.0 <= pt[1] <= 1.0, f"{name}/{joint} y={pt[1]}"


def _paw_x(pose, paw):
    return pose.keypoints[paw][0]


def test_walk_is_four_beat():
    """Lateral-sequence walk: each paw peaks forward at a different quarter."""
    poses = walk_cycle(8)
    peaks = {}
    for paw in ("l_front_paw", "r_front_paw", "l_back_paw", "r_back_paw"):
        xs = [_paw_x(p, paw) for p in poses]
        peaks[paw] = int(np.argmax(xs))
    assert len(set(peaks.values())) == 4      # all four peaks at distinct frames


def test_trot_moves_diagonal_pairs_together():
    """Upper-leg swing (shoulder->elbow / hip->knee) carries the pure gait
    phase — paw positions also include the bend, which differs by design
    between front (bends back) and hind (bends forward) legs."""
    poses = trot_cycle(8)
    for pose in poses:
        kp = pose.keypoints
        lf = kp["l_elbow"][0] - kp["l_shoulder"][0]
        rh = kp["r_knee"][0] - kp["r_hip"][0]
        assert abs(lf - rh) < 1e-9            # diagonal pair: identical swing
    # and the two diagonals oppose at mid-swing
    kp2 = poses[2].keypoints
    lf2 = kp2["l_elbow"][0] - kp2["l_shoulder"][0]
    rf2 = kp2["r_elbow"][0] - kp2["r_shoulder"][0]
    assert lf2 * rf2 < 0


def test_gallop_pitches_the_spine():
    poses = gallop_cycle(8)
    pitches = [p.keypoints["neck"][1] - p.keypoints["tail_root"][1] for p in poses]
    assert max(pitches) > 0.01 and min(pitches) < -0.01  # rocks both ways


def test_jump_has_crouch_and_airborne():
    poses = ACTIONS["jump"](6)
    neck_ys = [p.keypoints["neck"][1] for p in poses]
    assert neck_ys[0] > min(neck_ys) + 0.05   # starts crouched vs airborne peak
    assert np.argmin(neck_ys) in (2, 3)       # highest mid-sequence


def test_render_and_json_round_trip(tmp_path):
    poses = walk_cycle(4)
    img = render_quadruped(poses[0], size=256)
    assert img.size == (256, 256)
    assert (np.asarray(img) > 0).any()

    save_poses(poses, tmp_path / "walk.json")
    loaded = load_poses(tmp_path / "walk.json")
    assert len(loaded) == 4
    for orig, back in zip(poses, loaded, strict=True):
        for j in QUAD_JOINTS:
            assert np.allclose(orig.keypoints[j], back.keypoints[j])


def test_rig_for_selects_bodies():
    actions, frames, render = rig_for("quadruped")
    assert set(actions) == {"walk", "trot", "gallop", "jump"}
    h_actions, _, _ = rig_for("humanoid")
    assert set(h_actions) == {"walk", "run", "jump"}
    with pytest.raises(ValueError):
        rig_for("centipede")
