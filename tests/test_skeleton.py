from __future__ import annotations

import numpy as np

from spriteforge.animate.skeleton import (
    ACTIONS,
    DEFAULT_FRAMES,
    JOINTS,
    load_poses,
    render_openpose,
    save_poses,
    walk_cycle,
)


def test_actions_produce_default_frame_counts():
    for name, fn in ACTIONS.items():
        poses = fn(DEFAULT_FRAMES[name])
        assert len(poses) == DEFAULT_FRAMES[name]
        for pose in poses:
            assert set(pose.keypoints) == set(JOINTS)


def test_keypoints_stay_in_frame():
    for name, fn in ACTIONS.items():
        for pose in fn(DEFAULT_FRAMES[name]):
            for joint, pt in pose.keypoints.items():
                assert pt is not None
                assert 0.0 <= pt[0] <= 1.0, f"{name}/{joint} x={pt[0]}"
                assert 0.0 <= pt[1] <= 1.0, f"{name}/{joint} y={pt[1]}"


def test_walk_legs_alternate():
    poses = walk_cycle(8)
    # frame 2 (phase pi/2): right leg swung forward, left back
    kp = poses[2].keypoints
    assert kp["r_ankle"][0] > kp["r_hip"][0]
    assert kp["l_ankle"][0] < kp["l_hip"][0]
    # half a cycle later they swap
    kp6 = poses[6].keypoints
    assert kp6["r_ankle"][0] < kp6["r_hip"][0]
    assert kp6["l_ankle"][0] > kp6["l_hip"][0]


def test_walk_is_cyclic_and_deterministic():
    a, b = walk_cycle(8), walk_cycle(8)
    for pa, pb in zip(a, b, strict=True):
        assert pa.keypoints == pb.keypoints
    # anatomy: hip above knee above ankle throughout (y grows downward)
    for pose in a:
        kp = pose.keypoints
        assert kp["r_hip"][1] < kp["r_knee"][1] < kp["r_ankle"][1] + 0.05


def test_render_openpose_draws_something():
    img = render_openpose(walk_cycle(8)[0], size=256)
    assert img.size == (256, 256)
    arr = np.asarray(img)
    assert (arr > 0).any()                       # limbs drawn
    assert (arr == 0).all(axis=-1).mean() > 0.5  # on a mostly-black background


def test_pose_json_round_trip(tmp_path):
    poses = walk_cycle(4)
    path = tmp_path / "walk.json"
    save_poses(poses, path)
    loaded = load_poses(path)
    assert len(loaded) == 4
    for orig, back in zip(poses, loaded, strict=True):
        for j in JOINTS:
            assert np.allclose(orig.keypoints[j], back.keypoints[j])
