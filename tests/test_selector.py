from __future__ import annotations

import numpy as np
import pytest

from spriteforge.animate.selector import (
    frame_distance,
    medoid_index,
    select_frames,
)


def solid(r, g, b, a=255, size=8):
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3] = r, g, b, a
    return arr


def test_frame_distance_zero_for_identical():
    f = solid(100, 150, 200)
    assert frame_distance(f, f) == 0.0


def test_frame_distance_orders_by_similarity():
    base = solid(100, 100, 100)
    near = solid(110, 110, 110)
    far = solid(255, 0, 0)
    assert frame_distance(base, near) < frame_distance(base, far)


def test_frame_distance_penalises_silhouette_pop():
    base = solid(100, 100, 100)
    ghost = solid(100, 100, 100)
    ghost[:4] = 0  # top half turns transparent — silhouette pop
    assert frame_distance(base, ghost) > 0


def test_frame_distance_shape_mismatch():
    with pytest.raises(ValueError):
        frame_distance(solid(0, 0, 0, size=8), solid(0, 0, 0, size=16))


def test_medoid_picks_the_central_candidate():
    frames = [solid(0, 0, 0), solid(120, 120, 120), solid(255, 255, 255)]
    assert medoid_index(frames) == 1


def test_selection_prefers_continuity():
    """Each frame's candidates include one smooth continuation and distractors —
    the selector must walk the smooth path."""
    grey = [solid(v, v, v) for v in (100, 110, 120, 130)]
    candidates = [
        [solid(255, 0, 0), grey[0], solid(0, 0, 255)],   # frame 0: medoid = grey
        [solid(255, 0, 0), solid(0, 255, 0), grey[1]],   # smooth option at index 2
        [grey[2], solid(255, 0, 255), solid(0, 0, 0)],   # smooth option at index 0
        [solid(0, 0, 0), grey[3], solid(255, 255, 0)],   # smooth option at index 1
    ]
    sel = select_frames(candidates)
    assert sel.indices == [1, 2, 0, 1]
    assert sel.costs[0] == 0.0
    assert all(c < 0.1 for c in sel.costs[1:])  # the walked path is smooth


def test_pose_scores_break_ties_and_lead_frame0():
    a, b = solid(100, 100, 100), solid(100, 100, 100)
    candidates = [[a, b], [a, b]]
    pose_scores = [[0.1, 0.9], [0.8, 0.2]]
    sel = select_frames(candidates, pose_scores=pose_scores)
    assert sel.indices == [1, 0]  # frame 0 by pose; frame 1 pose wins on equal continuity


def test_pose_vs_continuity_weighting():
    base = solid(100, 100, 100)
    smooth = solid(105, 105, 105)   # continuity favourite
    jumpy = solid(250, 250, 250)    # pose favourite
    candidates = [[base], [smooth, jumpy]]
    pose_scores = [[1.0], [0.0, 1.0]]
    # pose term dwarfs continuity -> jumpy wins
    sel = select_frames(candidates, pose_scores=pose_scores, w_pose=100.0)
    assert sel.indices[1] == 1
    # continuity term dwarfs pose -> smooth wins
    sel = select_frames(candidates, pose_scores=pose_scores, w_continuity=100.0,
                        w_pose=0.01)
    assert sel.indices[1] == 0


def test_empty_candidates_rejected():
    with pytest.raises(ValueError):
        select_frames([])
    with pytest.raises(ValueError):
        select_frames([[solid(0, 0, 0)], []])


def test_mismatched_pose_scores_rejected():
    with pytest.raises(ValueError):
        select_frames([[solid(0, 0, 0)]], pose_scores=[[0.5, 0.5]])
