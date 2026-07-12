"""Brute-force frame selection (handover §13) — the candidate edge.

Per frame: many candidates, score each, lock the winner, advance. The
continuity term is the key bit: each selection is conditioned on the previous
*locked* frame, explicitly optimising for smooth motion instead of picking
frames in isolation — which is exactly what kills jitter.

Scoring happens in PIXELIZED space (PLAN.md decision §6.6): the pixelized frame
is the shipped artefact, so wobble that vanishes in the downscale shouldn't
cost a candidate its slot. Pose fidelity is enforced upstream by the ControlNet
conditioning; a per-candidate pose scorer can additionally be plugged in via
`pose_scores` (one list of floats per frame) for the task-15 experiment.

Pure numpy/CPU — fully testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..color import srgb_to_oklab

# Alpha mismatch hurts about as much as a large colour shift: a pixel popping
# in/out of the silhouette is the most visible kind of jitter at sprite scale.
ALPHA_WEIGHT = 0.5


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Perceptual distance between two pixelized RGBA frames (same shape).

    Mean OKLab distance over pixels opaque in both, plus a penalty for pixels
    whose opacity differs (silhouette pop). Lower = smoother transition.
    """
    if a.shape != b.shape:
        raise ValueError(f"frame shapes differ: {a.shape} vs {b.shape}")
    a_op = a[..., 3] > 0
    b_op = b[..., 3] > 0
    both = a_op & b_op
    if both.any():
        lab_a = srgb_to_oklab(a[..., :3][both].astype(np.float64))
        lab_b = srgb_to_oklab(b[..., :3][both].astype(np.float64))
        color_term = float(np.linalg.norm(lab_a - lab_b, axis=1).mean())
    else:
        color_term = 0.0
    alpha_term = float((a_op != b_op).mean())
    return color_term + ALPHA_WEIGHT * alpha_term


def medoid_index(frames: list[np.ndarray]) -> int:
    """The most representative candidate: minimal total distance to the others.

    Used to pick frame 0, which has no previous locked frame to flow from.
    """
    n = len(frames)
    if n == 1:
        return 0
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = frame_distance(frames[i], frames[j])
            dists[i, j] = dists[j, i] = d
    return int(dists.sum(1).argmin())


@dataclass
class Selection:
    indices: list[int]          # chosen candidate index per frame
    frames: list[np.ndarray]    # the locked pixelized RGBA frames
    costs: list[float]          # transition cost paid at each frame (0.0 for frame 0)


def select_frames(
    candidates_per_frame: list[list[np.ndarray]],
    pose_scores: list[list[float]] | None = None,
    w_pose: float = 1.0,
    w_continuity: float = 1.0,
) -> Selection:
    """Lock-and-advance selection over pixelized RGBA candidate arrays.

    candidates_per_frame[k] are the candidates for animation frame k.
    pose_scores[k][i] (optional, higher = better) is candidate i's pose
    fidelity for frame k, blended as  w_pose * score - w_continuity * distance.
    """
    if not candidates_per_frame or any(len(c) == 0 for c in candidates_per_frame):
        raise ValueError("every frame needs at least one candidate")
    if pose_scores is not None and [len(p) for p in pose_scores] != \
            [len(c) for c in candidates_per_frame]:
        raise ValueError("pose_scores shape must match candidates_per_frame")

    indices: list[int] = []
    locked: list[np.ndarray] = []
    costs: list[float] = []

    for k, cands in enumerate(candidates_per_frame):
        if k == 0:
            if pose_scores is not None:
                best = int(np.argmax(pose_scores[0]))
            else:
                best = medoid_index(cands)
            cost = 0.0
        else:
            prev = locked[-1]
            dists = [frame_distance(prev, c) for c in cands]
            if pose_scores is not None:
                blended = [w_pose * pose_scores[k][i] - w_continuity * d
                           for i, d in enumerate(dists)]
                best = int(np.argmax(blended))
            else:
                best = int(np.argmin(dists))
            cost = dists[best]
        indices.append(best)
        locked.append(cands[best])
        costs.append(cost)

    return Selection(indices=indices, frames=locked, costs=costs)
