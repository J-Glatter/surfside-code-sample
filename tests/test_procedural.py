from __future__ import annotations

import numpy as np
from PIL import Image

from spriteforge.animate.procedural import PROCEDURAL_ACTIONS, bounce_cycle, idle_cycle


def _sprite(size=(32, 24)):
    return Image.new("RGBA", size, (40, 160, 70, 255))


def test_bounce_frame_count_and_canvas():
    frames = bounce_cycle(_sprite(), frames=12)
    assert len(frames) == 12
    sizes = {f.size for f in frames}
    assert len(sizes) == 1                       # constant canvas — engine friendly
    w, h = sizes.pop()
    assert w >= 32 and h >= 24 * 2               # room for the jump (default = height)


def test_bounce_actually_leaves_the_ground():
    frames = bounce_cycle(_sprite(), frames=12, jump=40)

    def lowest_opaque_row(img):
        alpha = np.asarray(img)[..., 3]
        return int(np.nonzero(alpha.any(axis=1))[0].max())

    ground = lowest_opaque_row(frames[0])
    apex = min(lowest_opaque_row(f) for f in frames)
    assert ground - apex >= 30                   # visibly airborne at the apex


def test_bounce_deterministic():
    a = bounce_cycle(_sprite(), frames=8)
    b = bounce_cycle(_sprite(), frames=8)
    for fa, fb in zip(a, b, strict=True):
        assert np.array_equal(np.asarray(fa), np.asarray(fb))


def test_idle_breathes_in_place():
    frames = idle_cycle(_sprite(), frames=6, amount=0.1)
    assert len(frames) == 6
    heights = []
    for f in frames:
        alpha = np.asarray(f)[..., 3]
        rows = np.nonzero(alpha.any(axis=1))[0]
        heights.append(int(rows.max() - rows.min()) + 1)
        assert rows.max() >= f.height - 2        # bottom-anchored throughout
    assert max(heights) > min(heights)           # it does actually breathe


def test_sway_pivots_at_the_base():
    from spriteforge.animate.procedural import sway_cycle

    frames = sway_cycle(_sprite((20, 60)), frames=8, amount=0.2)
    assert len(frames) == 8
    assert len({f.size for f in frames}) == 1     # constant canvas

    def row_center(img, y):
        alpha = np.asarray(img)[..., 3]
        xs = np.nonzero(alpha[y])[0]
        return float(xs.mean()) if len(xs) else None

    base_y, top_y = 59, 0
    base_centers = [row_center(f, base_y) for f in frames]
    top_centers = [row_center(f, top_y) for f in frames]
    # trunk planted, canopy leaning both ways across the cycle
    assert max(base_centers) - min(base_centers) <= 1.5
    assert max(top_centers) - min(top_centers) >= 0.2 * 60 * 1.2  # ~both peaks


def test_sway_loops_seamlessly():
    from spriteforge.animate.procedural import sway_cycle

    frames = sway_cycle(_sprite((20, 40)), frames=8)
    # frame 0 has zero shear; the cycle returns to it (sin(0) == sin(2pi))
    again = sway_cycle(_sprite((20, 40)), frames=8)
    assert np.array_equal(np.asarray(frames[0]), np.asarray(again[0]))
    alpha0 = np.asarray(frames[0])[..., 3]
    alpha4 = np.asarray(frames[4])[..., 3]        # half cycle also unsheared
    assert np.array_equal(alpha0, alpha4)


def test_sway_bias_leans_into_the_wind():
    from spriteforge.animate.procedural import sway_cycle

    frames = sway_cycle(_sprite((20, 60)), frames=8, amount=0.03, bias=0.05)

    def top_center(img):
        alpha = np.asarray(img)[..., 3]
        xs = np.nonzero(alpha[0])[0]
        return float(xs.mean())

    canvas_mid = (frames[0].width - 1) / 2
    # every frame's canopy sits on the wind side of centre: bias > amount
    assert all(top_center(f) > canvas_mid for f in frames)


def test_action_registry():
    from spriteforge.animate.procedural import PROCEDURAL_FPS

    assert set(PROCEDURAL_ACTIONS) == {"bounce", "idle", "sway"}
    assert set(PROCEDURAL_FPS) == set(PROCEDURAL_ACTIONS)
