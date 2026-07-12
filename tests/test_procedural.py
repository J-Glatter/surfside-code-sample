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


def test_action_registry():
    assert set(PROCEDURAL_ACTIONS) == {"bounce", "idle"}
