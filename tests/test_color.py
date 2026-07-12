from __future__ import annotations

import numpy as np

from spriteforge.color import oklab_to_srgb, srgb_to_oklab


def test_round_trip_is_identity():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 255, size=(1000, 3))
    back = oklab_to_srgb(srgb_to_oklab(rgb))
    # ~1e-4 numerical error on the 0-255 scale is inherent to the cbrt/pow
    # round-trip; far below the uint8 quantisation step, so invisible in output.
    assert np.allclose(back, rgb, atol=1e-3)


def test_known_anchors():
    # OKLab: black -> L=0, white -> L=1, both achromatic (a=b=0)
    black = srgb_to_oklab(np.array([0.0, 0.0, 0.0]))
    white = srgb_to_oklab(np.array([255.0, 255.0, 255.0]))
    assert np.allclose(black, [0.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(white, [1.0, 0.0, 0.0], atol=1e-4)


def test_greys_are_achromatic():
    greys = np.stack([np.full(3, v) for v in (32.0, 128.0, 220.0)])
    lab = srgb_to_oklab(greys)
    assert np.allclose(lab[:, 1:], 0.0, atol=1e-6)
    # lightness strictly increasing with value
    assert np.all(np.diff(lab[:, 0]) > 0)
