from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def gradient_scene() -> Image.Image:
    """A 512px synthetic scene: colour gradients + a transparent border region.

    Rich enough in colour to exercise the quantiser, with soft alpha edges to
    exercise the crisp-alpha step. Deterministic (no randomness).
    """
    size = 512
    y, x = np.mgrid[0:size, 0:size]
    r = (x / (size - 1) * 255).astype(np.uint8)
    g = (y / (size - 1) * 255).astype(np.uint8)
    b = ((x + y) / (2 * (size - 1)) * 255).astype(np.uint8)
    # radial soft alpha: opaque centre, soft fade, transparent corners
    cx = cy = (size - 1) / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = np.clip((size * 0.45 - dist) / (size * 0.1) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([r, g, b, alpha]), "RGBA")


@pytest.fixture
def noise_image() -> Image.Image:
    """A small fully-opaque image with many distinct colours (seeded noise)."""
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    alpha = np.full((96, 96, 1), 255, dtype=np.uint8)
    return Image.fromarray(np.concatenate([rgb, alpha], axis=2), "RGBA")


def opaque_colors(img: Image.Image) -> set[tuple[int, int, int]]:
    """Distinct RGB values among fully opaque pixels."""
    arr = np.asarray(img.convert("RGBA"))
    opaque = arr[..., 3] == 255
    return {tuple(px) for px in arr[opaque][:, :3]}
