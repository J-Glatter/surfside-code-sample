from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spriteforge.pixelize import pixelize, upscale_preview

from .conftest import opaque_colors

REFERENCE_SCRIPT = Path(__file__).parents[1] / "reference" / "pixelize.py"


def test_color_count_is_bounded(noise_image):
    out = pixelize(noise_image, size=64, colors=16)
    assert len(opaque_colors(out)) <= 16


def test_exact_color_count_when_input_has_few(gradient_scene):
    # An image with exactly 3 distinct colours quantised to 16 keeps <= 3
    arr = np.zeros((60, 60, 4), dtype=np.uint8)
    arr[..., 3] = 255
    arr[:20, :, :3] = (255, 0, 0)
    arr[20:40, :, :3] = (0, 255, 0)
    arr[40:, :, :3] = (0, 0, 255)
    img = Image.fromarray(arr, "RGBA")
    out = pixelize(img, size=60, colors=16)
    assert len(opaque_colors(out)) <= 3


def test_alpha_is_crisp(gradient_scene):
    out = pixelize(gradient_scene, size=64, colors=16)
    alpha = np.asarray(out)[..., 3]
    assert set(np.unique(alpha).tolist()) <= {0, 255}
    # the soft radial fade must produce both fully-opaque and fully-transparent pixels
    assert (alpha == 0).any() and (alpha == 255).any()


def test_deterministic(gradient_scene):
    a = pixelize(gradient_scene, size=64, colors=16, seed=3)
    b = pixelize(gradient_scene, size=64, colors=16, seed=3)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_longest_side_and_aspect():
    img = Image.new("RGBA", (300, 100), (200, 30, 30, 255))
    out = pixelize(img, size=64, colors=4)
    assert max(out.size) == 64
    assert out.size == (64, 21)  # round(100 * 64/300) = 21


def test_all_transparent_input():
    img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    out = pixelize(img, size=32, colors=16)
    assert np.all(np.asarray(out)[..., 3] == 0)


def test_tiny_input():
    img = Image.new("RGBA", (2, 2), (10, 200, 30, 255))
    out = pixelize(img, size=64, colors=16)
    assert max(out.size) == 64


def test_upscale_preview():
    img = Image.new("RGBA", (8, 6), (1, 2, 3, 255))
    up = upscale_preview(img, 4)
    assert up.size == (32, 24)
    # nearest-neighbour: no new colours introduced
    assert opaque_colors(up) == {(1, 2, 3)}


@pytest.mark.skipif(not REFERENCE_SCRIPT.exists(), reason="reference script not present")
def test_regression_matches_reference_script(gradient_scene, noise_image):
    """The port must be bit-identical to the proven Phase-0 script."""
    spec = importlib.util.spec_from_file_location("reference_pixelize", REFERENCE_SCRIPT)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    for img in (gradient_scene, noise_image):
        ours = pixelize(img, size=64, colors=16, seed=0)
        theirs = ref.pixelize(img, size=64, colors=16, seed=0)
        assert np.array_equal(np.asarray(ours), np.asarray(theirs))
