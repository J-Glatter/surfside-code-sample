from __future__ import annotations

import numpy as np
from PIL import Image

from spriteforge.tiling import seam_error, tile_preview


def _tileable(size=64):
    """A genuinely wrap-around pattern: sinusoids with whole-number periods."""
    y, x = np.mgrid[0:size, 0:size]
    r = 128 + 100 * np.sin(2 * np.pi * x / size * 3)
    g = 128 + 100 * np.sin(2 * np.pi * y / size * 2)
    b = 128 + 60 * np.sin(2 * np.pi * (x + y) / size * 4)
    return Image.fromarray(np.dstack([r, g, b]).astype(np.uint8), "RGB")


def _seamy(size=64):
    """A strong left->right gradient: wraps terribly."""
    y, x = np.mgrid[0:size, 0:size]
    v = (x / (size - 1) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([v, v, v]), "RGB")


def test_seam_error_separates_tileable_from_seamy():
    assert seam_error(_tileable()) < 2.0
    assert seam_error(_seamy()) > 10.0


def test_tile_preview_grid():
    img = _tileable(16)
    grid = tile_preview(img, repeat=3)
    assert grid.size == (48, 48)
    # the tile repeats exactly
    assert grid.getpixel((5, 5)) == grid.getpixel((21, 21)) == grid.getpixel((37, 37))


def test_enable_disable_tiling_patch_conv_layers():
    torch = __import__("pytest").importorskip("torch")
    from spriteforge.tiling import disable_tiling, enable_tiling

    class FakePipe:
        pass

    pipe = FakePipe()
    pipe.unet = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU())
    pipe.vae = torch.nn.Sequential(torch.nn.Conv2d(8, 3, 3))

    assert enable_tiling(pipe) == 2
    assert pipe.unet[0].padding_mode == "circular"
    assert pipe.vae[0].padding_mode == "circular"
    assert disable_tiling(pipe) == 2
    assert pipe.unet[0].padding_mode == "zeros"
