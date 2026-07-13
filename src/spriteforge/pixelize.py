"""Stage 2 — turn a normal image into clean pixel art.

The generator (Stable Diffusion) gives you "pixel-style" mush. This module does
the part that actually makes it pixel art:

  1. Downscale to a small grid with a good area filter (not nearest — that aliases).
  2. Quantise the colours to a limited palette using k-means in OKLab space, so the
     reduced palette is perceptually even rather than muddy.
  3. Keep alpha crisp (hard edges, no semi-transparent fringe).
  4. Optionally re-upscale with nearest-neighbour for a big, sharp preview.

Pure CPU / numpy + Pillow. No GPU needed, runs anywhere.

Algorithm ported unchanged from reference/pixelize.py (proven Phase-0 code).
Default grid: 64px logical, displayed upscaled — the classic look (PLAN.md §6;
briefly 256 before the chunky-vs-smooth comparison settled it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .color import linear_to_srgb, oklab_to_linear, srgb_to_oklab
from .kmeans import kmeans

if TYPE_CHECKING:
    from .palette import Palette

DEFAULT_SIZE = 64
DEFAULT_COLORS = 16


def pixelize(
    img: Image.Image,
    size: int = DEFAULT_SIZE,
    colors: int = DEFAULT_COLORS,
    alpha_threshold: int = 128,
    seed: int = 0,
    palette: Palette | None = None,
) -> Image.Image:
    """Return a small RGBA pixel-art image (longest side == `size`).

    With `palette` set, every opaque pixel maps to the nearest locked-palette
    colour (OKLab distance) instead of a per-image k-means palette — the world
    cohesion mode (handover §15). `colors` is ignored in that case.
    """
    img = img.convert("RGBA")
    w, h = img.size

    # 1. Downscale to the target grid. BOX = area averaging: clean, no aliasing.
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = img.resize((nw, nh), Image.BOX)

    arr = np.asarray(small).astype(np.float64)
    rgb, alpha = arr[..., :3], arr[..., 3]
    opaque = alpha >= alpha_threshold          # only cluster real pixels

    # 2. Palette quantise over the opaque pixels: locked palette if given,
    #    otherwise per-image k-means in OKLab.
    out_rgb = rgb.copy()
    if opaque.any():
        if palette is not None:
            out_rgb[opaque] = palette.map(rgb[opaque])
        else:
            lab = srgb_to_oklab(rgb)
            flat = lab[opaque]
            distinct = np.unique(np.round(flat, 4), axis=0).shape[0]
            k = max(1, min(colors, distinct))
            centers, labels = kmeans(flat, k, seed=seed)
            out_rgb[opaque] = linear_to_srgb(oklab_to_linear(centers[labels]))

    # 3. Crisp alpha — no soft fringe.
    out_alpha = np.where(alpha >= alpha_threshold, 255, 0)
    out = np.dstack([np.round(out_rgb).astype(np.uint8),
                     out_alpha.astype(np.uint8)])
    return Image.fromarray(out, "RGBA")


def upscale_preview(img: Image.Image, factor: int) -> Image.Image:
    """Nearest-neighbour blow-up so you can actually see the little sprite."""
    return img.resize((img.width * factor, img.height * factor), Image.NEAREST)
