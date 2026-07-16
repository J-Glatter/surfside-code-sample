"""Subject isolation — strip the background SD paints behind every sprite.

Checkpoint A/B finding: raw renders arrive fully opaque with sky/grass baked
in, so the pixelizer's crisp-alpha step never fires and no engine can use the
sprite. Fix: prompt for a plain background (director's job), then remove it
here — estimate the background colour from the image border, flood-fill the
border-connected region that matches it (perceptual OKLab distance), and make
that region transparent.

Deliberately conservative: if no plain border-connected background is found
(busy scene, subject touching all edges), the image is returned untouched and
`found` is False — better an opaque sprite than an eaten subject. Tiles never
want isolation. Pure numpy/CPU, deterministic.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .color import srgb_to_oklab

DEFAULT_TOLERANCE = 0.10   # OKLab distance counted as "background-coloured"
MIN_COVERAGE = 0.15        # below this, assume there was no plain background
MAX_COVERAGE = 0.98        # above this, something went wrong — don't eat the image


def isolate_subject(
    img: Image.Image,
    tolerance: float = DEFAULT_TOLERANCE,
    min_coverage: float = MIN_COVERAGE,
    max_coverage: float = MAX_COVERAGE,
) -> tuple[Image.Image, bool]:
    """Return (RGBA image, background_found).

    Background = the connected region of near-border-colour pixels that
    touches the image border. Interior pixels of similar colour (eyes,
    highlights) are preserved — connectivity is what protects them.
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float64)
    lab = srgb_to_oklab(arr[..., :3])

    border = np.concatenate([lab[0], lab[-1], lab[1:-1, 0], lab[1:-1, -1]])
    bg_color = np.median(border, axis=0)
    candidate = np.linalg.norm(lab - bg_color, axis=-1) < tolerance

    # connected component of candidates touching the border (iterative flood)
    mask = np.zeros_like(candidate)
    mask[0], mask[-1] = candidate[0], candidate[-1]
    mask[:, 0], mask[:, -1] = candidate[:, 0], candidate[:, -1]
    while True:
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        grown &= candidate
        if (grown == mask).all():
            break
        mask = grown

    coverage = float(mask.mean())
    if not (min_coverage <= coverage <= max_coverage):
        return rgba, False

    out = arr.astype(np.uint8).copy()
    out[..., 3] = np.where(mask, 0, out[..., 3])
    return Image.fromarray(out, "RGBA"), True
