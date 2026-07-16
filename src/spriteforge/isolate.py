"""Subject isolation — strip the background SD paints behind every sprite.

Checkpoint A/B finding: raw renders arrive fully opaque with sky/grass baked
in, so the pixelizer's crisp-alpha step never fires and no engine can use the
sprite. Fix: prompt for a plain background (director's job), then remove it
here — estimate the background colour from the image border, flood-fill the
border-connected region that matches it (perceptual OKLab distance), and make
that region transparent.

Two tiers (field-tested at Checkpoint A/B — the pixel-art LoRA was trained on
full scenes, so prompting for a plain background often loses):

  1. flood fill — free and deterministic when the background really is plain;
  2. rembg/u2net — ML salient-object cutout for arbitrary backgrounds (skies,
     hills, starfields). Optional [isolate] extra; used when tier 1 finds
     nothing and the package is installed.

If neither succeeds the image is returned untouched with method None — better
an opaque sprite than an eaten subject. Tiles never want isolation.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .color import srgb_to_oklab

DEFAULT_TOLERANCE = 0.10   # OKLab distance counted as "background-coloured"
MIN_COVERAGE = 0.15        # below this, assume there was no plain background
MAX_COVERAGE = 0.98        # above this, something went wrong — don't eat the image

_REMBG_SESSION = None      # u2net loads once (~170 MB weights, cached on disk)


def _rembg_cutout(img: Image.Image) -> Image.Image | None:
    """ML cutout via rembg/u2net; None if unavailable, failing, or degenerate.

    Any failure (package missing, weight download blocked, runtime error) must
    degrade to None — isolation is an enhancement, never allowed to take down
    a generation job.
    """
    global _REMBG_SESSION
    try:
        from rembg import new_session, remove

        if _REMBG_SESSION is None:
            _REMBG_SESSION = new_session("u2net")  # first use downloads ~170 MB
        out = remove(img.convert("RGBA"), session=_REMBG_SESSION)
    except ImportError:
        return None
    except Exception as e:  # noqa: BLE001 — e.g. weight download blocked/offline
        print(f"isolate: rembg unavailable ({type(e).__name__}: {e})")
        return None
    removed = float((np.asarray(out)[..., 3] < 128).mean())
    if not (0.02 <= removed <= 0.98):   # cut nothing or cut everything: distrust
        return None
    return out


def isolate_subject(
    img: Image.Image,
    tolerance: float = DEFAULT_TOLERANCE,
    min_coverage: float = MIN_COVERAGE,
    max_coverage: float = MAX_COVERAGE,
) -> tuple[Image.Image, str | None]:
    """Return (RGBA image, method) — method is "flood", "rembg", or None.

    Tier 1 background = the connected region of near-border-colour pixels that
    touches the image border. Interior pixels of similar colour (eyes,
    highlights) are preserved — connectivity is what protects them.
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float64)
    lab = srgb_to_oklab(arr[..., :3])

    # An essentially uniform image has no subject to find — bail before either
    # tier (u2net happily hallucinates a cutout on a blank canvas).
    spread = float(np.linalg.norm(lab - lab.mean(axis=(0, 1)), axis=-1).max())
    if spread < tolerance:
        return rgba, None

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
    if min_coverage <= coverage <= max_coverage:
        out = arr.astype(np.uint8).copy()
        out[..., 3] = np.where(mask, 0, out[..., 3])
        return Image.fromarray(out, "RGBA"), "flood"

    cutout = _rembg_cutout(rgba)               # tier 2: arbitrary backgrounds
    if cutout is not None:
        return cutout, "rembg"
    return rgba, None
