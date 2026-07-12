"""Seamless/tiling generation for environment tiles (handover §15/§19).

The standard Stable Diffusion trick: switch every Conv2d in the UNet and VAE to
circular padding, so the canvas wraps around and the output tiles edge-to-edge.
Same style LoRA + locked palette rules apply on top, so tiles stay coherent
with the rest of the world.

`seam_error` is the CPU-side verifier: how hard the wrapped edges disagree.
Circular-padded generations should score near the interior-difference baseline;
ordinary images show a clear seam spike.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .color import srgb_to_oklab


def enable_tiling(pipe) -> int:
    """Switch all Conv2d layers in the pipe's UNet/VAE to circular padding.

    Returns the number of layers patched. Idempotent.
    """
    import torch

    patched = 0
    for component in ("unet", "vae"):
        module = getattr(pipe, component, None)
        if module is None:
            continue
        for layer in module.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.padding_mode = "circular"
                patched += 1
    return patched


def disable_tiling(pipe) -> int:
    """Restore default zero padding. Returns the number of layers restored."""
    import torch

    restored = 0
    for component in ("unet", "vae"):
        module = getattr(pipe, component, None)
        if module is None:
            continue
        for layer in module.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.padding_mode = "zeros"
                restored += 1
    return restored


def seam_error(img: Image.Image) -> float:
    """Mean OKLab discontinuity across the wrap-around seams, normalised by the
    image's interior discontinuity (1.0 ≈ seams look like any other pixel row;
    >> 1 means a visible seam)."""
    arr = np.asarray(img.convert("RGB")).astype(np.float64)
    lab = srgb_to_oklab(arr)

    def row_diff(a, b):
        return float(np.linalg.norm(a - b, axis=-1).mean())

    seam = (row_diff(lab[0], lab[-1]) + row_diff(lab[:, 0], lab[:, -1])) / 2
    interior = (row_diff(lab[1:], lab[:-1]) + row_diff(lab[:, 1:], lab[:, :-1])) / 2
    return seam / max(interior, 1e-9)


def tile_preview(img: Image.Image, repeat: int = 3) -> Image.Image:
    """The eyeball check: the tile repeated in an NxN grid."""
    out = Image.new(img.mode, (img.width * repeat, img.height * repeat))
    for y in range(repeat):
        for x in range(repeat):
            out.paste(img, (x * img.width, y * img.height))
    return out
