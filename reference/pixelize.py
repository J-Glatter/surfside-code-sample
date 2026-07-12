"""
pixelize.py — turn a normal image into clean pixel art.

The generator (Stable Diffusion) gives you "pixel-style" mush. This module does
the part that actually makes it pixel art:

  1. Downscale to a small grid with a good area filter (not nearest — that aliases).
  2. Quantise the colours to a fixed palette using k-means in OKLab space, so the
     reduced palette is perceptually even rather than muddy (the RGB-quantise problem).
  3. Keep alpha crisp (hard edges, no semi-transparent fringe).
  4. Optionally re-upscale with nearest-neighbour for a big, sharp preview.

Pure CPU / numpy + Pillow. No GPU needed, runs anywhere.

CLI:
    python pixelize.py input.png -o out.png --size 64 --colors 16 --preview 8
"""

from __future__ import annotations
import argparse
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Colour space: sRGB <-> linear <-> OKLab
# k-means in OKLab clusters colours the way the eye groups them, so a 16-colour
# palette looks intentional instead of blotchy.
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)
    return np.clip(c * 255.0, 0, 255)


def _linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ], axis=-1)


def _oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return np.stack([
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ], axis=-1)


# ---------------------------------------------------------------------------
# k-means (k-means++ init) — small enough to keep dependency-free.
# Operates on the handful of thousand pixels in a downscaled sprite, so it's fast.
# ---------------------------------------------------------------------------

def _kmeans(data: np.ndarray, k: int, iters: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    centers = np.empty((k, data.shape[1]), dtype=data.dtype)
    centers[0] = data[rng.integers(n)]
    d2 = ((data - centers[0]) ** 2).sum(1)
    for i in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centers[i] = data[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, ((data - centers[i]) ** 2).sum(1))

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        new_labels = dists.argmin(1)
        new_centers = np.array([
            data[new_labels == j].mean(0) if np.any(new_labels == j) else centers[j]
            for j in range(k)
        ])
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            centers, labels = new_centers, new_labels
            break
        centers, labels = new_centers, new_labels
    return centers, labels


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def pixelize(
    img: Image.Image,
    size: int = 64,
    colors: int = 16,
    alpha_threshold: int = 128,
    seed: int = 0,
) -> Image.Image:
    """Return a small RGBA pixel-art image (longest side == `size`)."""
    img = img.convert("RGBA")
    w, h = img.size

    # 1. Downscale to the target grid. BOX = area averaging: clean, no aliasing.
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = img.resize((nw, nh), Image.BOX)

    arr = np.asarray(small).astype(np.float64)
    rgb, alpha = arr[..., :3], arr[..., 3]
    opaque = alpha >= alpha_threshold          # only cluster real pixels

    # 2. Palette quantise in OKLab over the opaque pixels.
    lab = _linear_to_oklab(_srgb_to_linear(rgb))
    flat = lab[opaque]
    out_rgb = rgb.copy()
    if flat.shape[0] > 0:
        distinct = np.unique(np.round(flat, 4), axis=0).shape[0]
        k = max(1, min(colors, distinct))
        centers, labels = _kmeans(flat, k, seed=seed)
        out_rgb[opaque] = _linear_to_srgb(_oklab_to_linear(centers[labels]))

    # 3. Crisp alpha — no soft fringe.
    out_alpha = np.where(alpha >= alpha_threshold, 255, 0)
    out = np.dstack([np.round(out_rgb).astype(np.uint8),
                     out_alpha.astype(np.uint8)])
    return Image.fromarray(out, "RGBA")


def upscale_preview(img: Image.Image, factor: int) -> Image.Image:
    """Nearest-neighbour blow-up so you can actually see the little sprite."""
    return img.resize((img.width * factor, img.height * factor), Image.NEAREST)


def _cli():
    p = argparse.ArgumentParser(description="Convert an image into pixel art.")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="pixel.png")
    p.add_argument("--size", type=int, default=64, help="longest side in pixels")
    p.add_argument("--colors", type=int, default=16, help="palette size")
    p.add_argument("--alpha-threshold", type=int, default=128)
    p.add_argument("--preview", type=int, default=0,
                   help="also save an NxN-upscaled preview (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    src = Image.open(a.input)
    px = pixelize(src, size=a.size, colors=a.colors,
                  alpha_threshold=a.alpha_threshold, seed=a.seed)
    px.save(a.output)
    print(f"wrote {a.output} ({px.width}x{px.height}, <= {a.colors} colours)")
    if a.preview > 0:
        prev = a.output.rsplit(".", 1)[0] + f"_preview.png"
        upscale_preview(px, a.preview).save(prev)
        print(f"wrote {prev} ({px.width * a.preview}x{px.height * a.preview})")


if __name__ == "__main__":
    _cli()
