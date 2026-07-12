"""Locked palettes — the world-cohesion trick (handover §15).

Crush every asset (characters, enemies, buildings) to the *same* fixed set of
colours and separately-generated assets instantly read as one game.

Formats:
  * JSON        — {"name": "...", "colors": ["#aabbcc", ...]}
  * .hex        — one hex colour per line (Lospec palette format; lospec.com
                  offers thousands of curated game palettes as .hex downloads)
  * PNG swatch  — a small image whose distinct opaque colours are the palette

A palette can also be *extracted* from a set of reference images (k-means in
OKLab over their combined opaque pixels) — generate freely, pick favourites,
extract, then lock the result for everything after.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .color import oklab_to_srgb, srgb_to_oklab
from .kmeans import kmeans

_EXTRACT_MAX_SIDE = 128  # bound extraction cost; plenty of pixels for k-means


class Palette:
    """An ordered set of RGB colours, with nearest-colour mapping in OKLab."""

    def __init__(self, colors, name: str = "palette"):
        arr = np.asarray(list(colors), dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] == 0:
            raise ValueError("palette needs at least one colour")
        self.colors = np.clip(np.round(arr), 0, 255).astype(np.uint8)
        self.name = name
        self._oklab: np.ndarray | None = None

    # -- basics ---------------------------------------------------------------

    def __len__(self) -> int:
        return self.colors.shape[0]

    def __eq__(self, other) -> bool:
        return isinstance(other, Palette) and np.array_equal(self.colors, other.colors)

    @property
    def hex_colors(self) -> list[str]:
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in self.colors]

    @property
    def oklab(self) -> np.ndarray:
        if self._oklab is None:
            self._oklab = srgb_to_oklab(self.colors.astype(np.float64))
        return self._oklab

    # -- nearest-colour mapping -------------------------------------------------

    def map(self, rgb: np.ndarray) -> np.ndarray:
        """Map sRGB values (..., 3) to the nearest palette colour (OKLab distance).

        Returns uint8 with exactly the palette's colour values.
        """
        lab = srgb_to_oklab(np.asarray(rgb, dtype=np.float64))
        flat = lab.reshape(-1, 3)
        d2 = ((flat[:, None, :] - self.oklab[None, :, :]) ** 2).sum(2)
        idx = d2.argmin(1)
        return self.colors[idx].reshape(np.asarray(rgb).shape)

    # -- extraction --------------------------------------------------------------

    @classmethod
    def extract(
        cls,
        images: list[Image.Image],
        colors: int = 16,
        alpha_threshold: int = 128,
        seed: int = 0,
        name: str = "extracted",
    ) -> Palette:
        """Derive a shared palette from reference images: k-means in OKLab over
        their combined opaque pixels. Deterministic for a given seed."""
        pixels = []
        for img in images:
            img = img.convert("RGBA")
            scale = _EXTRACT_MAX_SIDE / max(img.size)
            if scale < 1:
                img = img.resize(
                    (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                    Image.BOX,
                )
            arr = np.asarray(img).astype(np.float64)
            opaque = arr[..., 3] >= alpha_threshold
            pixels.append(arr[..., :3][opaque])
        flat = np.concatenate(pixels) if pixels else np.empty((0, 3))
        if flat.shape[0] == 0:
            raise ValueError("no opaque pixels to extract a palette from")

        lab = srgb_to_oklab(flat)
        distinct = np.unique(np.round(lab, 4), axis=0).shape[0]
        k = max(1, min(colors, distinct))
        centers, _ = kmeans(lab, k, seed=seed)
        rgb = oklab_to_srgb(centers)
        # stable, human-friendly ordering: dark -> light
        order = np.argsort(centers[:, 0])
        return cls(rgb[order], name=name)

    # -- I/O ----------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> Palette:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text())
            return cls([_parse_hex(c) for c in data["colors"]],
                       name=data.get("name", path.stem))
        if suffix == ".hex":
            lines = [ln.strip() for ln in path.read_text().splitlines()]
            hexes = [ln for ln in lines if ln and not ln.startswith(";")]
            return cls([_parse_hex(h) for h in hexes], name=path.stem)
        if suffix == ".png":
            return cls.from_swatch(Image.open(path), name=path.stem)
        raise ValueError(f"unknown palette format: {path.name} (use .json, .hex or .png)")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            path.write_text(json.dumps(
                {"name": self.name, "colors": self.hex_colors}, indent=2) + "\n")
        elif suffix == ".hex":
            path.write_text("\n".join(h.lstrip("#") for h in self.hex_colors) + "\n")
        elif suffix == ".png":
            self.to_swatch().save(path)
        else:
            raise ValueError(f"unknown palette format: {path.name} (use .json, .hex or .png)")

    @classmethod
    def from_swatch(cls, img: Image.Image, name: str = "swatch") -> Palette:
        """Palette = distinct opaque colours of a swatch image, in scan order."""
        arr = np.asarray(img.convert("RGBA"))
        opaque = arr[..., 3] == 255
        seen: dict[tuple[int, int, int], None] = {}
        for px in arr[opaque][:, :3]:
            seen.setdefault(tuple(int(v) for v in px))
        if not seen:
            raise ValueError("swatch image has no opaque pixels")
        if len(seen) > 256:
            raise ValueError(f"swatch has {len(seen)} colours — not a palette image")
        return cls(list(seen), name=name)

    def to_swatch(self, cell: int = 1) -> Image.Image:
        """A (N*cell)x(cell) image, one cell per colour, left to right."""
        row = np.repeat(self.colors[None, :, :], cell, axis=0)
        row = np.repeat(row, cell, axis=1)
        rgba = np.dstack([row, np.full(row.shape[:2], 255, dtype=np.uint8)])
        return Image.fromarray(rgba, "RGBA")


def _parse_hex(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"bad hex colour: {s!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
