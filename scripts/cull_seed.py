#!/usr/bin/env python3
"""Auto-cull a seed batch down to clean single-subject sprites for LoRA training.

The two failure modes a LoRA must never learn from:
  * collage / sprite-sheet — SDXL drew a GRID of the item, not one item. The
    opaque pixels fragment into many disconnected blobs.
  * failed isolation — the background wasn't stripped (white box), so opaque
    pixels fill the frame edge to edge (little transparency).

Both are cheap to detect: a clean isolated subject has plenty of transparent
margin AND most of its opaque pixels in ONE connected blob. Keepers are copied
to --out; the rest are moved aside for inspection.

    python scripts/cull_seed.py out/seed/_review -o style_refs
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def largest_component_frac(mask: np.ndarray) -> tuple[float, int]:
    """(biggest blob / all opaque, number of blobs) via iterative flood fill."""
    total = int(mask.sum())
    if total == 0:
        return 0.0, 0
    visited = np.zeros_like(mask)
    h, w = mask.shape
    best = comps = 0
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or visited[sy, sx]:
                continue
            comps += 1
            size = 0
            stack = [(sy, sx)]
            visited[sy, sx] = True
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                            and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            best = max(best, size)
    return best / total, comps


def metrics(path: Path) -> dict:
    a = np.asarray(Image.open(path).convert("RGBA"))
    opaque = a[..., 3] > 128
    transparent_frac = float((~opaque).mean())
    largest_frac, comps = largest_component_frac(opaque)
    return {"transparent_frac": transparent_frac,
            "largest_frac": largest_frac, "components": comps}


def is_clean(m: dict, min_transparent: float, min_largest: float) -> bool:
    # enough margin (not a full-frame collage / un-stripped bg) AND most opaque
    # pixels in one blob (not a grid of items)
    return (m["transparent_frac"] >= min_transparent
            and m["largest_frac"] >= min_largest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("review_dir", help="the seed batch's _review/ folder")
    ap.add_argument("-o", "--out", default="style_refs",
                    help="where clean keepers are copied")
    ap.add_argument("--reject-dir", default=None,
                    help="move rejects here for inspection (default: <review>/_rejected)")
    ap.add_argument("--min-transparent", type=float, default=0.12,
                    help="reject below this transparent fraction (bg not stripped)")
    ap.add_argument("--min-largest", type=float, default=0.55,
                    help="reject below this largest-blob fraction (collage)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, copy/move nothing")
    args = ap.parse_args()

    review = Path(args.review_dir)
    imgs = sorted(p for p in review.glob("*.png") if p.name != "contact_sheet.png")
    if not imgs:
        raise SystemExit(f"no candidate PNGs in {review}")

    out = Path(args.out)
    reject = Path(args.reject_dir) if args.reject_dir else review / "_rejected"
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        reject.mkdir(parents=True, exist_ok=True)

    kept = []
    for p in imgs:
        m = metrics(p)
        keep = is_clean(m, args.min_transparent, args.min_largest)
        tag = "keep" if keep else "CULL"
        print(f"{tag}  {p.name:32s} "
              f"transp={m['transparent_frac']:.2f} "
              f"largest={m['largest_frac']:.2f} blobs={m['components']}")
        if args.dry_run:
            if keep:
                kept.append(p)
            continue
        if keep:
            shutil.copy2(p, out / p.name)
            kept.append(p)
        else:
            shutil.move(str(p), reject / p.name)

    print(f"\nkept {len(kept)}/{len(imgs)} clean singles"
          + (f" -> {out}" if not args.dry_run else " (dry run)"))
    if len(kept) > 30:
        print("plenty — thin to your ~20 favourites before dataset prep.")
    elif len(kept) < 12:
        print("thin — loosen --min-largest / --min-transparent, or seed more.")
    print("then: spriteforge dataset prep "
          f"{out}/*.png -o style_ds --trigger myworld_style --name myworld-style")


if __name__ == "__main__":
    main()
