#!/usr/bin/env python3
"""Generate a diverse batch of clean candidates to seed a style-LoRA training set.

A style LoRA (Checkpoint C) learns YOUR look from a spread of subjects, so this
runs a list of prompts through the director — best-of-N each — and collects
every candidate still into one `_review/` folder with a contact sheet. Cherry-
pick the ~20 cleanest into `style_refs/` and hand them to `dataset prep`.

Builds the SDXL pipe ONCE and reuses it for every prompt (calling `spriteforge
make` per prompt would reload the ~7 GB model each time). GPU/pod only.

    python scripts/seed_style_set.py -o out/seed --candidates 10 \
        --palette out/world/game.json

Edit PROMPTS below to match the world you're building.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from spriteforge.director import execute_plan, plan_asset
from spriteforge.generate import build_pipe

# A spread across creatures / props / characters — one world, varied subjects.
# Keep them describable in a few words; the director art-directs the rest.
PROMPTS = [
    ("slime", "a small gelatinous slime monster"),
    ("mushroom", "a cute walking mushroom creature"),
    ("bat", "a small cartoon bat"),
    ("eyeball", "a floating eyeball monster"),
    ("chest", "a wooden treasure chest with iron bands"),
    ("barrel", "a wooden barrel"),
    ("potion", "a red health potion bottle"),
    ("sword", "an ornate steel sword"),
    ("shield", "a round wooden shield with iron rim"),
    ("torch", "a lit wooden wall torch"),
    ("knight", "a brave knight in green armour with a sword and shield"),
    ("wizard", "an old wizard in a blue robe holding a staff"),
    ("goblin", "a small green goblin warrior"),
    ("skeleton", "a skeleton warrior holding a sword"),
]


def _collect_stills(asset_dir: Path) -> list[Path]:
    """The pixelized candidate stills a run produced, whatever the workstream:
    candidates/cand_*.png (creatures/props) or hero_*.png (characters)."""
    stills: list[Path] = []
    cand = asset_dir / "candidates"
    if cand.exists():
        stills += sorted(cand.glob("cand_*.png"))
    stills += [p for p in sorted(asset_dir.glob("hero_*.png"))
               if "_raw" not in p.name]
    if not stills and (asset_dir / "sprite.png").exists():
        stills = [asset_dir / "sprite.png"]
    return stills


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="out/seed")
    ap.add_argument("--candidates", type=int, default=10,
                    help="candidates per prompt (more = more to pick from)")
    ap.add_argument("--palette", default=None, help="lock to a world palette")
    ap.add_argument("--offline", action="store_true",
                    help="skip the LLM director (keyword routing only)")
    ap.add_argument("--sd15", action="store_true")
    ap.add_argument("--style-lora", default=None,
                    help="seed from an existing style LoRA (rare)")
    ap.add_argument("--style-trigger", default=None)
    args = ap.parse_args()

    backend = "sd15" if args.sd15 else None
    palette = None
    if args.palette:
        from spriteforge.palette import Palette

        palette = Palette.load(args.palette)

    out = Path(args.output)
    review = out / "_review"
    review.mkdir(parents=True, exist_ok=True)

    print(f"building the {backend or 'sdxl'} pipe once "
          f"(first run downloads weights)...")
    pipe = build_pipe(backend=backend, style_lora=args.style_lora)

    collected: list[Path] = []
    for slug, prompt in PROMPTS:
        print(f"\n== {slug}: {prompt} ==")
        plan = plan_asset(prompt, offline=args.offline)
        plan.actions = []          # stills only — skip procedural animation here
        execute_plan(
            plan, out / slug, pipe=pipe, palette=palette,
            candidates=args.candidates, backend=backend,
            pick=0,                # collect all; skip the CLIP pick (no model load)
            style_lora=args.style_lora, style_trigger=args.style_trigger,
        )
        stills = _collect_stills(out / slug)
        for p in stills:
            dest = review / f"{slug}__{p.stem}.png"
            shutil.copy2(p, dest)
            collected.append(dest)
        print(f"  collected {len(stills)} stills")

    if collected:
        thumbs = [Image.open(p).convert("RGBA") for p in collected]
        cw = max(i.width for i in thumbs)
        ch = max(i.height for i in thumbs)
        cols = 10
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGBA", (cols * cw, rows * ch), (40, 40, 40, 255))
        for k, im in enumerate(thumbs):
            r, c = divmod(k, cols)
            sheet.paste(im, (c * cw, r * ch), im)
        sheet.resize((cols * cw * 3, rows * ch * 3), Image.NEAREST).save(
            review / "contact_sheet.png")

    print(f"\n{len(collected)} candidates in {review}")
    print("open contact_sheet.png, then copy the ~20 cleanest into style_refs/:")
    print("  spriteforge dataset prep style_refs/*.png -o style_ds "
          "--trigger myworld_style --name myworld-style")


if __name__ == "__main__":
    main()
