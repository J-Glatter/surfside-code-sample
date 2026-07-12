"""The `spriteforge` command line.

Subcommands (Milestone A set — `palette` lands in Milestone B):

    spriteforge pixelize input.png -o out.png --size 256 --colors 16 --preview 4
    spriteforge generate "a brave knight in green armour, full body" -o knight.png

Flag names preserved from the original Phase-0 scripts (reference/*.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pixelize import DEFAULT_COLORS, DEFAULT_SIZE, pixelize, upscale_preview


def _add_common_output_args(p: argparse.ArgumentParser, default_output: str) -> None:
    p.add_argument("-o", "--output", default=default_output)
    p.add_argument("--size", type=int, default=DEFAULT_SIZE,
                   help="final sprite size, longest side in pixels")
    p.add_argument("--colors", type=int, default=DEFAULT_COLORS,
                   help="palette size (ignored with --palette)")
    p.add_argument("--palette", default=None, metavar="FILE",
                   help="locked palette (.json/.hex/.png) — world cohesion mode")
    p.add_argument("--preview", type=int, default=0,
                   help="also save an Nx nearest-neighbour preview (0 = off)")


def _load_palette(a: argparse.Namespace):
    if a.palette is None:
        return None
    from .palette import Palette

    return Palette.load(a.palette)


def _save_sprite(sprite, output: str, colors: int, preview: int) -> None:
    sprite.save(output)
    print(f"wrote {output} ({sprite.width}x{sprite.height}, <= {colors} colours)")
    if preview > 0:
        prev = output.rsplit(".", 1)[0] + "_preview.png"
        upscale_preview(sprite, preview).save(prev)
        print(f"wrote {prev} ({sprite.width * preview}x{sprite.height * preview})")


def _cmd_pixelize(a: argparse.Namespace) -> None:
    from PIL import Image

    src = Image.open(a.input)
    sprite = pixelize(src, size=a.size, colors=a.colors,
                      alpha_threshold=a.alpha_threshold, seed=a.seed,
                      palette=_load_palette(a))
    _save_sprite(sprite, a.output, a.colors, a.preview)


def _cmd_generate(a: argparse.Namespace) -> None:
    # Lazy import: needs the [generate] extra (torch/diffusers) — GPU boxes only.
    from .generate import DEFAULT_NEGATIVE, build_pipe, generate  # noqa: F401

    fp16 = True if a.fp16 else False if a.fp32 else None  # None = auto per device
    use_lora = not a.no_lora
    pipe = build_pipe(fp16=fp16, use_lora=use_lora)
    if a.tile:
        from .tiling import enable_tiling

        patched = enable_tiling(pipe)
        print(f"seamless tiling on ({patched} conv layers wrapped)")
    print("generating...")
    raw = generate(pipe, a.prompt, negative=a.negative, steps=a.steps,
                   guidance=a.guidance, seed=a.seed, use_lora=use_lora)
    if a.raw:
        raw.save(a.raw)
        print(f"wrote raw render {a.raw}")

    sprite = pixelize(raw, size=a.size, colors=a.colors, palette=_load_palette(a))
    _save_sprite(sprite, a.output, a.colors, a.preview)


def _cmd_palette_extract(a: argparse.Namespace) -> None:
    from PIL import Image

    from .palette import Palette

    images = [Image.open(p) for p in a.images]
    pal = Palette.extract(images, colors=a.colors, seed=a.seed,
                          name=a.name or "extracted")
    pal.save(a.output)
    print(f"wrote {a.output} ({len(pal)} colours): {' '.join(pal.hex_colors)}")


def _cmd_dataset_prep(a: argparse.Namespace) -> None:
    from .dataset import prep_dataset

    train_dir = prep_dataset(a.images, a.output, trigger=a.trigger,
                             repeats=a.repeats, class_word=a.class_word,
                             name=a.name)
    print(f"kohya dataset ready: {train_dir}")
    print(f"see {a.output}/NOTES.md for the training command")


def _cmd_refine(a: argparse.Namespace) -> None:
    # Lazy: needs the [generate] extra — GPU boxes only.
    from PIL import Image

    from .refine import build_img2img_pipe, refine

    fp16 = True if a.fp16 else False if a.fp32 else None
    pipe = build_img2img_pipe(fp16=fp16, character_lora=a.character_lora)
    hero = Image.open(a.hero)
    print("refining...")
    written = refine(pipe, hero, a.prompt, a.output,
                     per_variation=a.per_variation, strength=a.strength,
                     seed=a.seed)
    print(f"wrote {len(written)} candidates to {a.output}")


def _cmd_curate(a: argparse.Namespace) -> None:
    from .curate import curate

    candidates = sorted(p for p in Path(a.candidates).iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    winners = curate(candidates, a.hero, out_dir=a.output, keep=a.keep,
                     min_quality=a.min_quality)
    print(f"kept {len(winners)}/{len(candidates)} -> {a.output}")
    for s in winners:
        print(f"  identity={s.identity:.3f} quality={s.quality:+.3f}  {s.path.name}")


def _cmd_skeleton(a: argparse.Namespace) -> None:
    from .animate.skeleton import ACTIONS, DEFAULT_FRAMES, render_openpose, save_poses

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    poses = ACTIONS[a.action](a.frames or DEFAULT_FRAMES[a.action])
    for k, pose in enumerate(poses):
        render_openpose(pose, size=a.size).save(out / f"{a.action}_{k:02d}.png")
    save_poses(poses, out / f"{a.action}.json")
    print(f"wrote {len(poses)} conditioning frames + {a.action}.json to {out}")


def _cmd_animate(a: argparse.Namespace) -> None:
    # Lazy: needs the [animate] extra — GPU boxes only.
    from .animate.frames import build_animation_pipe
    from .animate.pipeline import animate_action

    fp16 = True if a.fp16 else False if a.fp32 else None
    pipe = build_animation_pipe(character_lora=a.character_lora, fp16=fp16)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"animating {a.action}: {a.frames or 'default'} frames x "
          f"{a.candidates} candidates...")
    locked, selection = animate_action(
        pipe, a.action, a.prompt,
        size=a.size, colors=a.colors, palette=_load_palette(a),
        frames=a.frames, n_candidates=a.candidates, seed=a.seed,
        raw_dir=a.raw_dir,
    )
    for k, frame in enumerate(locked):
        frame.save(out / f"{a.action}_{k:02d}.png")
    avg = sum(selection.costs[1:]) / max(1, len(selection.costs) - 1)
    print(f"wrote {len(locked)} frames to {out} "
          f"(mean transition cost {avg:.4f}; lower = smoother)")


def _cmd_sheet(a: argparse.Namespace) -> None:
    from PIL import Image

    from .animate.sheet import save_sheet

    actions: dict[str, list] = {}
    for spec in a.actions:
        if "=" not in spec:
            raise SystemExit(f"expected ACTION=FRAMES_DIR, got {spec!r}")
        name, dir_ = spec.split("=", 1)
        frames = sorted(Path(dir_).glob("*.png"))
        if not frames:
            raise SystemExit(f"no .png frames in {dir_}")
        actions[name] = [Image.open(p) for p in frames]

    fps = {}
    for spec in a.fps or []:
        name, value = spec.split("=", 1)
        fps[name] = int(value)

    metadata = save_sheet(actions, a.output, fps=fps or None)
    print(f"wrote {a.output} + sidecar json "
          f"({metadata['columns']}x{metadata['rows']} cells of "
          f"{metadata['cell_width']}x{metadata['cell_height']})")


def _cmd_plan(a: argparse.Namespace) -> None:
    from .director import DIRECTOR_MODEL, plan_asset

    plan = plan_asset(a.prompt, offline=a.offline, model=a.model or DIRECTOR_MODEL)
    print(plan.to_json())


def _cmd_make(a: argparse.Namespace) -> None:
    from .director import DIRECTOR_MODEL, Plan, execute_plan, plan_asset

    if a.plan_file:
        plan = Plan.from_json(Path(a.plan_file).read_text())
    elif a.prompt is None:
        raise SystemExit("make: provide a PROMPT or --plan-file")
    else:
        plan = plan_asset(a.prompt, offline=a.offline, model=a.model or DIRECTOR_MODEL)
    print(f"plan [{plan.source}]: {plan.workstream} — {plan.reasoning}")
    print(f"  prompt: {plan.enriched_prompt}")

    fp16 = True if a.fp16 else False if a.fp32 else None
    results = execute_plan(plan, a.output, palette=_load_palette(a),
                           seed=a.seed, fp16=fp16)
    for key, value in results.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        print(f"  {key}: {value}")


def _cmd_preview(a: argparse.Namespace) -> None:
    from .preview import gif_from_dir

    out = gif_from_dir(a.frames, a.output, fps=a.fps, scale=a.scale)
    print(f"wrote {out} ({a.fps} fps, {a.scale}x)")


def _cmd_palette_show(a: argparse.Namespace) -> None:
    from .palette import Palette

    pal = Palette.load(a.palette)
    print(f"{pal.name}: {len(pal)} colours")
    for h in pal.hex_colors:
        print(f"  {h}")
    if a.swatch:
        pal.to_swatch(cell=a.cell).save(a.swatch)
        print(f"wrote {a.swatch}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spriteforge",
        description="Prompt -> palette-coherent, game-ready pixel-art sprites.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pix = sub.add_parser("pixelize", help="convert an image into pixel art")
    p_pix.add_argument("input")
    _add_common_output_args(p_pix, default_output="pixel.png")
    p_pix.add_argument("--alpha-threshold", type=int, default=128)
    p_pix.add_argument("--seed", type=int, default=0)
    p_pix.set_defaults(func=_cmd_pixelize)

    p_gen = sub.add_parser("generate", help="generate a pixel-art sprite from a prompt")
    p_gen.add_argument("prompt")
    _add_common_output_args(p_gen, default_output="sprite.png")
    p_gen.add_argument("--steps", type=int, default=28)
    p_gen.add_argument("--guidance", type=float, default=7.0)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.add_argument("--negative",
                       default="3d render, realistic, photo, blurry, jpeg artifacts, "
                               "smooth shading")
    p_gen.add_argument("--no-lora", action="store_true")
    fp = p_gen.add_mutually_exclusive_group()
    fp.add_argument("--fp16", action="store_true", help="force float16 (auto-on for CUDA)")
    fp.add_argument("--fp32", action="store_true", help="force float32 (auto-on for MPS/CPU)")
    p_gen.add_argument("--raw", help="also save the pre-pixelized 512px render here")
    p_gen.add_argument("--tile", action="store_true",
                       help="seamless mode for environment tiles (wraps edge-to-edge)")
    p_gen.set_defaults(func=_cmd_generate)

    p_pal = sub.add_parser("palette", help="create and inspect locked palettes")
    pal_sub = p_pal.add_subparsers(dest="palette_command", required=True)

    p_ext = pal_sub.add_parser(
        "extract", help="derive a shared palette from reference images")
    p_ext.add_argument("images", nargs="+")
    p_ext.add_argument("-o", "--output", default="palette.json",
                       help="output palette (.json/.hex/.png)")
    p_ext.add_argument("--colors", type=int, default=DEFAULT_COLORS)
    p_ext.add_argument("--seed", type=int, default=0)
    p_ext.add_argument("--name", default=None)
    p_ext.set_defaults(func=_cmd_palette_extract)

    p_show = pal_sub.add_parser("show", help="print a palette's colours")
    p_show.add_argument("palette")
    p_show.add_argument("--swatch", default=None, metavar="FILE",
                        help="also save a swatch PNG here")
    p_show.add_argument("--cell", type=int, default=16,
                        help="swatch cell size in pixels")
    p_show.set_defaults(func=_cmd_palette_show)

    p_ds = sub.add_parser("dataset", help="prepare LoRA training datasets")
    ds_sub = p_ds.add_subparsers(dest="dataset_command", required=True)
    p_prep = ds_sub.add_parser("prep", help="build a kohya_ss-ready dataset")
    p_prep.add_argument("images", nargs="+")
    p_prep.add_argument("-o", "--output", required=True, help="dataset directory")
    p_prep.add_argument("--trigger", required=True,
                        help="unique trigger token baked into every caption")
    p_prep.add_argument("--repeats", type=int, default=10)
    p_prep.add_argument("--class-word", default=None,
                        help='optional class word, e.g. "character"')
    p_prep.add_argument("--name", default=None, help="LoRA output name")
    p_prep.set_defaults(func=_cmd_dataset_prep)

    p_ref = sub.add_parser(
        "refine", help="img2img variations from a hero image (the ratchet)")
    p_ref.add_argument("hero", help="the anchor image")
    p_ref.add_argument("--prompt", required=True,
                       help="base character description")
    p_ref.add_argument("-o", "--output", default="refined",
                       help="output directory for candidates")
    p_ref.add_argument("--per-variation", type=int, default=6)
    p_ref.add_argument("--strength", type=float, default=0.5,
                       help="img2img denoise strength (0=copy, 1=ignore anchor)")
    p_ref.add_argument("--character-lora", default=None,
                       help="stack a trained character LoRA (ratchet rounds >= 2)")
    p_ref.add_argument("--seed", type=int, default=0)
    ref_fp = p_ref.add_mutually_exclusive_group()
    ref_fp.add_argument("--fp16", action="store_true")
    ref_fp.add_argument("--fp32", action="store_true")
    p_ref.set_defaults(func=_cmd_refine)

    p_cur = sub.add_parser(
        "curate", help="auto-select the best candidates via CLIP scoring")
    p_cur.add_argument("candidates", help="directory of candidate images")
    p_cur.add_argument("--hero", required=True, help="the identity anchor image")
    p_cur.add_argument("-o", "--output", default="curated")
    p_cur.add_argument("--keep", type=int, default=10)
    p_cur.add_argument("--min-quality", type=float, default=0.0,
                       help="quality-margin bar candidates must clear")
    p_cur.set_defaults(func=_cmd_curate)

    p_skel = sub.add_parser(
        "skeleton", help="dump pose-conditioning images for an action (CPU)")
    p_skel.add_argument("--action", choices=("walk", "run", "jump"), required=True)
    p_skel.add_argument("-o", "--output", default="skeletons")
    p_skel.add_argument("--frames", type=int, default=None)
    p_skel.add_argument("--size", type=int, default=512)
    p_skel.set_defaults(func=_cmd_skeleton)

    p_anim = sub.add_parser(
        "animate", help="generate an animated action via pose control + selection")
    p_anim.add_argument("prompt", help="character description")
    p_anim.add_argument("--action", choices=("walk", "run", "jump"), required=True)
    p_anim.add_argument("-o", "--output", default="frames")
    p_anim.add_argument("--frames", type=int, default=None,
                        help="frame count (default: per-action canonical)")
    p_anim.add_argument("--candidates", type=int, default=100,
                        help="candidates generated per frame")
    p_anim.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p_anim.add_argument("--colors", type=int, default=DEFAULT_COLORS)
    p_anim.add_argument("--palette", default=None, metavar="FILE")
    p_anim.add_argument("--character-lora", default=None)
    p_anim.add_argument("--seed", type=int, default=0)
    p_anim.add_argument("--raw-dir", default=None,
                        help="also keep the raw 512px candidates here (debug)")
    anim_fp = p_anim.add_mutually_exclusive_group()
    anim_fp.add_argument("--fp16", action="store_true")
    anim_fp.add_argument("--fp32", action="store_true")
    p_anim.set_defaults(func=_cmd_animate)

    p_sheet = sub.add_parser("sheet", help="pack frame dirs into a sprite sheet")
    p_sheet.add_argument("actions", nargs="+", metavar="ACTION=FRAMES_DIR")
    p_sheet.add_argument("-o", "--output", default="sheet.png")
    p_sheet.add_argument("--fps", action="append", metavar="ACTION=FPS",
                         help="fps hint per action (repeatable)")
    p_sheet.set_defaults(func=_cmd_sheet)

    p_prev = sub.add_parser(
        "preview", help="loop a directory of frames as a GIF at game speed")
    p_prev.add_argument("frames", help="directory of frame PNGs (sorted by name)")
    p_prev.add_argument("-o", "--output", default="preview.gif")
    p_prev.add_argument("--fps", type=int, default=10)
    p_prev.add_argument("--scale", type=int, default=4,
                        help="nearest-neighbour upscale factor")
    p_prev.set_defaults(func=_cmd_preview)

    p_plan = sub.add_parser(
        "plan", help="route a request to a workstream (LLM director, stage 0)")
    p_plan.add_argument("prompt", help="the asset request")
    p_plan.add_argument("--offline", action="store_true",
                        help="use keyword heuristics instead of the LLM")
    p_plan.add_argument("--model", default=None,
                        help="director model (default: claude-opus-4-8)")
    p_plan.set_defaults(func=_cmd_plan)

    p_make = sub.add_parser(
        "make", help="plan a request and run the chosen workstream end-to-end")
    p_make.add_argument("prompt", nargs="?", default=None,
                        help="the asset request (omit with --plan-file)")
    p_make.add_argument("--plan-file", default=None,
                        help="execute a saved/edited plan JSON instead of planning")
    p_make.add_argument("-o", "--output", default="assets")
    p_make.add_argument("--palette", default=None, metavar="FILE")
    p_make.add_argument("--seed", type=int, default=0)
    p_make.add_argument("--offline", action="store_true")
    p_make.add_argument("--model", default=None)
    make_fp = p_make.add_mutually_exclusive_group()
    make_fp.add_argument("--fp16", action="store_true")
    make_fp.add_argument("--fp32", action="store_true")
    p_make.set_defaults(func=_cmd_make)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
