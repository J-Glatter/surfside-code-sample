"""The `spriteforge` command line.

Subcommands (Milestone A set — `palette` lands in Milestone B):

    spriteforge pixelize input.png -o out.png --size 256 --colors 16 --preview 4
    spriteforge generate "a brave knight in green armour, full body" -o knight.png

Flag names preserved from the original Phase-0 scripts (reference/*.py).
"""

from __future__ import annotations

import argparse

from .pixelize import DEFAULT_COLORS, DEFAULT_SIZE, pixelize, upscale_preview


def _add_common_output_args(p: argparse.ArgumentParser, default_output: str) -> None:
    p.add_argument("-o", "--output", default=default_output)
    p.add_argument("--size", type=int, default=DEFAULT_SIZE,
                   help="final sprite size, longest side in pixels")
    p.add_argument("--colors", type=int, default=DEFAULT_COLORS, help="palette size")
    p.add_argument("--preview", type=int, default=0,
                   help="also save an Nx nearest-neighbour preview (0 = off)")


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
                      alpha_threshold=a.alpha_threshold, seed=a.seed)
    _save_sprite(sprite, a.output, a.colors, a.preview)


def _cmd_generate(a: argparse.Namespace) -> None:
    # Lazy import: needs the [generate] extra (torch/diffusers) — GPU boxes only.
    from .generate import DEFAULT_NEGATIVE, build_pipe, generate  # noqa: F401

    fp16 = True if a.fp16 else False if a.fp32 else None  # None = auto per device
    use_lora = not a.no_lora
    pipe = build_pipe(fp16=fp16, use_lora=use_lora)
    print("generating...")
    raw = generate(pipe, a.prompt, negative=a.negative, steps=a.steps,
                   guidance=a.guidance, seed=a.seed, use_lora=use_lora)
    if a.raw:
        raw.save(a.raw)
        print(f"wrote raw render {a.raw}")

    sprite = pixelize(raw, size=a.size, colors=a.colors)
    _save_sprite(sprite, a.output, a.colors, a.preview)


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
    p_gen.set_defaults(func=_cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
