#!/usr/bin/env python3
"""Throwaway SDXL bake-off — is SDXL + a real pixel LoRA better than SD1.5?

Generates each prompt with SDXL base + nerijs/pixel-art-xl, then runs our own
pixelize + isolate so the comparison is apples-to-apples with the spriteforge
output. Nothing here touches the pipeline; if SDXL wins we wire it in properly.

    python scripts/sdxl_bakeoff.py -o out/sdxl
    python scripts/sdxl_bakeoff.py -o out/sdxl --prompt "a small slime monster" --n 6

Needs the [generate,isolate] extras (already installed on the pod). First run
downloads ~7 GB of SDXL weights to HF_HOME.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

from spriteforge.isolate import isolate_subject
from spriteforge.pixelize import pixelize, upscale_preview

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
PIXEL_LORA = "nerijs/pixel-art-xl"           # trigger: "pixel art"
LORA_FILE = "pixel-art-xl.safetensors"

# The director's art-direction, minus the "pixel art" style words the LoRA owns.
DEFAULT_PROMPTS = {
    "slime": ("pixel art, a small round gelatinous slime monster, glossy "
              "translucent lime-green body, two big round eyes, a wide "
              "cheerful grin, tiny stubby arms, cute chibi video-game creature, "
              "single subject, one creature only, full body, centered, floating "
              "on a plain solid white background, no shadow, no ground"),
    "knight": ("pixel art, a single brave knight in green armour, sword and "
               "shield, full body, centered, floating on a plain solid white "
               "background, no shadow, no ground"),
}
NEGATIVE = ("multiple creatures, crowd, collage, blurry, noisy, deformed, "
            "border, frame, busy background, scenery, pedestal, platform, "
            "shadow, ground, text, watermark")


def build_pipe() -> StableDiffusionXLPipeline:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(SDXL_BASE, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.load_lora_weights(PIXEL_LORA, weight_name=LORA_FILE)
    except Exception:  # noqa: BLE001 — filename varies; let diffusers autodetect
        pipe.load_lora_weights(PIXEL_LORA)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="out/sdxl")
    ap.add_argument("--prompt", default=None,
                    help="one custom prompt (default: built-in slime + knight)")
    ap.add_argument("--n", type=int, default=4, help="candidates per prompt")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--colors", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prompts = ({"custom": f"pixel art, {args.prompt}"} if args.prompt
               else DEFAULT_PROMPTS)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading SDXL + {PIXEL_LORA} (first run downloads ~7 GB)...")
    pipe = build_pipe()

    for name, prompt in prompts.items():
        print(f"\n== {name} ==\n{prompt}")
        for i in range(args.n):
            gen = torch.Generator(device=pipe.device).manual_seed(args.seed + i)
            raw = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=args.steps,
                       guidance_scale=args.guidance, generator=gen).images[0]
            raw.save(out / f"{name}_{i:02d}_raw.png")
            subject, method = isolate_subject(raw)
            sprite = pixelize(subject, size=args.size, colors=args.colors)
            sprite.save(out / f"{name}_{i:02d}.png")
            upscale_preview(sprite, 4).save(out / f"{name}_{i:02d}_x4.png")
            print(f"  {name}_{i:02d}: isolate={method}")
    print(f"\ndone -> {out}  (compare *_x4.png against the SD1.5 slime_v3)")


if __name__ == "__main__":
    main()
