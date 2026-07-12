"""
generate.py — text prompt -> pixel-art sprite, on an Apple Silicon Mac.

Pipeline: Stable Diffusion 1.5 (+ pixel-art LoRA) on the MPS backend,
then pixelize.py cleans the output into a real palette-limited sprite.

First run downloads the model (~4 GB) and caches it under ~/.cache/huggingface.

    python generate.py "a brave knight in green armour, full body" -o knight.png

Notes for Apple Silicon:
  * Runs on the "mps" device. Keep the diffusion canvas at 512x512 — SD 1.5's
    native size — and let the pixelizer do the shrinking.
  * float32 is the safe default on MPS. Try --fp16 for ~1.5x speed once it works.
"""

from __future__ import annotations
import argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from pixelize import pixelize, upscale_preview

# ---- Model choices that resolve on Hugging Face today -----------------------
# The old runwayml/stable-diffusion-v1-5 repo is deprecated; this is the rehost.
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# SD 1.5 pixel-art LoRA. Trigger tokens: "pixel art, PixArFK".
PIXEL_LORA = "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5"
LORA_TRIGGER = "pixel art, PixArFK"

DEFAULT_NEGATIVE = "3d render, realistic, photo, blurry, jpeg artifacts, smooth shading"


def build_pipe(fp16: bool = False, use_lora: bool = True) -> StableDiffusionPipeline:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: MPS not available — falling back to CPU (very slow).")

    dtype = torch.float16 if fp16 else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, safety_checker=None
    )
    # DPM++ gives good results in few steps.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # lower peak memory on unified RAM

    if use_lora:
        try:
            pipe.load_lora_weights(PIXEL_LORA)
            print(f"loaded LoRA: {PIXEL_LORA}")
        except Exception as e:  # noqa: BLE001
            print(f"couldn't load LoRA ({e}). Continuing base-model only.\n"
                  f"If it's a filename issue, pass weight_name= to load_lora_weights.")
    return pipe


def generate(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative: str = DEFAULT_NEGATIVE,
    steps: int = 28,
    guidance: float = 7.0,
    seed: int | None = None,
    use_lora: bool = True,
):
    full_prompt = f"{prompt}, {LORA_TRIGGER}" if use_lora else prompt
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=512,
        height=512,
        generator=generator,
    ).images[0]
    return image


def main():
    p = argparse.ArgumentParser(description="Generate a pixel-art sprite from a prompt.")
    p.add_argument("prompt")
    p.add_argument("-o", "--output", default="sprite.png")
    p.add_argument("--size", type=int, default=64, help="final sprite size (longest side)")
    p.add_argument("--colors", type=int, default=16, help="palette size")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--negative", default=DEFAULT_NEGATIVE)
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--fp16", action="store_true", help="try float16 on MPS (faster)")
    p.add_argument("--preview", type=int, default=8, help="Nx nearest-neighbour preview")
    p.add_argument("--raw", help="also save the pre-pixelized 512px render here")
    a = p.parse_args()

    use_lora = not a.no_lora
    pipe = build_pipe(fp16=a.fp16, use_lora=use_lora)
    print("generating...")
    raw = generate(pipe, a.prompt, negative=a.negative, steps=a.steps,
                   guidance=a.guidance, seed=a.seed, use_lora=use_lora)
    if a.raw:
        raw.save(a.raw)
        print(f"wrote raw render {a.raw}")

    sprite = pixelize(raw, size=a.size, colors=a.colors)
    sprite.save(a.output)
    print(f"wrote {a.output} ({sprite.width}x{sprite.height}, <= {a.colors} colours)")
    if a.preview > 0:
        prev = a.output.rsplit(".", 1)[0] + "_preview.png"
        upscale_preview(sprite, a.preview).save(prev)
        print(f"wrote {prev}")


if __name__ == "__main__":
    main()
