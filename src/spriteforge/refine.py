"""The ratchet's refine step (handover §10): hero image -> consistent multi-angle set.

img2img from the one good "hero" picture at medium denoise strength, prompting
for different angles/poses. Leaning on the anchor keeps outputs roughly on-model.
Generate dozens, then curate (spriteforge.curate) down to the 8-10 best for LoRA
training. On ratchet rounds >= 2, stack the previously-trained character LoRA to
tighten identity further.

GPU module: torch/diffusers imported lazily, mirrors spriteforge.generate.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .generate import (
    DEFAULT_NEGATIVE,
    build_prompt,
    default_fp16,
    get_backend,
    pick_device,
)

# Angle/pose sweep for building a character training set. Front/back/side views
# are what the LoRA most needs to learn a character in the round.
DEFAULT_VARIATIONS = [
    "front view, standing",
    "back view, standing",
    "side view, facing left",
    "side view, facing right",
    "three-quarter view",
    "walking",
    "running",
    "arms raised, celebrating",
]

# medium denoise: enough freedom to re-pose, anchored enough to stay on-model
DEFAULT_STRENGTH = 0.5


def build_img2img_pipe(
    fp16: bool | None = None,
    use_lora: bool = True,
    device: str | None = None,
    character_lora: str | None = None,
    backend=None,
):
    """img2img pipeline for the chosen backend; optionally stacks a character
    LoRA on the style LoRA."""
    import torch
    from diffusers import DPMSolverMultistepScheduler

    be = get_backend(backend)
    device = device or pick_device()
    if device == "cpu":
        print("WARNING: no GPU available — falling back to CPU (very slow).")
    if fp16 is None:
        fp16 = default_fp16(device)

    dtype = torch.float16 if fp16 else torch.float32
    if be.is_xl:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            be.base_model, torch_dtype=dtype)
    else:
        from diffusers import StableDiffusionImg2ImgPipeline

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            be.base_model, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device != "cuda":
        pipe.enable_attention_slicing()
    from .generate import _fix_sdxl_vae

    _fix_sdxl_vae(pipe, be, fp16)

    adapters = []
    if use_lora:
        try:
            if be.lora_weight_name:
                pipe.load_lora_weights(be.pixel_lora, weight_name=be.lora_weight_name,
                                       adapter_name="pixel")
            else:
                pipe.load_lora_weights(be.pixel_lora, adapter_name="pixel")
            adapters.append("pixel")
        except Exception as e:  # noqa: BLE001
            print(f"couldn't load style LoRA ({e}). Continuing without it.")
    if character_lora:
        try:
            pipe.load_lora_weights(character_lora, adapter_name="character")
            adapters.append("character")
        except Exception as e:  # noqa: BLE001
            print(f"couldn't load character LoRA ({e}). Continuing without it.")
    if len(adapters) > 1:
        pipe.set_adapters(adapters, [1.0] * len(adapters))
    return pipe


def fit_canvas(img: Image.Image, canvas: int = 512,
               background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Fit an image onto a square RGB canvas (SD 1.5 native), preserving aspect."""
    img = img.convert("RGBA")
    scale = canvas / max(img.size)
    img = img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.LANCZOS,
    )
    out = Image.new("RGB", (canvas, canvas), background)
    out.paste(img, ((canvas - img.width) // 2, (canvas - img.height) // 2),
              mask=img.getchannel("A"))
    return out


def refine(
    pipe,
    hero: Image.Image,
    base_prompt: str,
    out_dir: str | Path,
    variations: list[str] | None = None,
    per_variation: int = 6,
    strength: float = DEFAULT_STRENGTH,
    steps: int = 28,
    guidance: float = 7.0,
    negative: str = DEFAULT_NEGATIVE,
    seed: int = 0,
    use_lora: bool = True,
    backend=None,
) -> list[Path]:
    """Batch img2img from the hero. Deterministic filenames and seeds."""
    import torch

    be = get_backend(backend)
    variations = variations if variations is not None else DEFAULT_VARIATIONS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = fit_canvas(hero, canvas=be.size)

    written = []
    for i, variation in enumerate(variations):
        prompt = build_prompt(f"{base_prompt}, {variation}", use_lora, backend=be)
        for j in range(per_variation):
            gen = torch.Generator(device="cpu").manual_seed(seed + i * 1000 + j)
            image = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=anchor,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
            ).images[0]
            path = out_dir / f"{i:02d}_{_slug(variation)}_{j:02d}.png"
            image.save(path)
            written.append(path)
    return written


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
