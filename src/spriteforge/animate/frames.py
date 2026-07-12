"""Per-frame candidate generation: SD 1.5 + ControlNet(openpose) + LoRAs (GPU).

The three-lever stack from handover §12: pose locked by the skeleton
conditioning, identity locked by the character LoRA, style locked by the pixel
LoRA — then the selector (selector.py) picks for motion smoothness.
"""

from __future__ import annotations

from PIL import Image

from ..generate import (
    BASE_MODEL,
    DEFAULT_NEGATIVE,
    PIXEL_LORA,
    build_prompt,
    default_fp16,
    pick_device,
)

CONTROLNET_OPENPOSE = "lllyasviel/sd-controlnet-openpose"


def build_animation_pipe(
    character_lora: str | None = None,
    fp16: bool | None = None,
    use_style_lora: bool = True,
    device: str | None = None,
):
    """SD 1.5 + openpose ControlNet, with style/character LoRAs stacked."""
    import torch
    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetPipeline,
    )

    device = device or pick_device()
    if device == "cpu":
        print("WARNING: no GPU available — falling back to CPU (very slow).")
    if fp16 is None:
        fp16 = default_fp16(device)
    dtype = torch.float16 if fp16 else torch.float32

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_OPENPOSE, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device != "cuda":
        pipe.enable_attention_slicing()

    adapters = []
    if use_style_lora:
        try:
            pipe.load_lora_weights(PIXEL_LORA, adapter_name="pixel")
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


def generate_candidates(
    pipe,
    control_image: Image.Image,
    prompt: str,
    n: int = 100,
    steps: int = 28,
    guidance: float = 7.0,
    controlnet_scale: float = 1.0,
    negative: str = DEFAULT_NEGATIVE,
    base_seed: int = 0,
    use_lora: bool = True,
) -> list[Image.Image]:
    """n candidates for one animation frame, deterministically seeded."""
    import torch

    full_prompt = build_prompt(prompt, use_lora)
    out = []
    for j in range(n):
        gen = torch.Generator(device="cpu").manual_seed(base_seed + j)
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative,
            image=control_image,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=controlnet_scale,
            generator=gen,
        ).images[0]
        out.append(image)
    return out
