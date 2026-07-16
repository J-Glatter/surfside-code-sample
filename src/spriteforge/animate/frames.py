"""Per-frame candidate generation: base model + ControlNet(openpose) + LoRAs (GPU).

The three-lever stack from handover §12: pose locked by the skeleton
conditioning, identity locked by the character LoRA, style locked by the pixel
LoRA — then the selector (selector.py) picks for motion smoothness.

Backend-aware (generate.Backend): SDXL uses the XL ControlNet pipeline +
xinsir openpose; SD1.5 uses lllyasviel openpose and is the only backend with a
proven quadruped rig today.
"""

from __future__ import annotations

from PIL import Image

from ..generate import (
    DEFAULT_NEGATIVE,
    build_prompt,
    default_fp16,
    get_backend,
    pick_device,
)


def controlnet_for(body: str, backend=None) -> str:
    """The openpose ControlNet repo id for a body type on the chosen backend.

    Raises for a quadruped on a backend with no proven animal-openpose model
    (SDXL today) — the caller should pass --controlnet or fall back to --sd15
    rather than silently generating with the wrong conditioning.
    """
    be = get_backend(backend)
    if body == "quadruped":
        if be.controlnet_animal is None:
            raise ValueError(
                f"no proven quadruped openpose ControlNet for the {be.name} "
                f"backend yet — pass --controlnet <repo> or use --sd15 for "
                f"four-legged animation (verified at Checkpoint D).")
        return be.controlnet_animal
    return be.controlnet_openpose


# Back-compat: the SD1.5 constants some call sites and tests still reference.
CONTROLNET_OPENPOSE = "lllyasviel/sd-controlnet-openpose"
CONTROLNET_ANIMAL_OPENPOSE = "crishhh/animal_openpose"
CONTROLNET_BY_BODY = {
    "humanoid": CONTROLNET_OPENPOSE,
    "quadruped": CONTROLNET_ANIMAL_OPENPOSE,
}


def build_animation_pipe(
    character_lora: str | None = None,
    fp16: bool | None = None,
    use_style_lora: bool = True,
    device: str | None = None,
    controlnet_model: str | None = None,
    backend=None,
    body: str = "humanoid",
):
    """Base model + a pose ControlNet, with style/character LoRAs stacked.

    `controlnet_model` overrides the backend/body default when given.
    """
    import torch
    from diffusers import ControlNetModel, DPMSolverMultistepScheduler

    be = get_backend(backend)
    device = device or pick_device()
    if device == "cpu":
        print("WARNING: no GPU available — falling back to CPU (very slow).")
    if fp16 is None:
        fp16 = default_fp16(device)
    dtype = torch.float16 if fp16 else torch.float32

    controlnet_model = controlnet_model or controlnet_for(body, be)
    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=dtype)

    if be.is_xl:
        from diffusers import StableDiffusionXLControlNetPipeline

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            be.base_model, controlnet=controlnet, torch_dtype=dtype)
    else:
        from diffusers import StableDiffusionControlNetPipeline

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            be.base_model, controlnet=controlnet, torch_dtype=dtype,
            safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device != "cuda":
        pipe.enable_attention_slicing()

    adapters = []
    if use_style_lora:
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
    backend=None,
) -> list[Image.Image]:
    """n candidates for one animation frame, deterministically seeded."""
    import torch

    full_prompt = build_prompt(prompt, use_lora, backend=backend)
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
