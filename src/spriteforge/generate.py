"""Stage 1 — text prompt -> raw 512px render via Stable Diffusion 1.5 + pixel LoRA.

Ported from reference/generate.py (proven Phase-0 code) with one upgrade per
PLAN.md §6: device auto-detect is now CUDA -> MPS -> CPU (the RTX 3080 box is the
generation workhorse; the original was MPS-first for the Mac), with fp16 the
default on CUDA and fp32 the safe default on MPS.

torch/diffusers are imported lazily so the CPU core of the package installs and
runs without the `[generate]` extra. First real run downloads the model (~4 GB)
into ~/.cache/huggingface.
"""

from __future__ import annotations

# ---- Model choices that resolve on Hugging Face today -----------------------
# The old runwayml/stable-diffusion-v1-5 repo is deprecated; this is the rehost.
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# SD 1.5 pixel-art LoRA. Trigger tokens: "pixel art, PixArFK".
PIXEL_LORA = "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5"
LORA_TRIGGER = "pixel art, PixArFK"

DEFAULT_NEGATIVE = "3d render, realistic, photo, blurry, jpeg artifacts, smooth shading"
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 7.0


def pick_device() -> str:
    """CUDA -> MPS -> CPU, by strength (handover §3)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_fp16(device: str) -> bool:
    """fp16 by default on CUDA; fp32 is the safe default on MPS (and CPU)."""
    return device == "cuda"


def build_prompt(prompt: str, use_lora: bool = True) -> str:
    """Append the LoRA trigger tokens when the pixel LoRA is active."""
    return f"{prompt}, {LORA_TRIGGER}" if use_lora else prompt


def build_pipe(fp16: bool | None = None, use_lora: bool = True, device: str | None = None):
    """Assemble the SD 1.5 pipeline on the best available device.

    fp16=None means auto: fp16 on CUDA, fp32 elsewhere.
    """
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device = device or pick_device()
    if device == "cpu":
        print("WARNING: no GPU available — falling back to CPU (very slow).")
    if fp16 is None:
        fp16 = default_fp16(device)

    dtype = torch.float16 if fp16 else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, safety_checker=None
    )
    # DPM++ gives good results in few steps.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device != "cuda":
        pipe.enable_attention_slicing()  # lower peak memory on unified RAM / CPU

    if use_lora:
        try:
            pipe.load_lora_weights(PIXEL_LORA)
            print(f"loaded LoRA: {PIXEL_LORA}")
        except Exception as e:  # noqa: BLE001
            print(f"couldn't load LoRA ({e}). Continuing base-model only.\n"
                  f"If it's a filename issue, pass weight_name= to load_lora_weights.")
    return pipe


def generate(
    pipe,
    prompt: str,
    negative: str = DEFAULT_NEGATIVE,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    seed: int | None = None,
    use_lora: bool = True,
):
    """Run the pipeline once; returns the raw 512px PIL image (pre-pixelize)."""
    import torch

    full_prompt = build_prompt(prompt, use_lora)
    generator = None
    if seed is not None:
        # CPU generator keeps seeds reproducible across devices.
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
