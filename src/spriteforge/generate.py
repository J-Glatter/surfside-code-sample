"""Stage 1 — text prompt -> raw render via a diffusion backend + pixel LoRA.

Two backends (Checkpoint A/B bake-off finding — SDXL wins decisively):

  * sdxl (default): SDXL base + nerijs/pixel-art-xl. Coherent single subjects,
    real pixel grid, far less baked-in scenery than SD1.5. The A40/3080 run it
    comfortably. Native 1024px (downscaled to the 64px grid afterwards).
  * sd15: the original SD1.5 + PixArFK LoRA, kept reachable behind --sd15 —
    lighter, faster, and the only one with a proven quadruped openpose
    ControlNet today.

torch/diffusers are imported lazily so the CPU core of the package installs and
runs without the `[generate]` extra. First real run downloads the model (SDXL
~7 GB, SD1.5 ~4 GB) into HF_HOME.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    """A generation backend: base model, its pixel LoRA, and matching ControlNets."""

    name: str
    base_model: str
    pixel_lora: str
    lora_weight_name: str | None   # explicit safetensors file, or None to autodetect
    lora_trigger: str              # tokens appended to every prompt
    size: int                      # native generation resolution (square)
    is_xl: bool                    # SDXL pipelines differ from SD1.5
    controlnet_openpose: str
    controlnet_animal: str | None  # quadruped rig; None where none is proven yet
    train_script: str              # kohya sd-scripts entrypoint for this base


SDXL = Backend(
    name="sdxl",
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    pixel_lora="nerijs/pixel-art-xl",
    lora_weight_name="pixel-art-xl.safetensors",
    lora_trigger="pixel art",
    size=1024,
    is_xl=True,
    controlnet_openpose="xinsir/controlnet-openpose-sdxl-1.0",
    controlnet_animal=None,        # no proven SDXL animal-openpose yet — see frames.py
    train_script="sdxl_train_network.py",
)

SD15 = Backend(
    name="sd15",
    # The old runwayml/stable-diffusion-v1-5 repo is deprecated; this is the rehost.
    base_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
    pixel_lora="artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
    lora_weight_name=None,
    lora_trigger="pixel art, PixArFK",
    size=512,
    is_xl=False,
    controlnet_openpose="lllyasviel/sd-controlnet-openpose",
    controlnet_animal="crishhh/animal_openpose",
    train_script="train_network.py",
)

BACKENDS = {SDXL.name: SDXL, SD15.name: SD15}
DEFAULT_BACKEND = "sdxl"


def get_backend(backend: str | Backend | None) -> Backend:
    """Resolve a backend name (or pass one through); None -> the default (SDXL)."""
    if isinstance(backend, Backend):
        return backend
    if backend is None:
        return BACKENDS[DEFAULT_BACKEND]
    try:
        return BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r} (choose from {sorted(BACKENDS)})"
        ) from None


# Back-compat module aliases resolve to the default backend. Prefer get_backend().
BASE_MODEL = BACKENDS[DEFAULT_BACKEND].base_model
PIXEL_LORA = BACKENDS[DEFAULT_BACKEND].pixel_lora
LORA_TRIGGER = BACKENDS[DEFAULT_BACKEND].lora_trigger

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


def build_prompt(prompt: str, use_lora: bool = True,
                 backend: str | Backend | None = None,
                 trigger: str | None = None) -> str:
    """Prepend the active backend's LoRA trigger tokens when the pixel LoRA is on.

    Trigger FIRST, not last: CLIP truncates at 77 tokens, so a trailing trigger
    is the first thing dropped on a long art-directed prompt (field-tested — the
    style token was silently falling off). Leading with it guarantees it lands.
    """
    # `trigger` overrides the backend default (a custom style LoRA has its own
    # trigger token); trigger="" suppresses it entirely.
    tok = get_backend(backend).lora_trigger if trigger is None else trigger
    return f"{tok}, {prompt}" if use_lora and tok else prompt


def _fix_sdxl_vae(pipe, be: Backend, fp16: bool) -> None:
    """SDXL's VAE overflows in fp16 and can emit black/NaN images — upcast it to
    fp32. Cheap (a few hundred MB VRAM), no download, and it also silences the
    'AutoencoderKL should be kept in float32' warning. A black sprite would
    quietly poison a LoRA training set, so this is worth doing everywhere."""
    if be.is_xl and fp16 and hasattr(pipe, "upcast_vae"):
        pipe.upcast_vae()


def _load_pixel_lora(pipe, be: Backend) -> None:
    """Load a backend's pixel LoRA, tolerating a shifted weight filename."""
    try:
        if be.lora_weight_name:
            pipe.load_lora_weights(be.pixel_lora, weight_name=be.lora_weight_name)
        else:
            pipe.load_lora_weights(be.pixel_lora)
        print(f"loaded LoRA: {be.pixel_lora}")
    except Exception as e:  # noqa: BLE001
        print(f"couldn't load LoRA ({e}). Continuing base-model only.\n"
              f"If it's a filename issue, check weight_name for {be.pixel_lora}.")


def build_pipe(fp16: bool | None = None, use_lora: bool = True,
               device: str | None = None, backend: str | Backend | None = None,
               style_lora: str | None = None):
    """Assemble the pipeline for the chosen backend on the best available device.

    fp16=None means auto: fp16 on CUDA, fp32 elsewhere.
    `style_lora` (a local .safetensors path or Hub id) replaces the stock pixel
    LoRA — this is how a LoRA you trained at Checkpoint C plugs in. Pair it with
    generate(trigger=...) so the prompt carries your training token.
    """
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
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            be.base_model, torch_dtype=dtype)
    else:
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            be.base_model, torch_dtype=dtype, safety_checker=None)
    # DPM++ gives good results in few steps.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device != "cuda":
        pipe.enable_attention_slicing()  # lower peak memory on unified RAM / CPU
    _fix_sdxl_vae(pipe, be, fp16)

    if use_lora:
        if style_lora:
            try:
                pipe.load_lora_weights(style_lora)
                print(f"loaded style LoRA: {style_lora}")
            except Exception as e:  # noqa: BLE001
                print(f"couldn't load style LoRA {style_lora} ({e}). Base model only.")
        else:
            _load_pixel_lora(pipe, be)
    return pipe


def generate(
    pipe,
    prompt: str,
    negative: str = DEFAULT_NEGATIVE,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    seed: int | None = None,
    use_lora: bool = True,
    backend: str | Backend | None = None,
    size: int | None = None,
    trigger: str | None = None,
):
    """Run the pipeline once; returns the raw PIL image (pre-pixelize).

    `size` defaults to the backend's native resolution (SDXL 1024, SD1.5 512).
    `trigger` overrides the prepended style token (for a custom style LoRA).
    """
    import torch

    be = get_backend(backend)
    full_prompt = build_prompt(prompt, use_lora, backend=be, trigger=trigger)
    dim = size or be.size
    generator = None
    if seed is not None:
        # CPU generator keeps seeds reproducible across devices.
        generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=dim,
        height=dim,
        generator=generator,
    ).images[0]
    return image
