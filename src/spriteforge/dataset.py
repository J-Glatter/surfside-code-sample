"""kohya_ss dataset preparation (handover §9).

Takes curated images + a trigger token and emits the folder layout kohya's
sd-scripts expect for SD 1.5 LoRA training on the 3080 box:

    out_dir/
    ├── img/{repeats}_{trigger}/   0001.png + 0001.txt (caption with trigger baked in)
    ├── kohya_config.toml          training config template (10-12 GB VRAM friendly)
    └── NOTES.md                   how to launch the run

Quality >> quantity: 15-30 great images beat 100 sloppy ones. Per-image caption
detail can be supplied as a sidecar `<image>.txt` next to each source image.
Pure CPU, fully testable.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

TRAIN_RESOLUTION = 512  # train at 512 — output is tiny sprites, no reason to go bigger

_KOHYA_CONFIG = """\
# kohya_ss (sd-scripts train_network.py) template — SD 1.5 LoRA on 10-12 GB VRAM.
# Speed levers per the project handover: 512px, xformers, fp16, modest steps.
# Verify paths before launching.

pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
train_data_dir = "{train_data_dir}"
output_dir = "{output_dir}"
output_name = "{name}"

resolution = "512,512"
enable_bucket = true

network_module = "networks.lora"
network_dim = 16
network_alpha = 8

learning_rate = 1e-4
text_encoder_lr = 5e-5
optimizer_type = "AdamW8bit"
lr_scheduler = "cosine"

train_batch_size = 2
max_train_steps = {max_steps}

mixed_precision = "fp16"
xformers = true
cache_latents = true
gradient_checkpointing = true

save_model_as = "safetensors"
save_every_n_steps = 400
"""

_NOTES = """\
# Training run: {name}

Dataset: {n_images} images x {repeats} repeats, trigger token: `{trigger}`

On the 3080 box (kohya_ss checkout, venv active):

    accelerate launch train_network.py --config_file {config_path}

Expected wall clock: ~30-45 min. Result: {name}.safetensors in `{output_dir}` —
a few MB, stackable with the pixel-art style LoRA.

Use it: prompt with `{trigger}` and load the file via
`pipe.load_lora_weights("{output_dir}/{name}.safetensors")`.
"""


def prep_dataset(
    images: list[str | Path],
    out_dir: str | Path,
    trigger: str,
    repeats: int = 10,
    class_word: str | None = None,
    resolution: int = TRAIN_RESOLUTION,
    name: str | None = None,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """Build a kohya-ready dataset directory. Returns the img/ train_data_dir.

    Caption per image: "<trigger>[, <class_word>][, <sidecar text>]" where the
    sidecar is an optional `<image>.txt` next to the source file.
    """
    out_dir = Path(out_dir)
    name = name or trigger
    train_root = out_dir / "img"
    dest = train_root / f"{repeats}_{trigger}"
    dest.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, src in enumerate(sorted(Path(p) for p in images), start=1):
        img = Image.open(src).convert("RGBA")
        # cap the longest side at the training resolution
        if max(img.size) > resolution:
            scale = resolution / max(img.size)
            img = img.resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                Image.LANCZOS,
            )
        # SD trains on RGB: composite any transparency over a flat background
        rgb = Image.new("RGB", img.size, background)
        rgb.paste(img, mask=img.getchannel("A"))

        stem = f"{i:04d}"
        rgb.save(dest / f"{stem}.png")

        caption = [trigger]
        if class_word:
            caption.append(class_word)
        sidecar = src.with_suffix(".txt")
        if sidecar.exists():
            extra = sidecar.read_text().strip()
            if extra:
                caption.append(extra)
        (dest / f"{stem}.txt").write_text(", ".join(caption) + "\n")
        count += 1

    if count == 0:
        raise ValueError("no images provided")

    # ~100 steps per image x repeats/10, clamped to a sane band (handover: don't
    # over-crank step count)
    max_steps = min(3000, max(800, count * repeats * 10))
    (out_dir / "kohya_config.toml").write_text(_KOHYA_CONFIG.format(
        train_data_dir=train_root.resolve(),
        output_dir=(out_dir / "output").resolve(),
        name=name,
        max_steps=max_steps,
    ))
    (out_dir / "NOTES.md").write_text(_NOTES.format(
        name=name,
        n_images=count,
        repeats=repeats,
        trigger=trigger,
        config_path=(out_dir / "kohya_config.toml").resolve(),
        output_dir=(out_dir / "output").resolve(),
    ))
    (out_dir / "output").mkdir(exist_ok=True)
    return train_root
