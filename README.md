# spriteforge

Prompt → palette-coherent, game-ready pixel-art sprites.

```
[prompt] → generate (SD 1.5 + pixel LoRA, GPU) → pixelize (CPU, deterministic) → serve
```

Full design: [`reference/HANDOVER.md`](reference/HANDOVER.md) · Build plan & status: [`PLAN.md`](PLAN.md)

## Install

```bash
# CPU core (pixelize, palettes) — runs anywhere
pip install -e .

# On a GPU box (RTX 3080 / Apple Silicon), add stage-1 generation:
pip install -e ".[generate]"

# Development
pip install -e ".[dev]"
pytest
```

## Usage

```bash
# Convert any image into clean pixel art (pure CPU)
spriteforge pixelize input.png -o out.png --size 256 --colors 16 --preview 4

# Prompt -> sprite (needs [generate]; first run downloads ~4 GB of model weights)
spriteforge generate "a brave knight in green armour, full body" -o knight.png --seed 7
```

Device selection is automatic (CUDA → MPS → CPU) with fp16 on CUDA and fp32 on
MPS; override with `--fp16` / `--fp32`.

## How it works

- **Generate** — Stable Diffusion 1.5 + a pixel-art LoRA (`PixArFK` trigger
  auto-appended), DPM++ scheduler, native 512×512 canvas.
- **Pixelize** — area-filter downscale → k-means palette quantisation **in OKLab
  space** (perceptually even, not muddy) → hard alpha threshold. Deterministic,
  seeded, no GPU.

The originals of the proven Phase-0 scripts live in [`reference/`](reference/);
the port is pinned to them by a bit-identical regression test.
