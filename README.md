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
# Stage 0 — the LLM director routes a request to the right workstream
# ([director] extra + ANTHROPIC_API_KEY; falls back to keyword heuristics without)
spriteforge plan "a small slime monster"          # inspect the routing decision
spriteforge make "a small slime monster" -o slime --palette game.json
#   -> sprite + procedural bounce/idle frames + GIFs + sprite sheet, one command
#   (tiles get --tile + seam check; limbed characters get hero candidates +
#    ratchet next-steps, since LoRA training runs in kohya)

# Convert any image into clean pixel art (pure CPU)
spriteforge pixelize input.png -o out.png --size 64 --colors 16 --preview 4

# Prompt -> sprite (needs [generate]; first run downloads ~4 GB of model weights)
spriteforge generate "a brave knight in green armour, full body" -o knight.png --seed 7

# World cohesion: lock one palette for every asset (handover §15)
spriteforge palette extract best1.png best2.png best3.png -o game.json
spriteforge generate "a small slime monster" -o slime.png --palette game.json

# Character pipeline: hero -> variations -> auto-curate -> kohya LoRA dataset
spriteforge refine hero.png --prompt "a brave knight" -o candidates
spriteforge curate candidates --hero hero.png -o keep --keep 10    # [curate] extra
spriteforge dataset prep keep/*.png -o knight_ds --trigger sks_knight

# Environment tiles that wrap edge-to-edge
spriteforge generate "grassy meadow ground texture, top-down" -o grass.png --tile --palette game.json

# Animation: pose-controlled candidates + continuity-scored selection ([animate] extra)
spriteforge skeleton --action walk -o skel_preview                 # eyeball the cycle (CPU)
spriteforge animate "a brave knight, sks_knight" --action walk -o frames/walk \
    --character-lora knight_ds/output/sks_knight.safetensors --palette game.json
spriteforge preview frames/walk -o walk.gif --fps 10 --scale 4     # judge it at game speed
spriteforge sheet walk=frames/walk -o knight_sheet.png             # + JSON sidecar
```

```bash
# Remote-trigger mode: the GPU box watches a shared folder for job files
spriteforge worker C:\sprite-jobs --palette game.json
#   drop wolf.json (a plan) or cobble.txt (a prompt) in; collect done/<name>/
```

Device selection is automatic (CUDA → MPS → CPU) with fp16 on CUDA and fp32 on
MPS; override with `--fp16` / `--fp32`. GPU validation runbook:
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md) · Windows box as a wake-on-demand
render worker (WoL, SSH, jobs share, autostart):
[`docs/WINDOWS_SETUP.md`](docs/WINDOWS_SETUP.md).

## How it works

- **Generate** — Stable Diffusion 1.5 + a pixel-art LoRA (`PixArFK` trigger
  auto-appended), DPM++ scheduler, native 512×512 canvas.
- **Pixelize** — area-filter downscale → k-means palette quantisation **in OKLab
  space** (perceptually even, not muddy) → hard alpha threshold. Deterministic,
  seeded, no GPU.

The originals of the proven Phase-0 scripts live in [`reference/`](reference/);
the port is pinned to them by a bit-identical regression test.
