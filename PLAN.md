# spriteforge — Build Plan

> Engineering plan for the pixel-art generation & animation pipeline specified in
> [`reference/HANDOVER.md`](reference/HANDOVER.md). This document is the actionable
> distillation: what gets built, in what order, how it's tested, and which decisions
> are made vs. still open. The proven Phase-0 scripts being ported are preserved
> verbatim in [`reference/`](reference/).

---

## 1. What we're building

An installable Python package, **`spriteforge`**, that takes a text prompt to a
game-ready, palette-coherent, (eventually) animated pixel-art sprite:

```
[prompt] → generate (SD 1.5 + LoRA, GPU) → pixelize (CPU, deterministic) → serve
```

layered with **style LoRAs** (world glue), **character LoRAs** (identity), and a
**pose-control + brute-force-selector** animation stage (handover §12–13).

## 2. Environments — what runs where

| Where | What | Why |
|---|---|---|
| **This repo / CI / cloud dev container** | All code, plus full tests for everything CPU-side (pixelize, palettes, packing, dataset prep, scoring interfaces). GPU code gets mocked unit tests. | No GPU here; pixelize is pure CPU so it's fully verifiable in CI. |
| **Windows PC, RTX 3080** | Real generation, all LoRA training (kohya_ss), animation runs. | CUDA workhorse per handover §3. |
| **Apple Silicon Mac** | Dev driving + pixelize; MPS generation as fallback. | Handover §3. |

Every milestone ends with a **user-run GPU validation checkpoint** — a short,
scripted smoke test on the 3080 box. Code is not "done" until it passes there.

## 3. Package layout

```
spriteforge/
├── pyproject.toml            # deps: numpy, Pillow; extras: [generate] torch/diffusers,
│                             #       [curate] CLIP, [animate] controlnet extras
├── PLAN.md                   # this file
├── README.md
├── reference/                # frozen inputs: HANDOVER.md + original pixelize.py/generate.py
├── src/spriteforge/
│   ├── color.py              # sRGB/linear/OKLab conversions (extracted from pixelize.py)
│   ├── kmeans.py             # dependency-free k-means++ (extracted from pixelize.py)
│   ├── pixelize.py           # stage 2: downscale → quantise → crisp alpha → preview
│   ├── palette.py            # Palette type; JSON / .hex (Lospec) / PNG-swatch I/O; extract
│   ├── generate.py           # stage 1: SD1.5 + LoRA pipeline, device auto-detect
│   ├── refine.py             # img2img ratchet batches from a hero image        (M-C)
│   ├── curate.py             # CLIP identity + cleanliness scoring              (M-C)
│   ├── dataset.py            # kohya_ss dataset prep                            (M-C)
│   ├── animate/
│   │   ├── skeleton.py       # action keypoint sets → OpenPose conditioning imgs (M-D)
│   │   ├── selector.py       # brute-force frame selection                      (M-D)
│   │   └── sheet.py          # sprite-sheet packer + JSON metadata              (M-D)
│   └── cli.py                # `spriteforge` entry point, subcommand per module
└── tests/
```

Install profiles: `pip install spriteforge` gives the CPU core (pixelize/palette/sheet);
`spriteforge[generate]` adds torch+diffusers on the GPU boxes. Keeps the Mac/CI light.

## 4. Milestones

### Milestone A — scaffold + Phase-0 port (tasks 1–3)

Port the two proven scripts into the package **preserving behaviour exactly**:

- **`pixelize`** — BOX-filter downscale → OKLab k-means (k-means++, seeded,
  dependency-free) → hard alpha threshold → optional NN preview. Fully tested in CI:
  exact ≤N colour counts, alpha ∈ {0,255} only, determinism, odd sizes, and a
  regression test pinning output against the original script on a synthetic scene.
- **`generate`** — same model stack (SD 1.5 rehost checkpoint, PixArFK pixel LoRA +
  auto-appended trigger, DPM++ scheduler, 512×512, steps≈28 / guidance≈7, negative
  prompt). **One upgrade:** device auto-detect **CUDA → MPS → CPU** (the original is
  MPS-first; the 3080 is the primary box per §3), fp16 default on CUDA / fp32 on MPS,
  memory optimisations applied per-device. Mocked-pipeline unit tests for prompt
  assembly, seeding, and device pick.

**Done when:** CI green here; `spriteforge generate "a brave knight…"` produces a
sprite on the 3080 (Checkpoint A/B, task 7).

### Milestone B — cohesion: fixed palettes + CLI (tasks 4–6)

The handover's single highest-leverage upgrade (§15):

- **Fixed-palette mode** — `pixelize(img, palette=…)` maps every opaque pixel to the
  nearest palette colour by **OKLab distance**, replacing per-image k-means. Every
  asset snaps to the same colours → one coherent game.
- **Palette formats** — JSON (`{"name": …, "colors": ["#hex", …]}`), plain
  hex-per-line **`.hex` (Lospec format** — instant access to thousands of curated
  game palettes), and PNG swatch. Round-trip tested.
- **`palette extract`** — derive the game's locked palette from a set of favourite
  early renders: k-means in OKLab over their combined opaque pixels. This answers
  §19's "how is a fixed palette designed": *generate freely → pick favourites →
  extract palette → lock it for everything after.*
- **CLI** — `spriteforge generate | pixelize | palette extract|show`, keeping the
  original scripts' flag names where sensible.

**Done when:** on the 3080, a character, an enemy, and a building generated
separately with the same `--palette` visibly read as one game (Checkpoint A/B).

### Milestone C — LoRA tooling + the ratchet (tasks 8–10)

- **`dataset prep`** — curated images + trigger token → kohya_ss-ready folder tree
  (repeat-count dirs, per-image captions with the token baked in, 512px
  normalisation) + a documented kohya config template for SD 1.5 LoRA on 10–12 GB
  (xformers, fp16/bf16, per §9). Fully CI-testable.
- **`refine`** — the ratchet's step 2 (§10): batch img2img from `--hero` at medium
  denoise with angle/pose prompt variations; stacks a character LoRA for rounds ≥2.
- **`curate`** — §11: CLIP-embedding similarity to hero (identity) × clean-pixel-art
  check (quality); keep candidates passing both. Scorer interface designed to be
  reused by the Milestone-D selector.

**Done when:** user trains the style LoRA and completes character-ratchet rounds 1–2
on the 3080 (Checkpoint C, task 11) with drastically reduced hand-curation.

### Milestone D — animation (tasks 12–15)

- **Skeletons** — canonical keypoint sets (walk 8f / run 8f / jump 6f) stored as
  JSON, rendered to OpenPose-style conditioning images. *Decision:* hand-authored
  canonical sets first — deterministic and reusable across every character; extraction
  from reference video/sheets only if quality demands it later.
- **Per-frame generation** — SD 1.5 + ControlNet(openpose) + character LoRA + style
  LoRA, per §12's three-lever stack.
- **Brute-force selector (§13, the candidate edge)** — per frame: ~100 candidates,
  `score = w₁·pose_match + w₂·continuity(prev locked frame)`, lock winner, advance.
  *Proposed decision:* score continuity **in pixelized space** (per-pixel OKLab
  distance + changed-pixel count on the downscaled frames) — the pixelized frame is
  the shipped artefact, and wobble that vanishes in the downscale shouldn't cost a
  candidate its slot. LPIPS at raw resolution kept as a fallback metric.
- **Sheet packer** — fixed-cell grid, row per action, JSON sidecar (frame size,
  action→row, fps hint). CPU-only, CI-tested.
- **The pivotal experiment (task 15)** — same walk cycle via (a) our selector
  pipeline, (b) PixelLab's API, (c) optional hand-cleaned baseline; compare jitter
  metrics + eyeball. **This evidence decides build-vs-buy for animation** before any
  SaaS bet (§19's top open problem).

### Out of scope for now
LAN web UI (§7 — deprioritised by user), SDXL quality pass (§5 — revisit later),
seamless environment tiles (§15/§19 — after Milestone B proves cohesion),
all of SaaS/Phase 5 (§16–17 — gated on the task-15 experiment).

## 5. Test strategy

| Layer | Where verified | How |
|---|---|---|
| pixelize, palettes, sheet packer, dataset prep | **CI, fully** | Deterministic; synthetic images; exact assertions (colour counts, alpha, layout) |
| generate, refine, ControlNet path | CI (mocked) + **3080 (real)** | Unit tests mock the diffusers pipeline; scripted smoke tests per checkpoint |
| curate (CLIP) | CI (stub embedder) + 3080 (thresholds) | Threshold calibration needs real outputs |
| selector quality | **3080 experiment (task 15)** | vs PixelLab / hand baseline |

## 6. Decisions made in planning

1. **Name & shape:** `spriteforge`, installable src-layout package, single CLI. GPU
   deps behind an extra so the CPU core installs anywhere.
2. **Port, don't rewrite:** Phase-0 behaviour is proven; it moves into the package
   with a regression test, not a reimplementation.
3. **Device order CUDA → MPS → CPU** (original script was MPS-first).
4. **Palette formats include Lospec `.hex`** for free access to curated palettes.
5. **Locked palettes are *extracted* from curated early renders** (answers part of §19).
6. **Selector continuity scored in pixelized space** (proposed — cheap, artefact-true).
7. **Hand-authored canonical skeletons first** for walk/run/jump.
8. **Milestones gate on user-run GPU checkpoints** — nothing is "done" on mocks alone.
9. **Default sprite grid: 256px** (user decision, was 64). Consequences: 512→256 is
   only a 2× downscale, so the "small sprites hide wobble" effect (§12) largely
   disappears — the animation milestone leans harder on the selector — and the SDXL
   revisit (1024 native → 4× headroom) rises in priority. 16 colours may feel tight
   at 256px; judge with real outputs at Checkpoint A/B. Size stays a parameter.
10. **PixelLab comparison deferred** — no API account yet; revisit at Milestone D
   (task 15 needs it or a substitute baseline).

## 7. Open questions (carried + new)

- §19 carried: does the brute-force selector beat PixelLab / hand-cleanup? How much
  jitter survives LoRA + pose + selector? SDXL revisit. Tiling mode. Cold starts (SaaS).
- **Logical vs display resolution at 256:** does "256" mean a 256px logical grid
  (fine pixels), or a smaller logical grid (e.g. 64) displayed at 4× for the classic
  chunky look? The pipeline supports both (size parameter + NN preview upscale);
  worth settling when assets meet the game engine.
- **Selector weights** (w₁ pose vs w₂ continuity) and candidate count (100 assumed)
  — tune empirically in task 15.

## 8. Task index

| # | Task | Milestone |
|---|---|---|
| 1 | Scaffold spriteforge package | A |
| 2 | Port pixelize + full test suite | A |
| 3 | Port generate + device auto-detect | A |
| 4 | Fixed-palette mode | B |
| 5 | Palette formats + extract tool | B |
| 6 | CLI | B |
| 7 | **Checkpoint A/B** — user validates on 3080 | gate |
| 8 | kohya dataset prep | C |
| 9 | img2img ratchet refine | C |
| 10 | CLIP auto-curation | C |
| 11 | **Checkpoint C** — style LoRA + ratchet rounds on 3080 | gate |
| 12 | Pose skeletons + ControlNet generation | D |
| 13 | Brute-force frame selector | D |
| 14 | Sprite-sheet packer | D |
| 15 | **Selector experiment** vs PixelLab — build-vs-buy decision | D |
