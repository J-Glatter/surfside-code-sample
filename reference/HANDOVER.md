# Pixel Art Generation & Animation Pipeline — Engineering Handover

> **For the next builder (human or Claude instance).** This is the full design for a
> PixelLab-equivalent pipeline: prompt → refined character → trained model → animated,
> palette-coherent, game-ready sprites. It starts as a local tool and has a clear path
> to a SaaS. Working code for the first stage already exists (see `pixelize.py`,
> `generate.py`). Everything else here is specified to be built.
>
> **How to read this:** Sections 1–7 are the working foundation. 8–15 are the design
> for the full system. 16–17 cover the SaaS/economics angle. 18 is the build order.
> 19 lists what's still undecided. Where a decision was made, it's stated as a decision;
> where something is speculative, it's flagged.

---

## 1. Goal & scope

Build a tool that turns a text prompt into game-ready pixel-art assets for a top-down RPG:
characters (with walk/run/jump animations), enemies, and environment/buildings — all
sharing one coherent art style and palette.

Two horizons:

- **Near term (personal tool):** generate + refine + animate one character for the game.
- **Longer term (SaaS):** a customer types a prompt and, within a few minutes, gets a
  fully animated, sprite-sheeted character in a consistent world style.

**Design stance we settled on:** don't rebuild the whole of PixelLab. The *generation*
half is assemble-able from open-source parts (and is largely done). The *animation* half
is the genuinely hard, unsolved-at-quality part. We have a candidate approach for it
(Section 13) that may be a real edge, but the fallback remains: use PixelLab's API for the
animation step if ours underperforms.

---

## 2. Current state — what already exists

In this repo:

- **`pixelize.py`** — the post-processor. Downscale (area filter) → OKLab k-means palette
  quantisation → crisp alpha. Pure CPU, no GPU, runs anywhere. **Tested and verified**
  (produces exact N-colour output). This is the deterministic core of the whole system.
- **`generate.py`** — Stable Diffusion 1.5 (+ pixel LoRA) on Apple-Silicon MPS, piped
  through `pixelize`. Runnable; needs a first-run model download.
- **`README.md`**, **`requirements.txt`** — setup + usage.
- **`demo_*.png`** — proof images showing the pixelizer converting a 512px gradient scene
  to a clean 64×64 / 16-colour sprite.

Everything below extends this.

---

## 3. Hardware & where things run

The user has two machines. Use them by strength:

| Machine | Role |
|---|---|
| **Windows PC w/ RTX 3080 (10–12 GB)** | **Generation + training workhorse.** CUDA — all the training tooling loves it. Faster generation than the Mac. Train LoRAs here. |
| **Apple Silicon Mac** | **Driver / dev machine.** Runs `pixelize.py` fine (CPU). Can generate on MPS but slower — prefer the 3080. |

**Decisions:**
- Training and stage-1 generation → the 3080 box.
- `pixelize.py` (stage 2) → anywhere, it's CPU-only.
- Eventually put a thin web/API layer on the Windows box so the Mac (or a browser) can
  fire prompts at it over the LAN. (Not built; low priority per user.)
- The 3080 is **sufficient for SD 1.5 LoRA training** — do NOT rent cloud GPUs for that.
  Rent only if (a) moving to SDXL training, or (b) SaaS on-demand per-customer training
  where wall-clock speed is the product (Section 16).

---

## 4. Architecture overview

**Runtime pipeline (per asset):**

```
[prompt] --> STAGE 1: Generate (SD1.5 + LoRA, GPU) --> raw 512px image
          --> STAGE 2: Pixelize (CPU: downscale + OKLab quantise + crisp alpha)
          --> STAGE 3: Serve (preview / save PNG / pack sheet)
```

**Asset-creation loop (per character), layered on top:**

```
generate hero --> refine to a consistent set --> train character LoRA
             --> animate (pose-driven, per action) --> pixelize frames --> sprite sheet
```

Key insight: **stage 2 already works on anything, including animation frames.** The hard
part is never pixelizing — it's getting stage 1 to emit *consistent* frames. That's what
the character LoRA (Section 10) and the brute-force selector (Section 13) address.

---

## 5. Stage 1 — Generation

- **Model:** Stable Diffusion 1.5. Chosen over SDXL because it's light, fast, and the
  output is crushed to 64px/16 colours anyway — SDXL's fine-detail edge is mostly thrown
  away by the pixelizer. What survives the downscale is *composition, silhouette, prompt
  adherence*; SDXL helps those but it's a smaller win than for full-res art. Keep SDXL as
  an optional quality pass (`nerijs/pixel-art-xl` + LCM-LoRA, ~8 steps, guidance ~1.5) —
  only worth it with ≥16 GB VRAM/unified memory.
- **Base checkpoint:** `stable-diffusion-v1-5/stable-diffusion-v1-5` (the old `runwayml`
  repo is deprecated). Any SD 1.5 checkpoint is swappable.
- **Style LoRA:** `artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5`,
  trigger tokens `pixel art, PixArFK` (auto-appended to the prompt).
- **Canvas:** generate at **512×512** (SD 1.5 native), then let stage 2 shrink. Generating
  tiny directly looks worse.
- **Settings that matter:** `num_inference_steps` ~28, `guidance_scale` ~7 (DPM++ scheduler).
  Negative prompt strips realism/blur/smooth-shading.
- **MPS note:** float32 is the safe default; fp16 is faster once stable. `enable_attention_slicing()`.

See `generate.py` for the working implementation.

---

## 6. Stage 2 — Pixelize (the secret sauce)

Three operations, in order:

1. **Downscale to the target grid** with an **area/box filter** (NOT nearest — nearest
   aliases on downscale). Longest side = `size` (default 64).
2. **Palette quantise** to `colors` (default 16) using **k-means in OKLab space**, so the
   reduced palette is perceptually even instead of muddy. (RGB quantisation is the thing
   that produces blotchy pixel art; OKLab fixes it.) Only opaque pixels are clustered.
3. **Crisp alpha** — hard-threshold to fully opaque/transparent, no semi-transparent fringe.

Optional preview: nearest-neighbour upscale for viewing the tiny sprite large.

This is deterministic and CPU-only. **It is also where world cohesion gets enforced**
(Section 15): swap the per-image k-means for a *fixed shared palette* and every asset
snaps to the same colours. That upgrade is the single highest-leverage change for making
separately-generated assets look like one game. Implemented core is in `pixelize.py`.

---

## 7. Stage 3 — Serve

Minimal for now: preview, save PNG, (later) pack frames into a sprite sheet. A thin web UI
over `generate.py` on the 3080 box is the natural next step for LAN access, but the user
deprioritised it. For SaaS this becomes the API layer (Section 16).

---

## 8. LoRAs — two kinds, two jobs

This distinction is central to the whole system:

- **Style LoRA** — locks the *overall look*: palette, shading, line weight, vibe. Trained
  once on the game's aesthetic. It's the **glue**: everything generated with it on
  (enemies, blobs, buildings, props) automatically matches the world. **No hero image
  needed** for a new enemy — just prompt `"a little slime monster"` with the style LoRA on.
- **Character LoRA** — locks *one specific character's identity* (that exact guy, his
  colours and shape) across angles and poses. This is what makes animation hold together.

You can stack them: character LoRA + style LoRA + fixed palette = an on-model character
that also belongs to the world.

---

## 9. Training LoRAs

- **Why LoRA, not full fine-tune:** full fine-tuning rewrites billions of weights (too
  heavy). A LoRA trains a tiny adapter layer — output is a few MB, not ~4 GB — and is
  swappable/stackable.
- **Toolkit:** `kohya_ss` (the standard). Runs on the Windows/3080 box (CUDA).
- **Dataset:** ~15–30 images, **quality ≫ quantity** (20 great images beat 100 sloppy
  ones). Each image gets a caption; bake a unique **trigger token** into every caption —
  that token later summons the style/character in a prompt.
- **Resolution:** train at **512px** — no reason to go bigger when output is tiny sprites.
- **Speed levers (real):** use LoRA (already), keep images at 512, enable **xformers** and
  **mixed precision (fp16/bf16)**, don't over-crank step count. These cut runtime with no
  meaningful quality loss.
- **Time on the 3080:** ~30–45 min per SD 1.5 LoRA. Comfortable in 10–12 GB.
- **When to rent a bigger GPU:** NOT for SD 1.5 (the 3080 already fits; a bigger card just
  finishes the same job faster and the real bottleneck is human curation, which no GPU
  speeds up). Rent only for **SDXL training** (genuinely heavier, 3080 gets tight) or
  **SaaS on-demand training** (Section 16).

---

## 10. The character bootstrapping loop (the ratchet)

Chicken-and-egg problem: to train a consistent character you need consistent images, but
making consistent images is the thing you're trying to solve. Break it with a ratchet:

1. **Hero image.** Pick the single best generated picture of the character. This is the anchor.
2. **Refine to a set.** Use **img2img** feeding the hero back in at *medium denoise
   strength*, prompting for different angles/poses. Leaning on the anchor keeps outputs
   roughly on-model (same colours, same vibe). Generate *dozens*.
3. **Curate.** Hand-pick the 8–10 that genuinely read as the same character; bin the rest;
   touch up in an editor if needed. → training set.
4. **Train** the character LoRA on those (Section 9).
5. **Ratchet.** The trained model now emits *more* consistent images → harvest an even
   better dataset → retrain. Repeat.

**Rounds & time:** usually **2–3 rounds**. Round 1 is the painful hand-curate; round 2 is
easier because outputs are more on-model; most people have a reliable character by round
2–3 (diminishing returns after). Each round ≈ **half a day, mostly human curation** (~1 hr
generate+curate + 30–45 min training on the 3080). Two rounds over a weekend → character nailed.

Curation is the bottleneck — which motivates Section 11.

---

## 11. Auto-curation (remove the human from the loop)

Replace/reduce manual curation with automated scoring — essential for SaaS (customers can't
hand-curate):

- **Identity filter:** use **CLIP** (or similar image embedding) to score every generated
  candidate's similarity to the hero image; keep the closest, drop the rest.
- **Stack a second check:** "is this clean pixel art, not a mess?" as a separate scorer.
  Only candidates passing *both* survive.
- **Honest limitation:** CLIP is good at "roughly the same character" but weak on fine
  errors a human eye catches instantly (wrong finger count, off eye). So it gets you most
  of the way and hugely cuts manual work, but a quick final human glance may still help for
  the personal tool. For SaaS, automate the bulk and accept the residual error rate.

---

## 12. Animation — the hard problem

Pixelizing frames is trivial. The difficulty is **stage 1 emitting a coherent sequence**:
the model has **no memory between frames**, so frame 2 forgets exactly what frame 1 drew →
flicker, wobbling faces, melting/reappearing hands. This is the **temporal consistency /
identity drift** problem and it is genuinely unsolved at high quality (even PixelLab's
output needs hand-cleanup).

How the real tools wrangle it (stack all three):

1. **Skeleton / pose control** — feed a per-frame stick-figure skeleton (ControlNet-style)
   so the pose is *locked* and only the intended limbs move.
2. **Reference conditioning** — keep showing the model the one good character as an anchor
   so it drifts less.
3. **Inpainting** — freeze the parts that shouldn't change (face, torso) and only repaint
   the moving bits (legs).

**What a character LoRA does and doesn't fix:** it fixes **identity drift** (stays the same
guy) — a real win. It does **not** fix **frame-to-frame motion wobble** (a foot jitters, an
arm pops), because frames are still drawn semi-independently. Net effect: you go from "every
frame is a different person" to "clearly the right guy, but the walk is a touch janky" —
much smaller cleanup, not zero.

**Other levers:**
- **AnimateDiff** (or similar motion modules) — bakes a notion of motion *across* frames
  rather than rolling each independently; attacks flicker at the root.
- **Lean into small sprites** — at 16×16 / 32×32 a lot of wobble simply vanishes in the
  downscale. Small sprites hide a multitude of sins.

---

## 13. Brute-force animation selection (our candidate edge)

The idea worth prototyping, because it directly attacks the wobble the off-the-shelf tools
struggle with, and it's cheap+automatable (ideal for SaaS):

**Per frame, generate ~100 candidates, then automatically select the best one.** Scoring:

- **Pose match** — how well the candidate matches that frame's target skeleton.
- **Continuity** — how smoothly it flows from the *previous already-locked frame*.

Pick the winner, **lock it**, move to the next frame (each selection conditioned on the
locked previous frame). The **continuity term is the key bit** — you're explicitly
selecting for smooth motion, not picking each frame in isolation, which is exactly what
kills jitter.

Cost: ~100× generations per frame, but generation is a fraction of a penny, so it's fine
(economics in Section 16). Combine with the character LoRA (identity) + pose control (pose)
+ this selector (motion) for the best shot at usable animation without hand-cleanup.

> Status: **unproven, promising.** Prototype and measure against a hand-animated baseline
> before betting the SaaS on it. Fallback stays: PixelLab API for the animation step.

---

## 14. End-to-end character journey (the product flow)

1. **Generate** the character → batch → user (or auto-curator) picks the **hero**.
2. **Refine** → img2img from hero → consistent multi-angle set → curate.
3. **Train** the **character LoRA** on the set (few min on big GPU / ~30–45 min on 3080).
4. **Animate** each action (walk ~8 frames, run, jump ~6 frames): feed per-frame **pose
   skeletons**; because the model now *knows* the character, he comes out on-model in each
   pose. Apply the Section-13 selector for motion smoothness.
5. **Pixelize** every frame (stage 2), pack into a **sprite sheet** → game-ready.

The middle **training** step is what makes step 4's animation actually cohere.

---

## 15. World cohesion (tying the whole game together)

Beyond one character — enemies, blobs, buildings, environment:

- **Style LoRA is the glue** (Section 8): trained once on the game's aesthetic. New enemy?
  Prompt `"a little slime monster"` with the style LoRA on — no hero image needed, because
  you're pinning *style*, not identity.
- **Shared locked palette is the real cohesion trick:** stage 2 already crushes everything
  to a fixed set of colours — **lock the *same* 16-colour palette for every asset**
  (characters, enemies, buildings) and they instantly look like one game even when
  generated completely separately. → upgrade `pixelize.py` to accept a fixed palette
  instead of per-image k-means.
- **Environment tiles** have one extra requirement: they must **tile seamlessly** edge-to-
  edge — that's a generation setting (seamless/tiling mode) on top of the same style rules.

Big picture: **one style LoRA + one locked palette = a coherent world.**

---

## 16. SaaS architecture & economics

**Shape:** customer prompts → on-demand generate/refine/train/animate → returns an animated
sprite sheet in a few minutes.

**Compute model:**
- **Per-customer training on demand:** rent a **big card by the second (A100/H100)** — not
  because the job needs the power, but to **buy down the customer's wait**. On an H100, SD
  1.5 LoRA training on a small set can drop to **a few minutes**. Scale to zero between
  customers.
- **Generation (incl. brute-force):** a fraction of a penny per image.

**Rough cost per fully-animated character:**
- Brute-force animation: ~100 candidates/frame × ~8 frames × 3 actions ≈ **~2,000+
  generations**. Even at ~0.1¢ each → **~$2–4 of raw compute**.
- One-off LoRA training on rented H100 (~$4–5/hr, few min) → **~$0.30–0.40**.
- CLIP scoring + pixelize → **~free** (light/CPU).
- **All-in ≈ $2–4 raw compute** prompt-to-animated-character.

**Pricing:** at, say, **$15–20/character**, margin is healthy even after overhead.

**The real cost/eng risk is NOT generation — it's cold starts & idle workers.** Loading
model weights on spin-up can eat a minute or two; that's the thing to engineer around
(warm pools, weight caching, hiding the wait behind UX). This is the actual hard part of
the SaaS, not the ML.

---

## 17. Hosting options & costs (as of mid-2026, verify before committing)

Ranked for this workload (bursty, low-to-moderate volume):

1. **Local first** — the 3080 box / Mac. Free. Use while iterating on prompts, LoRAs, the pixelizer.
2. **Serverless pay-per-generation** — **Modal** or **RunPod Serverless** for the custom
   pipeline (LoRA + pixelizer in your own container), billed per GPU-second; a gen is a few
   seconds → sub-cent. Modal has a $30/mo free tier; failed requests generally not billed.
   For *stock* SDXL+LoRA, **fal.ai / Replicate** do ~$0.002–0.003/image (fal ~20–40%
   cheaper than Replicate).
3. **AWS** — only if consolidating with existing AWS infra (the user already has an
   **S3/boto3 pipeline**). g5 (A10G) ~$1/hr on-demand, g6 (L4) a bit cheaper, spot cuts
   60–70%; but it bills egress and has **no native per-image mode** (you babysit the box or
   wire up a SageMaker async endpoint). Bedrock's managed Stable Diffusion **can't run your
   custom LoRA/pixelizer**, so it's out for the custom pipeline.
4. **Dedicated always-on GPU** — only at sustained high volume. A RunPod community 4090 is
   ~$0.34/hr (~$248/mo 24/7); wasteful for bursty use.

Reference points to re-verify: RunPod 4090 $0.34/hr (community) / $0.69 (secure), A100 80GB
$1.39/hr, serverless SDXL ~$0.002–0.003/image, AWS g5.xlarge ~$1.006/hr, H100 ~$4–5/hr.

---

## 18. Phased build roadmap

**Phase 0 — foundation (DONE):** `pixelize.py` + `generate.py`. Static prompt→sprite works.

**Phase 1 — solid local generator:**
- Move stage-1 generation onto the 3080 box (CUDA).
- Add fixed-palette mode to `pixelize.py` (Section 15) — highest-leverage cohesion upgrade.
- Optional thin LAN web UI over `generate.py`.

**Phase 2 — style LoRA:**
- Assemble a small style dataset; train a style LoRA in kohya on the 3080.
- Verify enemies/props/buildings come out coherent with just the style LoRA + locked palette.

**Phase 3 — character pipeline:**
- Implement the hero → img2img refine → curate → train ratchet (Section 10).
- Add CLIP auto-curation (Section 11).

**Phase 4 — animation:**
- Pose-skeleton driven generation (ControlNet-style) for walk/run/jump.
- Prototype the brute-force selector (Section 13); measure vs a hand-animated baseline and
  vs PixelLab. Decide build-vs-buy for the animation step on the evidence.
- Sprite-sheet packing.

**Phase 5 — SaaS (only if pursuing it):**
- Containerise the full pipeline; deploy on Modal/RunPod Serverless with scale-to-zero.
- Solve cold-start/idle (warm pools, weight caching).
- On-demand per-customer H100 training; billing; spend caps.

---

## 19. Open problems / decisions not yet made

- **Does the brute-force selector (13) actually beat hand-cleanup / PixelLab?** Unproven —
  the pivotal experiment. Everything SaaS-differentiating rides on it.
- **Character LoRA fixes identity but not motion** — confirm how much residual jitter
  remains after LoRA + pose control + selector, and whether it's shippable without a human pass.
- **SDXL vs SD 1.5** for the final quality bar — revisit once the pipeline works; gated by
  VRAM and the fact the pixelizer discards fine detail.
- **Tiling/seamless environment generation** — not yet specified in detail.
- **Cold-start engineering** — the real SaaS hard part; unspecified.
- **Fixed-palette design** — per-game palette definition, and how a customer picks/derives one.

---

### Appendix — the one-line mental model

**Generate (fuzzy, swappable) → Pixelize (deterministic core; palette = cohesion) → Serve.**
Layer a **style LoRA** (world glue) and **character LoRAs** (identity) on top; animate with
**pose control + a continuity-scored brute-force selector**; keep it all **local now,
serverless when it needs to leave the machine.**
