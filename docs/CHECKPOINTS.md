# GPU validation checkpoints

Everything CPU-side is CI-tested; these are the runs that need real hardware.
Do them in order — each gates the next stage of the pipeline.

## Checkpoint A/B — generation + world cohesion (3080 box)

```bash
git clone https://github.com/J-Glatter/surfside-code-sample && cd surfside-code-sample
pip install -e ".[generate,director,isolate]"

# Backend: SDXL + nerijs/pixel-art-xl is the DEFAULT (bake-off winner — far more
# coherent single subjects and far less baked-in scenery than SD1.5). Pass
# --sd15 anywhere to fall back to the lighter SD1.5 + PixArFK stack (the only
# backend with a proven quadruped openpose ControlNet today).
#
# Let the director art-direct the prompt. A terse brief ("a small slime
# monster") gives the model little to draw, so the LLM director expands it into
# a concrete visual description (material, colour, features, expression) and
# forbids pedestals/shadows the isolator can't strip. Set the key to enable it
# (~1c/asset); without it the keyword heuristic still hardens composition.
export ANTHROPIC_API_KEY=sk-ant-...     # optional but recommended for creatures
spriteforge make "a small slime monster" -o out/slime --candidates 8   # best-of-8

# 1. CUDA path + basic generation (first run downloads ~4 GB)
#    Prompt for a plain white background and use --isolate so the sprite gets
#    real transparency (field-tested: without this, sky/grass is baked in).
#    -o without an extension is fine now — it defaults to .png.
spriteforge generate "a single brave knight in green armour, one figure, full body, centered, floating on a plain solid white background, no shadow, no ground" -o knight.png --seed 7 --preview 4 --isolate
#    -> confirm it reports no CPU warning; check the sprite's corners are transparent

# 2. Generate a few style references, then lock a world palette from the best
spriteforge generate "a small slime monster" -o slime.png --seed 3
spriteforge generate "a stone watchtower" -o tower.png --seed 11
spriteforge palette extract knight.png slime.png tower.png -o game.json --colors 16
spriteforge palette show game.json --swatch palette.png

# 3. The cohesion test: regenerate all three with the SAME locked palette
spriteforge generate "a brave knight in green armour, full body" -o knight2.png --seed 7 --palette game.json
spriteforge generate "a small slime monster" -o slime2.png --seed 3 --palette game.json
spriteforge generate "a stone watchtower" -o tower2.png --seed 11 --palette game.json
```

**Pass:** the three `*2.png` sprites read as one game. The default grid is 64px
logical (view with `--preview 4`); also judge whether 16 colours suits the art
direction — try `--colors 32` and re-extract if shading feels cramped.

## Checkpoint C — your own style LoRA (A40 / 24 GB+ box)

Replaces the stock `nerijs/pixel-art-xl` with a LoRA trained on YOUR sprites, so
every future asset shares one look. Needs a kohya_ss (sd-scripts) checkout.
SDXL LoRA training at 1024px wants ~16-24 GB VRAM — the A40 is comfortable, a
10 GB card is not (drop to `--sd15` there).

```bash
# 0. Gather the seed set: 15-30 of your cleanest, isolated, single-subject
#    sprites, all in the target look. The best-of-N candidates/ folders from
#    Checkpoint A/B are the source — copy the good ones into style_refs/.
#    Quality >> quantity; one bad sprite teaches the LoRA a bad habit.

# 1. Build the kohya dataset (emits the SDXL config + launch notes automatically)
spriteforge dataset prep style_refs/*.png -o style_ds --trigger myworld_style --name myworld-style
#    -> style_ds/kohya_config.toml targets stable-diffusion-xl-base-1.0 @ 1024,
#       sdxl_train_network.py. See style_ds/NOTES.md for the exact command:

# 2. Train (one-time kohya setup, then ~30-60 min on the A40)
git clone https://github.com/kohya-ss/sd-scripts && cd sd-scripts
pip install -r requirements.txt && accelerate config default
accelerate launch sdxl_train_network.py --config_file ../style_ds/kohya_config.toml
#    -> produces style_ds/output/myworld-style.safetensors (a few MB)

# 3. Plug it in — --style-lora replaces the stock LoRA, --style-trigger carries
#    your training token. Now the WHOLE pipeline speaks your style:
spriteforge make "a small slime monster" -o out/slime_myworld --candidates 12 \
    --style-lora style_ds/output/myworld-style.safetensors --style-trigger myworld_style \
    --palette out/world/game.json
```

**Pass:** assets made with `--style-lora` share a consistent look with each
other AND with your Checkpoint A/B set, with less scene-bias/prompt-fighting
than the stock LoRA needed. Generate 3-4 different subjects and eyeball them as
a set.

### Optional: the character ratchet (a consistent named hero across angles)

A *character* LoRA (vs the *style* LoRA above) locks one specific character so
it animates on-model. Uses the same tools:

```bash
# round 1: img2img a multi-angle set off one hero, curate, train
spriteforge refine hero.png --prompt "a brave knight in green armour" -o round1 --per-variation 8
pip install -e ".[curate]"   # first time only
spriteforge curate round1 --hero hero.png -o round1_keep --keep 10
spriteforge dataset prep round1_keep/*.png -o knight_ds --trigger sks_knight --class-word character
#    train per knight_ds/NOTES.md, then ratchet round 2 stacking the round-1 LoRA:
spriteforge refine hero.png --prompt "a brave knight in green armour" -o round2 \
    --character-lora knight_ds/output/sks_knight.safetensors
#    retrain -> most characters are reliable by round 2-3. The character LoRA
#    then feeds the animation stage (Checkpoint D, spriteforge animate).
```

**Pass:** round-2 outputs are clearly the same character across angles, and
curation needed only a quick glance rather than heavy hand-picking.

## Checkpoint D — animation + the pivotal experiment (task 15)

Backend split (openpose ControlNet availability, verified 2026):
- **Humanoid → SDXL.** `xinsir/controlnet-openpose-sdxl-1.0` is solid, and the
  frames pick up your style LoRA (`$SPRITEFORGE_STYLE_LORA`). The reliable path.
- **Quadruped → SD1.5 (`--sd15`).** There is NO SDXL animal-openpose ControlNet
  — the animal-pose ecosystem (`huchenlei`/`crishhh` animal_openpose) is
  SD1.5-only. So four-legged animation runs on SD1.5; lock the world with
  `--palette` for cohesion (the SDXL style LoRA can't cross architectures).
  `animate --body quadruped` on SDXL stops with a message pointing here.

```bash
pip install -e ".[animate]"
export SPRITEFORGE_STYLE_LORA=.../myworld-style.safetensors   # frames match the world
export SPRITEFORGE_STYLE_TRIGGER=myworld_style

# 0. Eyeball the canonical cycles first (CPU, instant):
spriteforge skeleton --action walk -o skel_preview           # --body quadruped for animals
#    tune constants in src/spriteforge/animate/skeleton*.py if the gait looks off

# 1a. Humanoid walk (SDXL + xinsir openpose + your style LoRA). Keep --candidates
#     modest: at SDXL 1024 each is ~10-15s, so 100 x 8 frames is hours.
spriteforge animate "a brave knight in green armour with a sword and shield" \
    --action walk -o frames/walk --palette game.json --candidates 16
#    note the reported mean transition cost (lower = smoother)

# 1b. Quadruped walk (SD1.5 animal openpose). UNVERIFIED: confirm the ControlNet
#     loads in diffusers and our AP-10K skeleton matches its keypoint convention
#     (the animal skeleton carries no ears/tail). If limbs don't track, adjust
#     skeleton_quadruped.py or try --controlnet huchenlei/animal_openpose.
spriteforge animate "a red fox" --body quadruped --sd15 \
    -o frames/fox_walk --palette game.json --candidates 16

# 2. Cheap ablation: --candidates 1 is "no selector" — compare the two walks
# 3. Pack and drop into the engine
spriteforge sheet walk=frames/walk run=frames/run jump=frames/jump -o knight_sheet.png
```

**The experiment (decides build-vs-buy):** same character, walk cycle via
(a) selector at 100 candidates, (b) selector off (1 candidate), and when a
PixelLab account exists (c) PixelLab's API. Compare mean transition cost +
eyeball at game speed. If (a) isn't clearly shippable-or-close, the fallback
stands: PixelLab for the animation step.
