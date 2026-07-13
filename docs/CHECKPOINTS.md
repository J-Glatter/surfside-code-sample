# GPU validation checkpoints

Everything CPU-side is CI-tested; these are the runs that need real hardware.
Do them in order — each gates the next stage of the pipeline.

## Checkpoint A/B — generation + world cohesion (3080 box)

```bash
git clone https://github.com/J-Glatter/surfside-code-sample && cd surfside-code-sample
pip install -e ".[generate]"

# 1. CUDA path + basic generation (first run downloads ~4 GB)
spriteforge generate "a brave knight in green armour, full body" -o knight.png --seed 7 --preview 2
#    -> confirm it reports no CPU warning, and speed feels right for a 3080

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

## Checkpoint C — style LoRA + the character ratchet (3080 box)

Needs a kohya_ss checkout on the box.

```bash
# 1. Style LoRA: curate 15-30 images that nail the game's look, then
spriteforge dataset prep style_refs/*.png -o style_ds --trigger myworld_style --name myworld-style
#    train per style_ds/NOTES.md (~30-45 min), then verify cohesion:
#    prompt "a little slime monster, myworld_style" with the trained LoRA loaded.

# 2. Character ratchet round 1: pick your hero image, then
spriteforge refine hero.png --prompt "a brave knight in green armour" -o round1 --per-variation 8
pip install -e ".[curate]"   # first time only
spriteforge curate round1 --hero hero.png -o round1_keep --keep 10
#    quick human glance over round1_keep (CLIP misses fine errors), then
spriteforge dataset prep round1_keep/*.png -o knight_ds --trigger sks_knight --class-word character
#    train the character LoRA per knight_ds/NOTES.md

# 3. Ratchet round 2: same, but stacking the round-1 LoRA
spriteforge refine hero.png --prompt "a brave knight in green armour" -o round2 --character-lora knight_ds/output/sks_knight.safetensors
spriteforge curate round2 --hero hero.png -o round2_keep --keep 10
#    retrain -> most characters are reliable by round 2-3
```

**Pass:** round-2 outputs are clearly the same character across angles, and
curation needed only a quick glance rather than heavy hand-picking.

## Checkpoint D — animation + the pivotal experiment (task 15)

```bash
pip install -e ".[animate]"

# 0. Eyeball the canonical cycles first (CPU, instant):
spriteforge skeleton --action walk -o skel_preview
#    tune constants in src/spriteforge/animate/skeleton.py if the gait looks off

# 1. Animate with the trained character LoRA + the locked palette
spriteforge animate "a brave knight in green armour, sks_knight" --action walk \
    -o frames/walk --character-lora knight_ds/output/sks_knight.safetensors \
    --palette game.json --candidates 100
#    note the reported mean transition cost (lower = smoother)

# 2. Cheap ablation: --candidates 1 is "no selector" — compare the two walks
# 3. Pack and drop into the engine
spriteforge sheet walk=frames/walk run=frames/run jump=frames/jump -o knight_sheet.png
```

**The experiment (decides build-vs-buy):** same character, walk cycle via
(a) selector at 100 candidates, (b) selector off (1 candidate), and when a
PixelLab account exists (c) PixelLab's API. Compare mean transition cost +
eyeball at game speed. If (a) isn't clearly shippable-or-close, the fallback
stands: PixelLab for the animation step.
