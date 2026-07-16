"""Stage 0 — the asset director: decide the workstream, then run it.

A prompt like "a small slime monster" needs a completely different path
through the pipeline than "mossy cobblestone ground" or "a knight in green
armour". The director makes that routing call and enriches the prompt for
Stable Diffusion, then `execute_plan` drives the chosen workstream:

    static_prop       generate -> pixelize
    simple_creature   generate -> pixelize -> procedural bounce/idle -> gif + sheet
    limbed_character  generate hero candidates -> print the ratchet next steps
                      (LoRA training runs in kohya; can't be one command yet)
    environment_tile  generate --tile -> pixelize -> seam check + preview grid

The decider is injectable: `llm_decider` calls Claude with structured outputs
(needs the [director] extra + credentials); `heuristic_decider` is a
deterministic keyword fallback so everything works offline, in CI, and when
the API is unreachable. Every workstream also stays directly invocable via its
own subcommand — the director is a front door, not a gatekeeper.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from .palette import Palette

WORKSTREAMS = ("static_prop", "simple_creature", "limbed_character", "environment_tile")

DIRECTOR_MODEL = "claude-opus-4-8"

DIRECTOR_SYSTEM = """\
You are the asset director AND concept artist for a top-down RPG pixel-art
pipeline. Given a customer's asset request, (1) route it to a workstream and
(2) ART-DIRECT it into a vivid Stable Diffusion 1.5 prompt. The customer's
words are a brief, not the final prompt — a terse request like "a small slime
monster" gives SD nothing to draw and yields a generic featureless blob. Your
job is to review the brief and improve it into something SD can render well.

Workstreams:
- static_prop: inanimate objects and portraits (a sword, a chest, a tree).
  Optionally animated procedurally — actions from: sway (trees, flags, plants,
  hanging signs — anything the wind moves), idle. Leave actions empty for
  truly static objects.
- simple_creature: limbless creatures animated by procedural squash & stretch
  (slimes, blobs, ghosts, coins, floating orbs). actions from: bounce, idle.
- limbed_character: anything with limbs needing pose-driven animation. Set
  body to "humanoid" (people, knights, goblins, bipedal monsters — actions
  from: walk, run, jump) or "quadruped" (wolves, horses, deer, dragons,
  four-legged beasts — actions from: walk, trot, gallop, jump).
- environment_tile: ground/wall textures that must tile seamlessly
  (grass, cobblestone, water, sand).

enriched_prompt — the core of your work. Rewrite the brief into ONE richly
specific subject description. Always name, concretely:
  * material / surface — gelatinous, glossy, metallic, furry, translucent...
  * colour — a specific hue, not just "coloured"
  * silhouette / shape language — round, blocky, spiky, tall
  * 2-4 distinctive features that make it read as THIS creature, not a generic
    one — eyes, mouth/expression, horns, armour trim, a weapon, a glow
  * mood / charm — "cute", "menacing", "friendly video-game creature"
End characters/creatures/props with a SHORT composition tail: "single subject,
full body, centered, plain white background". Tiles instead get "top-down view,
flat seamless texture, no objects".

CRITICAL length limit: the whole prompt must stay under ~60 words / 77 CLIP
tokens or the tail is silently truncated. Be vivid but tight — spend your words
on the subject, not on repeating what the negative prompt already forbids. Do
NOT add style words like "pixel art" — the trigger is prepended automatically.

Worked example — brief "a small slime monster" becomes:
  "a small round gelatinous slime monster, glossy translucent lime-green body,
  two big round eyes, a wide cheerful grin, tiny stubby arms, cute chibi
  video-game creature, single subject, full body, centered, plain white
  background"

negative_additions: put the what-NOT-to-draw enforcement here, not in the
positive prompt. For characters/creatures/props always include "multiple
creatures, crowd, collage, pattern, border, frame, busy background, scenery,
pedestal, platform, base, shadow, reflection, ground" plus anything
asset-specific; "" only for tiles. Pedestals and cast shadows matter: the
background remover can't tell them from the subject, so they must never be
generated.
isolate: true to strip the plain background to transparency after
generation (characters/creatures/props); false for tiles.
size: sprite longest side in px — 64 for everything (characters, creatures,
props, tiles all share the game's 64px logical grid; engines upscale for
display) unless the request implies tiny or huge.
colors: palette size, default 16.
reasoning: one sentence on the routing choice."""


@dataclass
class Plan:
    workstream: str
    enriched_prompt: str
    negative_additions: str = ""
    size: int = 64
    colors: int = 16
    actions: list[str] = field(default_factory=list)
    body: str = "humanoid"     # rig for limbed_character: humanoid | quadruped
    isolate: bool = True       # strip plain background to transparency (not tiles)
    reasoning: str = ""
    source: str = "heuristic"  # which decider produced this plan

    def __post_init__(self):
        if self.workstream not in WORKSTREAMS:
            raise ValueError(f"unknown workstream {self.workstream!r}")
        if self.body not in ("humanoid", "quadruped"):
            raise ValueError(f"unknown body type {self.body!r}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> Plan:
        return cls(**json.loads(text))


DeciderFn = Callable[[str], Plan]

_DEFAULT_ACTIONS = {
    "simple_creature": ["bounce", "idle"],
    "limbed_character": ["walk", "run", "jump"],
}
_QUADRUPED_ACTIONS = ["walk", "gallop", "jump"]

# ---------------------------------------------------------------------------
# Deciders
# ---------------------------------------------------------------------------

_TILE_WORDS = {"tile", "texture", "ground", "floor", "wall", "terrain", "grass",
               "cobblestone", "water", "sand", "seamless", "pattern", "pavement"}
_BLOB_WORDS = {"slime", "blob", "ghost", "jelly", "orb", "coin", "gem", "crystal",
               "pickup", "bubble", "spirit", "wisp", "eyeball"}
_LIMBED_WORDS = {"knight", "wizard", "warrior", "archer", "hero", "character",
                 "person", "man", "woman", "goblin", "skeleton", "zombie", "orc",
                 "villager", "monster", "soldier", "npc", "boss"}
_QUADRUPED_WORDS = {"wolf", "dog", "hound", "cat", "horse", "pony", "deer",
                    "stag", "fox", "boar", "bear", "lion", "tiger", "dragon",
                    "cow", "sheep", "goat", "pig", "rat", "beast"}
_SWAY_WORDS = {"tree", "flag", "banner", "plant", "flower", "bush", "sapling",
               "palm", "reed", "vine", "lantern", "sign", "sway", "swaying",
               "wind", "waving", "fluttering"}
_SUBJECT_NEGATIVE = ("multiple creatures, crowd, collage, pattern, border, "
                     "frame, busy background, scenery, landscape, sky, "
                     "clouds, hills, stars, pedestal, platform, base, "
                     "shadow, reflection, ground")


def heuristic_decider(prompt: str) -> Plan:
    """Deterministic keyword routing — the offline/CI/API-down fallback."""
    words = set(re.findall(r"[a-z]+", prompt.lower()))

    body = "humanoid"
    size = 64                  # one logical grid for everything (PLAN.md §6)
    # Forceful single-subject composition — weak hints produce collages
    # (Checkpoint A/B finding). Kept SHORT: CLIP truncates at 77 tokens, and the
    # anti-scenery / anti-pedestal / anti-shadow enforcement already lives in the
    # negative prompt, so repeating it here only crowds out the trigger. The rich
    # per-subject detail is the LLM director's job (see DIRECTOR_SYSTEM).
    subject_suffix = "full body, centered, plain white background"
    negative = _SUBJECT_NEGATIVE
    isolate = True
    # Subject words win over tile words: "a slime on the floor" is a creature,
    # not a floor tile. Only route to environment_tile when nothing names a
    # creature/character — real tile prompts ("mossy cobblestone ground") don't.
    if words & _BLOB_WORDS:
        workstream = "simple_creature"
        enriched = f"a single {prompt}, {subject_suffix}"
    elif words & _QUADRUPED_WORDS:
        workstream = "limbed_character"
        body = "quadruped"
        enriched = f"a single {prompt}, side view, {subject_suffix}"
    elif words & _LIMBED_WORDS:
        workstream = "limbed_character"
        enriched = f"a single {prompt}, {subject_suffix}"
    elif words & _TILE_WORDS:
        workstream = "environment_tile"
        enriched = f"{prompt}, top-down view, flat seamless texture, no objects"
        negative, isolate = "", False
    else:
        workstream = "static_prop"
        enriched = f"a single {prompt}, {subject_suffix}"

    if body == "quadruped":
        actions = list(_QUADRUPED_ACTIONS)
    else:
        actions = list(_DEFAULT_ACTIONS.get(workstream, []))
    if workstream == "static_prop" and words & _SWAY_WORDS:
        actions = ["sway"]

    return Plan(
        workstream=workstream,
        enriched_prompt=enriched,
        negative_additions=negative,
        size=size,
        actions=actions,
        body=body,
        isolate=isolate,
        reasoning="keyword heuristic",
        source="heuristic",
    )


def llm_decider(prompt: str, model: str = DIRECTOR_MODEL, client=None) -> Plan:
    """Route via Claude with structured outputs (needs the [director] extra)."""
    from typing import Literal

    import anthropic
    from pydantic import BaseModel, Field

    class PlanModel(BaseModel):
        workstream: Literal[
            "static_prop", "simple_creature", "limbed_character", "environment_tile"
        ]
        enriched_prompt: str
        negative_additions: str = ""
        size: int = 64
        colors: int = 16
        actions: list[str] = Field(default_factory=list)
        body: Literal["humanoid", "quadruped"] = "humanoid"
        isolate: bool = True
        reasoning: str = ""

    client = client or anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=2048,
        system=DIRECTOR_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_format=PlanModel,
    )
    p = response.parsed_output
    if not (actions := p.actions):
        actions = (_QUADRUPED_ACTIONS if p.body == "quadruped"
                   else _DEFAULT_ACTIONS.get(p.workstream, []))
    return Plan(
        workstream=p.workstream,
        enriched_prompt=p.enriched_prompt,
        negative_additions=p.negative_additions,
        size=p.size,
        colors=p.colors,
        actions=list(actions),
        body=p.body,
        isolate=p.isolate,
        reasoning=p.reasoning,
        source=f"llm:{model}",
    )


def plan_asset(prompt: str, offline: bool = False,
               model: str = DIRECTOR_MODEL, client=None) -> Plan:
    """Plan with the LLM decider, falling back to heuristics on any failure
    (missing [director] extra, no credentials, API unreachable)."""
    if offline:
        return heuristic_decider(prompt)
    try:
        return llm_decider(prompt, model=model, client=client)
    except Exception as e:  # noqa: BLE001 — fall back, never block the pipeline
        print(f"director: LLM unavailable ({type(e).__name__}: {e}) — "
              f"using keyword heuristics")
        return heuristic_decider(prompt)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

_RATCHET_NEXT_STEPS = """\
Hero candidates written to {out_dir}. Next steps for the character ratchet
(handover §10; LoRA training runs in kohya, so this can't be one command yet):

  1. pick the best hero image, then:
     spriteforge refine {best} --prompt "{prompt}" -o {out_dir}/round1
  2. spriteforge curate {out_dir}/round1 --hero {best} -o {out_dir}/keep --keep 10
  3. spriteforge dataset prep {out_dir}/keep/*.png -o {out_dir}/lora --trigger <token>
  4. train per {out_dir}/lora/NOTES.md, then animate:
     spriteforge animate "{prompt}, <token>" --action walk{body_flag} \\
         -o {out_dir}/frames \\
         --character-lora {out_dir}/lora/output/<token>.safetensors"""

HERO_CANDIDATES = 4


def _pick_best_by_clip(images: list[Image.Image], prompt: str) -> int:
    """Best-of-N winner via CLIP (prompt alignment + clean-vs-blurry margin).

    Never allowed to block the pipeline: if the [curate] extra or its models
    are unavailable, fall back to the first candidate.
    """
    try:
        from .curate import rank_by_prompt

        ranked = rank_by_prompt(images, prompt)
        return ranked[0][0] if ranked else 0
    except Exception as e:  # noqa: BLE001 — CLIP missing/unreachable
        print(f"director: CLIP pick unavailable ({type(e).__name__}: {e}) — "
              f"keeping candidate 0")
        return 0


def execute_plan(
    plan: Plan,
    out_dir: str | Path,
    palette: Palette | None = None,
    seed: int = 0,
    fp16: bool | None = None,
    pipe=None,
    candidates: int = 1,
    pick: int | None = None,
    pick_fn: Callable[[list[Image.Image], str], int] | None = None,
    backend=None,
) -> dict:
    """Run the plan's workstream. Returns a dict of output paths/metrics.

    `candidates` rolls that many seeds for the single-sprite workstreams
    (simple_creature / static_prop) and keeps the best — a hero shouldn't ride
    on one lucky seed. `pick` forces a specific candidate index (human
    override); otherwise `pick_fn` chooses (default: CLIP best-of-N). All
    candidates are saved so the choice can be revisited.

    `pipe`/`pick_fn` are injectable for tests; by default the SD pipeline is
    built lazily (needs the [generate] extra — GPU boxes).
    """
    from .generate import generate
    from .pixelize import pixelize

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    negative_extra = plan.negative_additions.strip()

    if pipe is None:
        from .generate import build_pipe

        pipe = build_pipe(fp16=fp16, backend=backend)

    def _generate(seed_offset: int = 0):
        kwargs = {}
        if negative_extra:
            from .generate import DEFAULT_NEGATIVE

            kwargs["negative"] = f"{DEFAULT_NEGATIVE}, {negative_extra}"
        return generate(pipe, plan.enriched_prompt, seed=seed + seed_offset,
                        backend=backend, **kwargs)

    results: dict = {"workstream": plan.workstream}

    if plan.workstream == "environment_tile":
        from .tiling import disable_tiling, enable_tiling, seam_error, tile_preview

        enable_tiling(pipe)
        try:
            raw = _generate()
        finally:
            disable_tiling(pipe)  # the pipe may be reused for non-tile jobs
        sprite = pixelize(raw, size=plan.size, colors=plan.colors, palette=palette)
        sprite.save(out_dir / "tile.png")
        tile_preview(sprite, 3).save(out_dir / "tile_grid.png")
        results.update(sprite=out_dir / "tile.png",
                       preview=out_dir / "tile_grid.png",
                       seam_error=seam_error(raw))
        return results

    # A baked cast shadow only misbehaves when the sprite bounces (it would ride
    # up off the ground), so only de-shadow simple_creatures. Stripping it from
    # a static prop or a character risks eating legitimately grey detail — iron
    # bands on a chest, a stone base (field bug at Checkpoint A/B).
    deshadow = plan.workstream == "simple_creature"

    def _isolated(raw):
        if not plan.isolate:
            return raw
        from .isolate import isolate_subject

        subject, method = isolate_subject(raw, trim_shadow=deshadow)
        if method is None:
            print("director: could not isolate the subject — keeping the full "
                  "render (install the [isolate] extra for ML cutout of busy "
                  "backgrounds)")
        return subject

    if plan.workstream == "limbed_character":
        # A single knight roll is a gamble (collages, baked scenes), so generate
        # a batch and CLIP-pick the best into a ready-to-use sprite.png — same
        # best-of-N deal as creatures/props. All heroes + raws are still kept for
        # the LoRA ratchet. --candidates overrides the default hero count.
        n = candidates if candidates > 1 else HERO_CANDIDATES
        heroes, hero_sprites = [], []
        for i in range(n):
            raw = _generate(seed_offset=i)
            path = out_dir / f"hero_{i:02d}.png"
            hero_sprite = pixelize(_isolated(raw), size=plan.size,
                                   colors=plan.colors, palette=palette)
            hero_sprite.save(path)
            raw.save(out_dir / f"hero_{i:02d}_raw.png")  # refine wants the raw render
            heroes.append(path)
            hero_sprites.append(hero_sprite)

        if pick is not None:
            winner = pick % n
        else:
            chooser = pick_fn or _pick_best_by_clip
            winner = chooser(hero_sprites, plan.enriched_prompt) % n
        hero_sprites[winner].save(out_dir / "sprite.png")
        print(f"director: kept hero {winner} of {n} "
              f"({'forced' if pick is not None else 'CLIP best-of-N'})")

        body_flag = " --body quadruped" if plan.body == "quadruped" else ""
        next_steps = _RATCHET_NEXT_STEPS.format(
            out_dir=out_dir, best=out_dir / f"hero_{winner:02d}_raw.png",
            prompt=plan.enriched_prompt, body_flag=body_flag)
        (out_dir / "NEXT_STEPS.md").write_text(next_steps + "\n")
        results.update(heroes=heroes, chosen=winner, sprite=out_dir / "sprite.png",
                       next_steps=out_dir / "NEXT_STEPS.md")
        return results

    # static_prop and simple_creature share the single-sprite start; both may
    # carry procedural actions (a slime bounces, a tree sways, a chest is still)
    n = max(1, candidates)
    sprites = []
    for i in range(n):
        raw = _generate(seed_offset=i)
        sprites.append(pixelize(_isolated(raw), size=plan.size,
                                colors=plan.colors, palette=palette))

    if n > 1:
        cand_dir = out_dir / "candidates"
        cand_dir.mkdir(exist_ok=True)
        cand_paths = []
        for i, s in enumerate(sprites):
            p = cand_dir / f"cand_{i:02d}.png"
            s.save(p)
            cand_paths.append(p)
        if pick is not None:
            winner = pick % n
        else:
            chooser = pick_fn or _pick_best_by_clip
            winner = chooser(sprites, plan.enriched_prompt) % n
        print(f"director: kept candidate {winner} of {n} "
              f"({'forced' if pick is not None else 'CLIP best-of-N'})")
        results.update(candidates=cand_paths, chosen=winner)
        sprite = sprites[winner]
    else:
        sprite = sprites[0]

    sprite_path = out_dir / "sprite.png"
    sprite.save(sprite_path)
    results["sprite"] = sprite_path

    if plan.actions:
        from .animate.procedural import PROCEDURAL_ACTIONS, PROCEDURAL_FPS
        from .animate.sheet import save_sheet
        from .preview import make_gif

        sheet_frames: dict[str, list[Image.Image]] = {}
        for action in plan.actions:
            cycle = PROCEDURAL_ACTIONS.get(action)
            if cycle is None:
                print(f"director: skipping unknown procedural action {action!r}")
                continue
            frames = cycle(sprite)
            frames_dir = out_dir / action
            frames_dir.mkdir(exist_ok=True)
            for k, frame in enumerate(frames):
                frame.save(frames_dir / f"{action}_{k:02d}.png")
            make_gif(frames, out_dir / f"{action}.gif",
                     fps=PROCEDURAL_FPS.get(action, 10), scale=2)
            sheet_frames[action] = frames
        if sheet_frames:
            save_sheet(sheet_frames, out_dir / "sheet.png", fps=PROCEDURAL_FPS)
            results.update(sheet=out_dir / "sheet.png",
                           actions=sorted(sheet_frames))
    return results
