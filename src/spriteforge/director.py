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
You are the asset director for a top-down RPG pixel-art pipeline. Given a
customer's asset request, decide which workstream produces it and enrich the
prompt for Stable Diffusion 1.5.

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

enriched_prompt: rewrite the request for SD 1.5 — subject first, add
composition hints ("full body, centered, plain background" for characters and
creatures; "top-down view, flat texture" for tiles). Do not add style words
like "pixel art" — a style LoRA handles that.
negative_additions: extra negative-prompt terms this asset needs, or "".
size: sprite longest side in px — 256 for characters/creatures/props unless
the request implies tiny or huge; 128 for tiles.
colors: palette size, default 16.
reasoning: one sentence on the routing choice."""


@dataclass
class Plan:
    workstream: str
    enriched_prompt: str
    negative_additions: str = ""
    size: int = 256
    colors: int = 16
    actions: list[str] = field(default_factory=list)
    body: str = "humanoid"     # rig for limbed_character: humanoid | quadruped
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


def heuristic_decider(prompt: str) -> Plan:
    """Deterministic keyword routing — the offline/CI/API-down fallback."""
    words = set(re.findall(r"[a-z]+", prompt.lower()))

    body = "humanoid"
    if words & _TILE_WORDS:
        workstream, size = "environment_tile", 128
        enriched = f"{prompt}, top-down view, flat texture"
    elif words & _BLOB_WORDS:
        workstream, size = "simple_creature", 256
        enriched = f"{prompt}, full body, centered, plain background"
    elif words & _QUADRUPED_WORDS:
        workstream, size = "limbed_character", 256
        body = "quadruped"
        enriched = f"{prompt}, full body, side view, centered, plain background"
    elif words & _LIMBED_WORDS:
        workstream, size = "limbed_character", 256
        enriched = f"{prompt}, full body, centered, plain background"
    else:
        workstream, size = "static_prop", 256
        enriched = f"{prompt}, centered, plain background"

    if body == "quadruped":
        actions = list(_QUADRUPED_ACTIONS)
    else:
        actions = list(_DEFAULT_ACTIONS.get(workstream, []))
    if workstream == "static_prop" and words & _SWAY_WORDS:
        actions = ["sway"]

    return Plan(
        workstream=workstream,
        enriched_prompt=enriched,
        size=size,
        actions=actions,
        body=body,
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
        size: int = 256
        colors: int = 16
        actions: list[str] = Field(default_factory=list)
        body: Literal["humanoid", "quadruped"] = "humanoid"
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


def execute_plan(
    plan: Plan,
    out_dir: str | Path,
    palette: Palette | None = None,
    seed: int = 0,
    fp16: bool | None = None,
    pipe=None,
) -> dict:
    """Run the plan's workstream. Returns a dict of output paths/metrics.

    `pipe` is injectable for tests; by default the SD pipeline is built lazily
    (needs the [generate] extra — GPU boxes).
    """
    from .generate import generate
    from .pixelize import pixelize

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    negative_extra = plan.negative_additions.strip()

    if pipe is None:
        from .generate import build_pipe

        pipe = build_pipe(fp16=fp16)

    def _generate(seed_offset: int = 0):
        kwargs = {}
        if negative_extra:
            from .generate import DEFAULT_NEGATIVE

            kwargs["negative"] = f"{DEFAULT_NEGATIVE}, {negative_extra}"
        return generate(pipe, plan.enriched_prompt, seed=seed + seed_offset, **kwargs)

    results: dict = {"workstream": plan.workstream}

    if plan.workstream == "environment_tile":
        from .tiling import enable_tiling, seam_error, tile_preview

        enable_tiling(pipe)
        raw = _generate()
        sprite = pixelize(raw, size=plan.size, colors=plan.colors, palette=palette)
        sprite.save(out_dir / "tile.png")
        tile_preview(sprite, 3).save(out_dir / "tile_grid.png")
        results.update(sprite=out_dir / "tile.png",
                       preview=out_dir / "tile_grid.png",
                       seam_error=seam_error(raw))
        return results

    if plan.workstream == "limbed_character":
        heroes = []
        for i in range(HERO_CANDIDATES):
            raw = _generate(seed_offset=i)
            path = out_dir / f"hero_{i:02d}.png"
            pixelize(raw, size=plan.size, colors=plan.colors, palette=palette).save(path)
            raw.save(out_dir / f"hero_{i:02d}_raw.png")  # refine wants the raw render
            heroes.append(path)
        body_flag = " --body quadruped" if plan.body == "quadruped" else ""
        next_steps = _RATCHET_NEXT_STEPS.format(
            out_dir=out_dir, best=out_dir / "hero_00_raw.png",
            prompt=plan.enriched_prompt, body_flag=body_flag)
        (out_dir / "NEXT_STEPS.md").write_text(next_steps + "\n")
        results.update(heroes=heroes, next_steps=out_dir / "NEXT_STEPS.md")
        return results

    # static_prop and simple_creature share the single-sprite start; both may
    # carry procedural actions (a slime bounces, a tree sways, a chest is still)
    raw = _generate()
    sprite = pixelize(raw, size=plan.size, colors=plan.colors, palette=palette)
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
