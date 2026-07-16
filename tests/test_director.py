from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spriteforge.director import (
    Plan,
    execute_plan,
    heuristic_decider,
    plan_asset,
)
from spriteforge.palette import Palette

from .conftest import opaque_colors
from .tools import fake_torch_module

# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("a small slime monster", "simple_creature"),
        ("a spinning gold coin pickup", "simple_creature"),
        ("a brave knight in green armour", "limbed_character"),
        ("a shambling zombie villager", "limbed_character"),
        ("mossy cobblestone ground texture", "environment_tile"),
        ("seamless water tile", "environment_tile"),
        ("a treasure chest", "static_prop"),
        ("an ancient rune-carved sword", "static_prop"),
    ],
)
def test_heuristic_routing(prompt, expected):
    plan = heuristic_decider(prompt)
    assert plan.workstream == expected
    assert plan.source == "heuristic"
    assert prompt in plan.enriched_prompt


def test_heuristic_creature_words_beat_incidental_tile_words():
    # "floor"/"ground" are tile words, but a named creature must still route to
    # its creature workstream, not an environment tile
    assert heuristic_decider("a slime blob on the floor").workstream == "simple_creature"
    assert heuristic_decider("a wolf standing on grass").workstream == "limbed_character"
    # a genuine tile (no creature named) still routes to environment_tile
    assert heuristic_decider("mossy cobblestone ground").workstream == "environment_tile"


def test_heuristic_defaults():
    # one 64px logical grid for every workstream (PLAN.md §6)
    assert heuristic_decider("grass texture").size == 64
    assert heuristic_decider("a knight").size == 64
    assert heuristic_decider("a slime").actions == ["bounce", "idle"]
    assert heuristic_decider("a knight").actions == ["walk", "run", "jump"]
    assert heuristic_decider("a chest").actions == []


def test_heuristic_quadruped_routing():
    for prompt in ("a grey wolf", "a war horse", "a young deer"):
        plan = heuristic_decider(prompt)
        assert plan.workstream == "limbed_character", prompt
        assert plan.body == "quadruped", prompt
        assert plan.actions == ["walk", "gallop", "jump"], prompt
    # bipeds keep the humanoid rig
    knight = heuristic_decider("a knight")
    assert knight.body == "humanoid"
    assert knight.actions == ["walk", "run", "jump"]


def test_plan_rejects_unknown_body():
    with pytest.raises(ValueError):
        Plan(workstream="limbed_character", enriched_prompt="x", body="centipede")


def test_quadruped_next_steps_carry_body_flag(exec_env, tmp_path):
    plan = Plan(workstream="limbed_character", enriched_prompt="a wolf",
                size=32, body="quadruped")
    execute_plan(plan, tmp_path, pipe=exec_env)
    assert "--body quadruped" in (tmp_path / "NEXT_STEPS.md").read_text()


def test_heuristic_single_subject_hardening():
    """Checkpoint A/B findings: forceful composition + isolation defaults."""
    slime = heuristic_decider("a small slime monster")
    assert "a single" in slime.enriched_prompt
    assert "white background" in slime.enriched_prompt
    # anti-pedestal/shadow enforcement lives in the negative prompt (keeps the
    # positive prompt short enough that CLIP won't truncate the trigger)
    assert "multiple creatures" in slime.negative_additions
    assert "pedestal" in slime.negative_additions
    assert "shadow" in slime.negative_additions
    assert slime.isolate is True

    tile = heuristic_decider("grass texture")
    assert tile.isolate is False
    assert tile.negative_additions == ""


def test_execute_isolates_subjects_but_not_tiles(monkeypatch, tmp_path):
    import numpy as np

    monkeypatch.setitem(sys.modules, "torch", fake_torch_module())

    def subject_pipe(**kwargs):
        arr = np.full((128, 128, 3), 250, dtype=np.uint8)   # plain background
        arr[40:90, 40:90] = (60, 170, 80)                    # the subject
        return types.SimpleNamespace(images=[Image.fromarray(arr, "RGB")])

    pipe = MagicMock(side_effect=subject_pipe)

    plan = Plan(workstream="simple_creature", enriched_prompt="a slime",
                size=32, actions=[], isolate=True)
    results = execute_plan(plan, tmp_path / "iso", pipe=pipe)
    alpha = np.asarray(Image.open(results["sprite"]))[..., 3]
    assert alpha[0, 0] == 0          # background stripped
    assert (alpha == 255).any()      # subject kept

    tile_plan = Plan(workstream="environment_tile", enriched_prompt="grass",
                     size=32, isolate=False)
    results = execute_plan(tile_plan, tmp_path / "tile", pipe=pipe)
    alpha = np.asarray(Image.open(results["sprite"]))[..., 3]
    assert (alpha == 255).all()      # tiles stay fully opaque


def test_heuristic_sway_props():
    for prompt in ("a tree swaying in the wind", "an old oak tree",
                   "a tattered flag on a pole"):
        plan = heuristic_decider(prompt)
        assert plan.workstream == "static_prop", prompt
        assert plan.actions == ["sway"], prompt
    # motion words alone also trigger it
    assert heuristic_decider("a scarecrow waving in the wind").actions == ["sway"]


def test_execute_static_prop_with_sway(exec_env, tmp_path):
    plan = Plan(workstream="static_prop", enriched_prompt="a tree",
                size=48, actions=["sway"])

    results = execute_plan(plan, tmp_path, pipe=exec_env)

    assert len(list((tmp_path / "sway").glob("*.png"))) == 12
    assert (tmp_path / "sway.gif").exists()
    assert (tmp_path / "sheet.png").exists()
    assert results["actions"] == ["sway"]
    import json

    sheet_meta = json.loads((tmp_path / "sheet.json").read_text())
    assert sheet_meta["actions"]["sway"]["fps"] == 8  # ambient motions run slow


def test_plan_json_round_trip():
    plan = heuristic_decider("a small slime monster")
    assert Plan.from_json(plan.to_json()) == plan


def test_plan_rejects_unknown_workstream():
    with pytest.raises(ValueError):
        Plan(workstream="magic", enriched_prompt="x")


def test_llm_decider_wiring():
    pytest.importorskip("pydantic")
    from spriteforge.director import DIRECTOR_SYSTEM, llm_decider

    parsed = types.SimpleNamespace(
        workstream="simple_creature",
        enriched_prompt="a slime, full body, centered",
        negative_additions="multiple creatures",
        size=256, colors=16, actions=[], body="humanoid", isolate=True,
        reasoning="limbless blob",
    )
    client = MagicMock()
    client.messages.parse.return_value = types.SimpleNamespace(parsed_output=parsed)
    # anthropic only imported for the default client — stub keeps CI network-free
    sys.modules.setdefault("anthropic", types.ModuleType("anthropic"))

    plan = llm_decider("a small slime monster", client=client)

    _, kwargs = client.messages.parse.call_args
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["system"] == DIRECTOR_SYSTEM
    assert kwargs["messages"] == [{"role": "user", "content": "a small slime monster"}]
    assert plan.workstream == "simple_creature"
    assert plan.actions == ["bounce", "idle"]      # defaults filled when LLM omits
    assert plan.source == "llm:claude-opus-4-8"


def test_plan_asset_falls_back_when_llm_unavailable(capsys):
    client = MagicMock()
    client.messages.parse.side_effect = RuntimeError("no route to host")
    sys.modules.setdefault("anthropic", types.ModuleType("anthropic"))
    pytest.importorskip("pydantic")

    plan = plan_asset("a small slime monster", client=client)

    assert plan.source == "heuristic"
    assert "using keyword heuristics" in capsys.readouterr().out


def test_plan_asset_offline_never_touches_llm():
    plan = plan_asset("a knight", offline=True, client=MagicMock())
    assert plan.source == "heuristic"


# ---------------------------------------------------------------------------
# Execution (diffusion pipe mocked; everything downstream is real)
# ---------------------------------------------------------------------------


def _fake_pipe():
    """A pipe returning a deterministic 'render' per seed; no unet/vae attrs so
    enable_tiling is a no-op without real conv layers."""

    class FakePipe:
        def __call__(self, **kwargs):
            v = 60 + (self._seed * 35) % 160
            img = Image.new("RGB", (128, 128), (v, 200, 120))
            return types.SimpleNamespace(images=[img])

    pipe = FakePipe()
    pipe._seed = 0
    return pipe


@pytest.fixture
def exec_env(monkeypatch):
    fake_torch = fake_torch_module()
    pipe = _fake_pipe()

    def manual_seed(s):
        pipe._seed = s
        return fake_torch.Generator.return_value

    fake_torch.Generator.return_value.manual_seed.side_effect = manual_seed
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return pipe


def test_execute_static_prop(exec_env, tmp_path):
    pal = Palette([(0, 0, 0), (120, 200, 120), (255, 255, 255)])
    plan = Plan(workstream="static_prop", enriched_prompt="a chest", size=32)

    results = execute_plan(plan, tmp_path, palette=pal, pipe=exec_env)

    sprite = Image.open(results["sprite"])
    assert max(sprite.size) == 32
    assert opaque_colors(sprite) <= set(map(tuple, pal.colors))


def test_execute_simple_creature_full_output(exec_env, tmp_path):
    plan = Plan(workstream="simple_creature", enriched_prompt="a slime",
                size=32, actions=["bounce", "idle"])

    results = execute_plan(plan, tmp_path, pipe=exec_env)

    assert (tmp_path / "sprite.png").exists()
    assert len(list((tmp_path / "bounce").glob("*.png"))) == 12
    assert len(list((tmp_path / "idle").glob("*.png"))) == 6
    assert (tmp_path / "bounce.gif").exists()
    assert (tmp_path / "sheet.png").exists()
    assert (tmp_path / "sheet.json").exists()
    assert results["actions"] == ["bounce", "idle"]


def test_execute_creature_batch_picks_and_saves_candidates(exec_env, tmp_path):
    plan = Plan(workstream="simple_creature", enriched_prompt="a slime",
                size=32, actions=["bounce"])
    seen = {}

    def pick_second(images, prompt):
        seen["n"] = len(images)
        seen["prompt"] = prompt
        return 1

    results = execute_plan(plan, tmp_path, pipe=exec_env,
                           candidates=3, pick_fn=pick_second)

    assert seen["n"] == 3 and seen["prompt"] == "a slime"
    assert results["chosen"] == 1
    assert len(list((tmp_path / "candidates").glob("cand_*.png"))) == 3
    # the winning candidate is the one promoted to sprite.png
    import numpy as np
    won = np.asarray(Image.open(tmp_path / "candidates" / "cand_01.png"))
    assert np.array_equal(won, np.asarray(Image.open(results["sprite"])))
    assert (tmp_path / "bounce").exists()      # animation still runs on the winner


def test_execute_creature_pick_overrides_chooser(exec_env, tmp_path):
    plan = Plan(workstream="simple_creature", enriched_prompt="a slime",
                size=32, actions=[])

    def never(images, prompt):
        raise AssertionError("pick_fn must not run when --pick is given")

    results = execute_plan(plan, tmp_path, pipe=exec_env,
                           candidates=4, pick=2, pick_fn=never)
    assert results["chosen"] == 2


def test_execute_single_candidate_unchanged(exec_env, tmp_path):
    # default candidates=1 keeps the old flat layout (no candidates/ dir)
    plan = Plan(workstream="simple_creature", enriched_prompt="a slime",
                size=32, actions=[])
    results = execute_plan(plan, tmp_path, pipe=exec_env)
    assert (tmp_path / "sprite.png").exists()
    assert not (tmp_path / "candidates").exists()
    assert "chosen" not in results


def test_execute_limbed_character_heroes_and_next_steps(exec_env, tmp_path):
    plan = Plan(workstream="limbed_character", enriched_prompt="a knight", size=32)

    results = execute_plan(plan, tmp_path, seed=10, pipe=exec_env)

    assert len(results["heroes"]) == 4
    # distinct seeds -> distinct hero renders
    colors = {Image.open(p).convert("RGB").getpixel((10, 10)) for p in results["heroes"]}
    assert len(colors) > 1
    steps = (tmp_path / "NEXT_STEPS.md").read_text()
    assert "spriteforge refine" in steps and "spriteforge curate" in steps


def test_execute_environment_tile(exec_env, tmp_path):
    plan = Plan(workstream="environment_tile", enriched_prompt="grass", size=32)

    results = execute_plan(plan, tmp_path, pipe=exec_env)

    assert (tmp_path / "tile.png").exists()
    grid = Image.open(tmp_path / "tile_grid.png")
    assert grid.size == (96, 96)                       # 3x3 of the 32px tile
    assert isinstance(results["seam_error"], float)
