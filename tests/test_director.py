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


def test_heuristic_defaults():
    assert heuristic_decider("grass texture").size == 128
    assert heuristic_decider("a slime").actions == ["bounce", "idle"]
    assert heuristic_decider("a knight").actions == ["walk", "run", "jump"]
    assert heuristic_decider("a chest").actions == []


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

    assert len(list((tmp_path / "sway").glob("*.png"))) == 8
    assert (tmp_path / "sway.gif").exists()
    assert (tmp_path / "sheet.png").exists()
    assert results["actions"] == ["sway"]


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
        size=256, colors=16, actions=[], reasoning="limbless blob",
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
