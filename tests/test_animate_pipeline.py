"""End-to-end animation loop with the diffusion pipe mocked (CPU).

Exercises: skeleton rendering -> candidate generation calls -> pixelization ->
selection -> locked frames, plus the frames-module pipeline wiring.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from spriteforge.animate.pipeline import animate_action
from spriteforge.generate import LORA_TRIGGER
from spriteforge.palette import Palette

from .conftest import opaque_colors
from .tools import fake_torch_module


def _scripted_pipe():
    """A pipe whose output colour depends on the seed, so selection is exercised:
    seeds where (seed % 10) == 0 give a consistent grey; others give loud noise."""
    pipe = MagicMock(name="controlnet_pipe")

    def run(**kwargs):
        seed = pipe._last_seed
        if seed % 10 == 0:
            v = 100 + (seed // 10_000) * 5   # smooth grey ramp across frames
            img = Image.new("RGB", (64, 64), (v, v, v))
        else:
            rng = np.random.default_rng(seed)
            img = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        result = MagicMock()
        result.images = [img]
        return result

    pipe.side_effect = lambda **kwargs: run(**kwargs)
    return pipe


def test_animate_action_end_to_end(monkeypatch, tmp_path):
    fake_torch = fake_torch_module()
    seeds = []

    def manual_seed(s):
        seeds.append(s)
        pipe._last_seed = s
        return fake_torch.Generator.return_value

    fake_torch.Generator.return_value.manual_seed.side_effect = manual_seed
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    pipe = _scripted_pipe()

    pal = Palette([(0, 0, 0), (128, 128, 128), (255, 255, 255)])
    locked, selection = animate_action(
        pipe, "walk", "a knight", size=32, palette=pal,
        frames=3, n_candidates=5, seed=0,
    )

    assert len(locked) == 3
    assert pipe.call_count == 15                       # 3 frames x 5 candidates
    assert seeds[:5] == [0, 1, 2, 3, 4]                # frame 0
    assert seeds[5] == 10_000                          # frame 1 seed block
    # the smooth grey candidates (seed % 10 == 0) should win every frame
    assert selection.indices == [0, 0, 0]
    for frame in locked:
        assert max(frame.size) == 32
        assert opaque_colors(frame) <= set(map(tuple, pal.colors))
    # prompt carries the LoRA trigger
    _, kwargs = pipe.call_args
    assert kwargs["prompt"] == f"a knight, {LORA_TRIGGER}"


def test_animate_unknown_action(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module())
    try:
        animate_action(MagicMock(), "fly", "a bird")
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "fly" in str(e)


def test_build_animation_pipe_wiring(monkeypatch):
    import types

    from spriteforge.animate import frames as frames_mod

    fake_torch = fake_torch_module(cuda=True)
    fake_diff = types.ModuleType("diffusers")
    pipe = MagicMock(name="pipe")
    pipe.to.return_value = pipe
    fake_diff.StableDiffusionControlNetPipeline = MagicMock()
    fake_diff.StableDiffusionControlNetPipeline.from_pretrained.return_value = pipe
    fake_diff.ControlNetModel = MagicMock()
    fake_diff.DPMSolverMultistepScheduler = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diff)

    out = frames_mod.build_animation_pipe(character_lora="hero.safetensors")

    assert out is pipe
    fake_diff.ControlNetModel.from_pretrained.assert_called_once()
    args, _ = fake_diff.ControlNetModel.from_pretrained.call_args
    assert args[0] == frames_mod.CONTROLNET_OPENPOSE
    # both LoRAs stacked
    names = [kw.get("adapter_name")
             for _, kw in pipe.load_lora_weights.call_args_list]
    assert names == ["pixel", "character"]
    pipe.set_adapters.assert_called_once_with(["pixel", "character"], [1.0, 1.0])
