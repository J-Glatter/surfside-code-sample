"""Refine logic with the diffusion pipe mocked; real runs happen on the GPU boxes."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from PIL import Image

from spriteforge.generate import LORA_TRIGGER
from spriteforge.refine import DEFAULT_VARIATIONS, fit_canvas, refine

from .tools import fake_torch_module


def _mock_pipe():
    pipe = MagicMock(name="img2img_pipe")
    result = MagicMock()
    result.images = [Image.new("RGB", (512, 512), (100, 100, 100))]
    pipe.return_value = result
    return pipe


def test_fit_canvas_square_and_centred():
    tall = Image.new("RGBA", (100, 400), (10, 20, 30, 255))
    out = fit_canvas(tall, canvas=512)
    assert out.size == (512, 512)
    assert out.mode == "RGB"
    assert out.getpixel((256, 256)) == (10, 20, 30)   # subject centred
    assert out.getpixel((5, 256)) == (255, 255, 255)  # padded sides


def test_refine_grid_and_naming(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module())
    pipe = _mock_pipe()
    hero = Image.new("RGBA", (512, 512), (50, 60, 70, 255))

    written = refine(pipe, hero, "a brave knight", tmp_path / "out",
                     variations=["front view", "side view, facing left"],
                     per_variation=3, seed=42)

    assert len(written) == 6
    assert pipe.call_count == 6
    names = [p.name for p in written]
    assert names[0] == "00_front-view_00.png"
    assert names[5] == "01_side-view-facing-left_02.png"
    assert all(p.exists() for p in written)


def test_refine_prompt_assembly_and_seeds(tmp_path, monkeypatch):
    fake_torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    pipe = _mock_pipe()
    hero = Image.new("RGBA", (512, 512), (0, 0, 0, 255))

    refine(pipe, hero, "a slime", tmp_path / "out",
           variations=["back view"], per_variation=2, seed=100, strength=0.6)

    _, kwargs = pipe.call_args_list[0]
    assert kwargs["prompt"] == f"{LORA_TRIGGER}, a slime, back view"
    assert kwargs["strength"] == 0.6
    seeds = [c.args[0] for c in
             fake_torch.Generator.return_value.manual_seed.call_args_list]
    assert seeds == [100, 101]  # deterministic per (variation, index)


def test_default_variations_cover_key_angles():
    text = " ".join(DEFAULT_VARIATIONS)
    for needed in ("front", "back", "side"):
        assert needed in text
