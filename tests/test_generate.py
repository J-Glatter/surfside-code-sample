"""Unit tests for the generation module's pure logic, with torch/diffusers stubbed.

Real pipeline runs are validated on the GPU boxes (Checkpoint A/B) — here we pin
device selection, dtype defaults, prompt assembly, and per-backend wiring.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from spriteforge.generate import (
    DEFAULT_NEGATIVE,
    SD15,
    SDXL,
    build_prompt,
    default_fp16,
    get_backend,
)


def _fake_torch(cuda: bool = False, mps: bool = False) -> types.ModuleType:
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    mod.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    mod.float16 = "float16"
    mod.float32 = "float32"
    generator = MagicMock(name="Generator_instance")
    mod.Generator = MagicMock(return_value=generator)
    return mod


def _fake_diffusers() -> tuple[types.ModuleType, MagicMock, MagicMock]:
    """Stub both the SD1.5 and SDXL pipeline classes; return (module, sd, sdxl)."""
    mod = types.ModuleType("diffusers")
    sd_pipe = MagicMock(name="sd_pipe")
    sd_pipe.to.return_value = sd_pipe
    sdxl_pipe = MagicMock(name="sdxl_pipe")
    sdxl_pipe.to.return_value = sdxl_pipe
    mod.StableDiffusionPipeline = MagicMock()
    mod.StableDiffusionPipeline.from_pretrained.return_value = sd_pipe
    mod.StableDiffusionXLPipeline = MagicMock()
    mod.StableDiffusionXLPipeline.from_pretrained.return_value = sdxl_pipe
    mod.DPMSolverMultistepScheduler = MagicMock()
    return mod, sd_pipe, sdxl_pipe


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),   # CUDA beats MPS
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_pick_device_order(monkeypatch, cuda, mps, expected):
    from spriteforge import generate as g

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=cuda, mps=mps))
    assert g.pick_device() == expected


def test_default_fp16_per_device():
    assert default_fp16("cuda") is True
    assert default_fp16("mps") is False
    assert default_fp16("cpu") is False


def test_default_backend_is_sdxl():
    assert get_backend(None) is SDXL
    assert get_backend("sd15") is SD15
    with pytest.raises(ValueError):
        get_backend("midjourney")


def test_build_prompt_appends_backend_trigger():
    # default (SDXL) trigger
    assert build_prompt("a knight") == "a knight, pixel art"
    assert build_prompt("a knight", use_lora=False) == "a knight"
    # SD1.5 keeps its PixArFK trigger
    assert build_prompt("a knight", backend="sd15") == "a knight, pixel art, PixArFK"


def _built_pipe(monkeypatch, *, cuda: bool, mps: bool = False, **kwargs):
    from spriteforge import generate as g

    fake_torch = _fake_torch(cuda=cuda, mps=mps)
    fake_diff, sd_pipe, sdxl_pipe = _fake_diffusers()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diff)
    g.build_pipe(**kwargs)
    return fake_diff, sd_pipe, sdxl_pipe


def test_build_pipe_defaults_to_sdxl(monkeypatch):
    fake_diff, _sd, sdxl = _built_pipe(monkeypatch, cuda=True)
    fake_diff.StableDiffusionPipeline.from_pretrained.assert_not_called()
    _, kwargs = fake_diff.StableDiffusionXLPipeline.from_pretrained.call_args
    assert kwargs["torch_dtype"] == "float16"          # fp16 auto-on for CUDA
    assert "safety_checker" not in kwargs              # SDXL has no such arg
    sdxl.to.assert_called_once_with("cuda")
    # SDXL LoRA loads by explicit weight filename
    sdxl.load_lora_weights.assert_called_once_with(
        SDXL.pixel_lora, weight_name=SDXL.lora_weight_name)


def test_build_pipe_sd15_backend(monkeypatch):
    fake_diff, sd, _sdxl = _built_pipe(monkeypatch, cuda=True, backend="sd15")
    fake_diff.StableDiffusionXLPipeline.from_pretrained.assert_not_called()
    _, kwargs = fake_diff.StableDiffusionPipeline.from_pretrained.call_args
    assert kwargs["safety_checker"] is None
    sd.load_lora_weights.assert_called_once_with(SD15.pixel_lora)  # autodetect file


def test_build_pipe_mps_defaults(monkeypatch):
    _fake_diff, _sd, sdxl = _built_pipe(monkeypatch, cuda=False, mps=True)
    _, kwargs = _fake_diff.StableDiffusionXLPipeline.from_pretrained.call_args
    assert kwargs["torch_dtype"] == "float32"          # safe default on MPS
    sdxl.to.assert_called_once_with("mps")
    sdxl.enable_attention_slicing.assert_called_once()


def test_build_pipe_no_lora(monkeypatch):
    _, _sd, sdxl = _built_pipe(monkeypatch, cuda=True, use_lora=False)
    sdxl.load_lora_weights.assert_not_called()


def test_build_pipe_survives_lora_failure(monkeypatch):
    from spriteforge import generate as g

    fake_torch = _fake_torch(cuda=True)
    fake_diff, _sd, sdxl = _fake_diffusers()
    sdxl.load_lora_weights.side_effect = RuntimeError("no such file")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diff)
    assert g.build_pipe() is sdxl  # falls back to base model, doesn't raise


def test_generate_call_wiring_sdxl(monkeypatch):
    from spriteforge import generate as g

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    pipe = MagicMock(name="pipe")

    g.generate(pipe, "a slime monster", seed=123)

    _, kwargs = pipe.call_args
    assert kwargs["prompt"] == "a slime monster, pixel art"   # SDXL trigger
    assert kwargs["negative_prompt"] == DEFAULT_NEGATIVE
    assert kwargs["width"] == 1024 and kwargs["height"] == 1024   # SDXL native
    assert kwargs["num_inference_steps"] == 28


def test_generate_call_wiring_sd15(monkeypatch):
    from spriteforge import generate as g

    fake_torch = _fake_torch(cuda=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    pipe = MagicMock(name="pipe")

    g.generate(pipe, "a slime monster", seed=123, backend="sd15")

    fake_torch.Generator.assert_called_once_with(device="cpu")
    fake_torch.Generator.return_value.manual_seed.assert_called_once_with(123)
    _, kwargs = pipe.call_args
    assert kwargs["prompt"] == "a slime monster, pixel art, PixArFK"
    assert kwargs["width"] == 512 and kwargs["height"] == 512


def test_generate_size_override(monkeypatch):
    from spriteforge import generate as g

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    pipe = MagicMock(name="pipe")
    g.generate(pipe, "a tree", size=768)
    _, kwargs = pipe.call_args
    assert kwargs["width"] == 768 and kwargs["height"] == 768


def test_generate_without_seed_passes_none(monkeypatch):
    from spriteforge import generate as g

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    pipe = MagicMock(name="pipe")
    g.generate(pipe, "a tree")
    _, kwargs = pipe.call_args
    assert kwargs["generator"] is None
