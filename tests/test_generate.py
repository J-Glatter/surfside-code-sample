"""Unit tests for the generation module's pure logic, with torch/diffusers stubbed.

Real pipeline runs are validated on the GPU boxes (Checkpoint A/B) — here we pin
device selection, dtype defaults, prompt assembly, and pipeline wiring.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from spriteforge.generate import (
    DEFAULT_NEGATIVE,
    LORA_TRIGGER,
    PIXEL_LORA,
    build_prompt,
    default_fp16,
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


def _fake_diffusers() -> tuple[types.ModuleType, MagicMock]:
    mod = types.ModuleType("diffusers")
    pipe = MagicMock(name="pipe")
    pipe.to.return_value = pipe
    mod.StableDiffusionPipeline = MagicMock()
    mod.StableDiffusionPipeline.from_pretrained.return_value = pipe
    mod.DPMSolverMultistepScheduler = MagicMock()
    return mod, pipe


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


def test_build_prompt_appends_trigger():
    assert build_prompt("a knight") == f"a knight, {LORA_TRIGGER}"
    assert build_prompt("a knight", use_lora=False) == "a knight"


def _built_pipe(monkeypatch, *, cuda: bool, mps: bool = False, **kwargs):
    from spriteforge import generate as g

    fake_torch = _fake_torch(cuda=cuda, mps=mps)
    fake_diff, pipe = _fake_diffusers()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diff)
    g.build_pipe(**kwargs)
    return fake_diff, pipe


def test_build_pipe_cuda_defaults(monkeypatch):
    fake_diff, pipe = _built_pipe(monkeypatch, cuda=True)
    _, kwargs = fake_diff.StableDiffusionPipeline.from_pretrained.call_args
    assert kwargs["torch_dtype"] == "float16"          # fp16 auto-on for CUDA
    assert kwargs["safety_checker"] is None
    pipe.to.assert_called_once_with("cuda")
    pipe.enable_attention_slicing.assert_not_called()  # not needed on the 3080
    pipe.load_lora_weights.assert_called_once_with(PIXEL_LORA)


def test_build_pipe_mps_defaults(monkeypatch):
    fake_diff, pipe = _built_pipe(monkeypatch, cuda=False, mps=True)
    _, kwargs = fake_diff.StableDiffusionPipeline.from_pretrained.call_args
    assert kwargs["torch_dtype"] == "float32"          # safe default on MPS
    pipe.to.assert_called_once_with("mps")
    pipe.enable_attention_slicing.assert_called_once()


def test_build_pipe_explicit_fp16_override(monkeypatch):
    fake_diff, _ = _built_pipe(monkeypatch, cuda=False, mps=True, fp16=True)
    _, kwargs = fake_diff.StableDiffusionPipeline.from_pretrained.call_args
    assert kwargs["torch_dtype"] == "float16"


def test_build_pipe_no_lora(monkeypatch):
    _, pipe = _built_pipe(monkeypatch, cuda=True, use_lora=False)
    pipe.load_lora_weights.assert_not_called()


def test_build_pipe_survives_lora_failure(monkeypatch):
    from spriteforge import generate as g

    fake_torch = _fake_torch(cuda=True)
    fake_diff, pipe = _fake_diffusers()
    pipe.load_lora_weights.side_effect = RuntimeError("no such file")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diff)
    assert g.build_pipe() is pipe  # falls back to base model, doesn't raise


def test_generate_call_wiring(monkeypatch):
    from spriteforge import generate as g

    fake_torch = _fake_torch(cuda=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    pipe = MagicMock(name="pipe")

    g.generate(pipe, "a slime monster", seed=123)

    fake_torch.Generator.assert_called_once_with(device="cpu")
    fake_torch.Generator.return_value.manual_seed.assert_called_once_with(123)
    _, kwargs = pipe.call_args
    assert kwargs["prompt"] == f"a slime monster, {LORA_TRIGGER}"
    assert kwargs["negative_prompt"] == DEFAULT_NEGATIVE
    assert kwargs["num_inference_steps"] == 28
    assert kwargs["guidance_scale"] == 7.0
    assert kwargs["width"] == 512 and kwargs["height"] == 512


def test_generate_without_seed_passes_none(monkeypatch):
    from spriteforge import generate as g

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    pipe = MagicMock(name="pipe")
    g.generate(pipe, "a tree")
    _, kwargs = pipe.call_args
    assert kwargs["generator"] is None
