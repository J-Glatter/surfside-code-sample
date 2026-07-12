"""Shared test fakes for the GPU-side modules."""

from __future__ import annotations

import types
from unittest.mock import MagicMock


def fake_torch_module(cuda: bool = False, mps: bool = False) -> types.ModuleType:
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
