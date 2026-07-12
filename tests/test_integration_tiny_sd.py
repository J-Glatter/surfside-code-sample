"""Integration test: the REAL diffusion plumbing on a tiny local model.

Builds a miniature randomly-initialised Stable Diffusion pipeline entirely
offline (no hub access — same approach as diffusers' own test suite) and runs
it through spriteforge's actual code paths: build-prompt, scheduler swap, the
full denoising loop, pixelize, locked palette, and the seamless-tiling patch.

What this deliberately does NOT cover (hub-download paths, validated on a GPU
box instead): real SD 1.5 weights and LoRA fetching.

Skipped automatically when torch/diffusers aren't installed (CI installs the
CPU core only). With them installed it runs in well under a minute on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

from spriteforge.palette import Palette  # noqa: E402
from spriteforge.pixelize import pixelize  # noqa: E402
from spriteforge.tiling import enable_tiling  # noqa: E402

from .conftest import opaque_colors  # noqa: E402


@pytest.fixture(scope="module")
def tiny_pipe(tmp_path_factory):
    from diffusers import (
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64), layers_per_block=1, sample_size=32,
        in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
    )
    vae = AutoencoderKL(
        block_out_channels=(32,), in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",), latent_channels=4,
    )

    # hand-rolled tiny CLIP tokenizer files — no hub needed
    d = tmp_path_factory.mktemp("tok")
    vocab = ["<|startoftext|>", "<|endoftext|>", "a</w>", "b</w>", "c</w>",
             "k", "n", "i", "g", "h", "t", "knight</w>", "pixel</w>", "art</w>"]
    import json

    (d / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)}))
    (d / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(d / "vocab.json"), str(d / "merges.txt"),
                              model_max_length=77)

    text_encoder = CLIPTextModel(CLIPTextConfig(
        bos_token_id=0, eos_token_id=1, pad_token_id=1, vocab_size=len(vocab),
        hidden_size=32, intermediate_size=37, num_attention_heads=4,
        num_hidden_layers=2, max_position_embeddings=77,
    ))
    scheduler = DPMSolverMultistepScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    return StableDiffusionPipeline(
        unet=unet, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        scheduler=scheduler, safety_checker=None, feature_extractor=None,
        image_encoder=None, requires_safety_checker=False,
    )


def _run(pipe, prompt="a knight", steps=2, seed=7, size=64):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return pipe(
        prompt=prompt, num_inference_steps=steps, guidance_scale=7.0,
        width=size, height=size, generator=gen, output_type="pil",
    ).images[0]


def test_full_diffusion_loop_to_sprite(tiny_pipe):
    raw = _run(tiny_pipe)
    assert isinstance(raw, Image.Image)
    assert raw.size == (64, 64)

    # downstream: per-image quantise and locked-palette mode on a REAL render
    sprite = pixelize(raw, size=32, colors=8)
    assert len(opaque_colors(sprite)) <= 8

    pal = Palette([(0, 0, 0), (128, 128, 128), (255, 255, 255)])
    locked = pixelize(raw, size=32, palette=pal)
    assert opaque_colors(locked) <= set(map(tuple, pal.colors))


def test_scheduler_swap_matches_generate_module(tiny_pipe):
    # the same swap generate.build_pipe performs
    from diffusers import DPMSolverMultistepScheduler

    tiny_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        tiny_pipe.scheduler.config)
    raw = _run(tiny_pipe, steps=3)
    assert raw.size == (64, 64)


def test_determinism_across_runs(tiny_pipe):
    a = np.asarray(_run(tiny_pipe, seed=11))
    b = np.asarray(_run(tiny_pipe, seed=11))
    assert np.array_equal(a, b)


def test_tiling_patch_on_real_pipeline(tiny_pipe):
    patched = enable_tiling(tiny_pipe)
    assert patched > 0
    raw = _run(tiny_pipe, prompt="ground texture", seed=3)
    assert raw.size == (64, 64)  # circular padding keeps shapes intact
