"""Auto-curation (handover §11): remove (most of) the human from the ratchet loop.

Two checks, candidates must do well on both:

  * identity — CLIP image-embedding similarity to the hero image
    ("roughly the same character")
  * quality  — CLIP text anchors: similarity to "clean pixel art" minus
    similarity to "blurry mess"

Honest limitation (per the handover): CLIP is weak on fine errors a human eye
catches instantly. This cuts the bulk of manual work; a final human glance over
the survivors is still worthwhile for the personal tool.

Embedders are injectable so scoring logic is fully testable without models;
the real CLIP embedders load lazily from transformers ([curate] extra).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CLIP_MODEL = "openai/clip-vit-base-patch32"
POSITIVE_QUALITY = "clean pixel art sprite, sharp, coherent single character"
NEGATIVE_QUALITY = "blurry, noisy, deformed, glitchy, smudged mess"

ImageEmbedFn = Callable[[list[Image.Image]], np.ndarray]  # -> (n, d), L2-normalised
TextEmbedFn = Callable[[list[str]], np.ndarray]           # -> (n, d), L2-normalised


@dataclass
class CandidateScore:
    path: Path
    identity: float  # cosine similarity to the hero image
    quality: float   # positive-anchor minus negative-anchor similarity


def clip_embedders(device: str | None = None) -> tuple[ImageEmbedFn, TextEmbedFn]:
    """Real CLIP embedders (lazy; needs the [curate] extra: torch + transformers)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if device is None:
        from .generate import pick_device

        device = pick_device()
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    def image_embed(images: list[Image.Image]) -> np.ndarray:
        inputs = processor(images=[im.convert("RGB") for im in images],
                           return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def text_embed(texts: list[str]) -> np.ndarray:
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    return image_embed, text_embed


def score_candidates(
    candidates: list[str | Path],
    hero: Image.Image | str | Path,
    image_embed: ImageEmbedFn | None = None,
    text_embed: TextEmbedFn | None = None,
) -> list[CandidateScore]:
    """Score every candidate against the hero (identity) and quality anchors."""
    if image_embed is None or text_embed is None:
        real_image, real_text = clip_embedders()
        image_embed = image_embed or real_image
        text_embed = text_embed or real_text

    paths = [Path(p) for p in candidates]
    if not paths:
        return []
    hero_img = hero if isinstance(hero, Image.Image) else Image.open(hero)

    embs = image_embed([Image.open(p) for p in paths])
    hero_emb = image_embed([hero_img])[0]
    anchors = text_embed([POSITIVE_QUALITY, NEGATIVE_QUALITY])

    identity = embs @ hero_emb
    quality = embs @ anchors[0] - embs @ anchors[1]
    return [CandidateScore(p, float(i), float(q))
            for p, i, q in zip(paths, identity, quality, strict=True)]


def curate(
    candidates: list[str | Path],
    hero: Image.Image | str | Path,
    out_dir: str | Path | None = None,
    keep: int = 10,
    min_quality: float = 0.0,
    image_embed: ImageEmbedFn | None = None,
    text_embed: TextEmbedFn | None = None,
) -> list[CandidateScore]:
    """Both checks (§11): drop candidates below the quality bar, then keep the
    top-`keep` by identity. Winners are copied to `out_dir` when given."""
    scores = score_candidates(candidates, hero, image_embed, text_embed)
    passing = [s for s in scores if s.quality >= min_quality]
    winners = sorted(passing, key=lambda s: s.identity, reverse=True)[:keep]

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, s in enumerate(winners, start=1):
            shutil.copy2(s.path, out_dir / f"{rank:02d}_{s.path.name}")
    return winners
