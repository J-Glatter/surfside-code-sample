"""Curation logic with injected embedders; real CLIP runs on the GPU boxes."""

from __future__ import annotations

import numpy as np
from PIL import Image

from spriteforge.curate import curate, rank_by_prompt, score_candidates


def _fixture(tmp_path, n=5):
    """Candidates + a hero, with a fake embedding space we fully control.

    Embedding design (3-dim, unit vectors):
      axis 0 = "looks like the hero", axis 1 = clean-quality, axis 2 = mess.
    Candidate i has identity 1 - i*0.2; even candidates are clean, odd are messy.
    """
    paths = []
    for i in range(n):
        p = tmp_path / f"cand_{i}.png"
        Image.new("RGBA", (8, 8), (i * 10, 0, 0, 255)).save(p)
        paths.append(p)
    hero = tmp_path / "hero.png"
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(hero)

    def image_embed(images):
        out = []
        for img in images:
            r = img.convert("RGBA").getpixel((0, 0))[0]
            if r == 255:                      # the hero
                out.append([1.0, 0.0, 0.0])
            else:
                i = r // 10
                ident = 1.0 - i * 0.2
                clean = 0.5 if i % 2 == 0 else -0.5
                v = np.array([ident, max(clean, 0), max(-clean, 0)])
                out.append(v / np.linalg.norm(v))
        return np.array(out)

    def text_embed(texts):
        # positive anchor -> clean axis, negative anchor -> mess axis
        return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    return paths, hero, image_embed, text_embed


def test_scores_shape_and_ordering(tmp_path):
    paths, hero, image_embed, text_embed = _fixture(tmp_path)
    scores = score_candidates(paths, hero, image_embed, text_embed)
    assert [s.path for s in scores] == paths
    idents = [s.identity for s in scores]
    assert idents == sorted(idents, reverse=True)  # by construction
    # even candidates positive quality margin, odd negative
    assert scores[0].quality > 0 > scores[1].quality


def test_curate_applies_both_checks(tmp_path):
    paths, hero, image_embed, text_embed = _fixture(tmp_path)
    winners = curate(paths, hero, keep=2, min_quality=0.0,
                     image_embed=image_embed, text_embed=text_embed)
    # odd (messy) candidates fail quality; best-identity clean ones win, in order
    assert [w.path.name for w in winners] == ["cand_0.png", "cand_2.png"]


def test_curate_copies_ranked_winners(tmp_path):
    paths, hero, image_embed, text_embed = _fixture(tmp_path)
    out = tmp_path / "kept"
    curate(paths, hero, out_dir=out, keep=2, min_quality=0.0,
           image_embed=image_embed, text_embed=text_embed)
    assert sorted(p.name for p in out.iterdir()) == \
        ["01_cand_0.png", "02_cand_2.png"]


def test_empty_candidates(tmp_path):
    _, hero, image_embed, text_embed = _fixture(tmp_path)
    assert curate([], hero, image_embed=image_embed, text_embed=text_embed) == []


def test_rank_by_prompt_prefers_clean_on_prompt(tmp_path):
    # no hero: rank fresh candidates by prompt alignment + clean-vs-blurry margin
    paths, _hero, image_embed, _t = _fixture(tmp_path)
    images = [Image.open(p) for p in paths]

    def text_embed(texts):
        # prompt anchor -> identity axis; positive -> clean; negative -> mess
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    ranked = rank_by_prompt(images, "a slime", image_embed, text_embed)
    order = [i for i, _ in ranked]
    assert len(order) == len(images)
    assert order[0] % 2 == 0                       # a clean candidate wins
    # every clean (even) candidate outranks every messy (odd) one
    evens = [order.index(i) for i in range(0, len(images), 2)]
    odds = [order.index(i) for i in range(1, len(images), 2)]
    assert max(evens) < min(odds)


def test_rank_by_prompt_empty():
    assert rank_by_prompt([], "x", lambda i: np.zeros((0, 3)),
                          lambda t: np.zeros((3, 3))) == []
