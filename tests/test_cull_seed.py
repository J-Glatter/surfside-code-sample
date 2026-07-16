"""The seed-set auto-culler's detection logic (scripts/cull_seed.py).

Gates LoRA-training quality, so pin the two failure modes it must reject:
collages (fragmented opaque) and failed isolation (no transparent margin)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cull_seed import is_clean, largest_component_frac, metrics  # noqa: E402


def _save(tmp_path, name, arr):
    p = tmp_path / name
    Image.fromarray(arr, "RGBA").save(p)
    return p


def test_clean_single_subject_kept(tmp_path):
    a = np.zeros((64, 64, 4), np.uint8)
    y, x = np.mgrid[0:64, 0:64]
    a[(x - 32) ** 2 + (y - 32) ** 2 < 18 ** 2] = (70, 190, 195, 255)
    m = metrics(_save(tmp_path, "single.png", a))
    assert m["components"] == 1 and m["largest_frac"] == 1.0
    assert is_clean(m, 0.12, 0.55)


def test_collage_rejected(tmp_path):
    a = np.zeros((64, 64, 4), np.uint8)
    for cy in range(8, 64, 16):
        for cx in range(8, 64, 16):
            a[cy - 4:cy + 4, cx - 4:cx + 4] = (150, 110, 70, 255)
    m = metrics(_save(tmp_path, "collage.png", a))
    assert m["components"] == 16               # a grid of items
    assert m["largest_frac"] < 0.2             # opaque fragmented across them
    assert not is_clean(m, 0.12, 0.55)


def test_failed_isolation_rejected(tmp_path):
    a = np.full((64, 64, 4), 255, np.uint8)    # opaque fills the frame
    m = metrics(_save(tmp_path, "whitebg.png", a))
    assert m["transparent_frac"] == 0.0
    assert not is_clean(m, 0.12, 0.55)


def test_largest_component_frac_empty():
    assert largest_component_frac(np.zeros((8, 8), bool)) == (0.0, 0)
