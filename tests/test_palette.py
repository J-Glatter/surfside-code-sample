from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spriteforge.palette import Palette
from spriteforge.pixelize import pixelize

from .conftest import opaque_colors

GAMEBOY = ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"]  # classic 4-colour palette


def test_hex_round_trip(tmp_path):
    pal = Palette([(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)], name="gb")
    path = tmp_path / "gb.hex"
    pal.save(path)
    assert path.read_text() == "0f380f\n306230\n8bac0f\n9bbc0f\n"
    assert Palette.load(path) == pal


def test_json_round_trip(tmp_path):
    pal = Palette([(255, 0, 0), (0, 0, 255)], name="duo")
    path = tmp_path / "duo.json"
    pal.save(path)
    loaded = Palette.load(path)
    assert loaded == pal
    assert loaded.name == "duo"
    assert loaded.hex_colors == ["#ff0000", "#0000ff"]


def test_png_swatch_round_trip(tmp_path):
    pal = Palette([(10, 20, 30), (200, 100, 50), (0, 255, 0)])
    path = tmp_path / "swatch.png"
    pal.save(path)
    loaded = Palette.load(path)
    assert set(map(tuple, loaded.colors)) == set(map(tuple, pal.colors))


def test_hex_parsing_tolerant(tmp_path):
    path = tmp_path / "p.hex"
    path.write_text("; a comment\n#0f380f\n\n306230\n")
    pal = Palette.load(path)
    assert pal.hex_colors == ["#0f380f", "#306230"]


def test_map_returns_only_palette_colors():
    pal = Palette([_hex(c) for c in GAMEBOY])
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 255, size=(500, 3))
    mapped = pal.map(rgb)
    palette_set = set(map(tuple, pal.colors))
    assert {tuple(px) for px in mapped} <= palette_set
    # an exact palette colour maps to itself
    assert tuple(pal.map(np.array([15.0, 56.0, 15.0]))) == (15, 56, 15)


def test_extract_deterministic_and_bounded(noise_image, gradient_scene):
    p1 = Palette.extract([noise_image, gradient_scene], colors=8, seed=1)
    p2 = Palette.extract([noise_image, gradient_scene], colors=8, seed=1)
    assert p1 == p2
    assert len(p1) <= 8
    # ordering is dark -> light
    lightness = p1.oklab[:, 0]
    assert np.all(np.diff(lightness) >= 0)


def test_extract_few_color_input():
    img = Image.new("RGBA", (32, 32), (200, 30, 30, 255))
    pal = Palette.extract([img], colors=16)
    assert len(pal) == 1


def test_pixelize_with_locked_palette(noise_image, gradient_scene):
    pal = Palette([_hex(c) for c in GAMEBOY])
    palette_set = set(map(tuple, pal.colors))
    outs = [pixelize(img, size=48, palette=pal) for img in (noise_image, gradient_scene)]
    for out in outs:
        assert opaque_colors(out) <= palette_set
    # different inputs, same colour universe — the cohesion property
    assert opaque_colors(outs[0]) | opaque_colors(outs[1]) <= palette_set


def test_pixelize_palette_overrides_colors(noise_image):
    pal = Palette([(0, 0, 0), (255, 255, 255)])
    out = pixelize(noise_image, size=48, colors=16, palette=pal)
    assert opaque_colors(out) <= {(0, 0, 0), (255, 255, 255)}


def test_empty_palette_rejected():
    with pytest.raises(ValueError):
        Palette([])


def _hex(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
