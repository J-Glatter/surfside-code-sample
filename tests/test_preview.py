from __future__ import annotations

import pytest
from PIL import Image

from spriteforge.preview import gif_from_dir, make_gif


def _frames(n=4, size=(16, 16)):
    return [Image.new("RGBA", size, (i * 40, 100, 100, 255)) for i in range(n)]


def test_make_gif_loops_at_fps(tmp_path):
    out = make_gif(_frames(4), tmp_path / "walk.gif", fps=10, scale=4)
    gif = Image.open(out)
    assert gif.format == "GIF"
    assert gif.n_frames == 4
    assert gif.size == (64, 64)                     # 16 * scale 4
    assert gif.info["duration"] == 100              # 1000/10 fps
    assert gif.info["loop"] == 0                    # loops forever


def test_transparency_composited_on_background(tmp_path):
    frame = Image.new("RGBA", (8, 8), (0, 0, 0, 0))  # fully transparent
    out = make_gif([frame], tmp_path / "t.gif", scale=1,
                   background=(10, 20, 30))
    px = Image.open(out).convert("RGB").getpixel((4, 4))
    assert px == (10, 20, 30)


def test_gif_from_dir_sorted(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i, f in enumerate(_frames(3)):
        f.save(d / f"walk_{i:02d}.png")
    out = gif_from_dir(d, tmp_path / "w.gif", fps=5, scale=2)
    gif = Image.open(out)
    assert gif.n_frames == 3
    assert gif.size == (32, 32)


def test_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        make_gif([], tmp_path / "x.gif")
    with pytest.raises(ValueError):
        make_gif(_frames(1), tmp_path / "x.gif", fps=0)
    with pytest.raises(ValueError):
        gif_from_dir(tmp_path, tmp_path / "x.gif")  # empty dir
