from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spriteforge.dataset import prep_dataset


def _make_images(tmp_path, n=3, size=(700, 500)):
    paths = []
    for i in range(n):
        p = tmp_path / f"src_{i}.png"
        arr = np.full((*size[::-1], 4), 255, dtype=np.uint8)
        arr[..., 0] = 40 * i
        Image.fromarray(arr, "RGBA").save(p)
        paths.append(p)
    return paths


def test_kohya_layout_and_captions(tmp_path):
    srcs = _make_images(tmp_path)
    out = tmp_path / "ds"

    train_dir = prep_dataset(srcs, out, trigger="sks_hero", repeats=12,
                             class_word="character")

    folder = train_dir / "12_sks_hero"
    assert folder.is_dir()
    pngs = sorted(folder.glob("*.png"))
    txts = sorted(folder.glob("*.txt"))
    assert len(pngs) == len(txts) == 3
    assert txts[0].read_text() == "sks_hero, character\n"
    # images capped at 512 on the longest side
    for p in pngs:
        assert max(Image.open(p).size) <= 512
    # config + notes emitted
    config = (out / "kohya_config.toml").read_text()
    assert 'resolution = "512,512"' in config
    assert "xformers = true" in config
    assert str(train_dir.resolve()) in config
    assert "sks_hero" in (out / "NOTES.md").read_text()


def test_sidecar_captions_are_appended(tmp_path):
    srcs = _make_images(tmp_path, n=2)
    srcs[0].with_suffix(".txt").write_text("front view, holding a sword\n")
    out = tmp_path / "ds"

    train_dir = prep_dataset(srcs, out, trigger="sks_hero")

    captions = sorted((train_dir / "10_sks_hero").glob("*.txt"))
    assert captions[0].read_text() == "sks_hero, front view, holding a sword\n"
    assert captions[1].read_text() == "sks_hero\n"


def test_transparency_composited_to_rgb(tmp_path):
    p = tmp_path / "sprite.png"
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[16:48, 16:48] = (200, 30, 30, 255)  # opaque square on transparent bg
    Image.fromarray(arr, "RGBA").save(p)

    train_dir = prep_dataset([p], tmp_path / "ds", trigger="t")

    img = Image.open(next((train_dir / "10_t").glob("*.png")))
    assert img.mode == "RGB"
    assert img.getpixel((0, 0)) == (255, 255, 255)   # bg composited to white
    assert img.getpixel((32, 32)) == (200, 30, 30)


def test_empty_input_rejected(tmp_path):
    with pytest.raises(ValueError):
        prep_dataset([], tmp_path / "ds", trigger="t")
