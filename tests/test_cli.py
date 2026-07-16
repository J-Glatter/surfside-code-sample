from __future__ import annotations

import numpy as np
from PIL import Image

from spriteforge.cli import main

from .conftest import opaque_colors


def test_pixelize_subcommand(tmp_path, gradient_scene):
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    gradient_scene.save(src)

    main(["pixelize", str(src), "-o", str(out), "--size", "64", "--colors", "8",
          "--preview", "4"])

    sprite = Image.open(out)
    assert max(sprite.size) == 64
    assert len(opaque_colors(sprite)) <= 8

    preview = Image.open(tmp_path / "out_preview.png")
    assert preview.size == (sprite.width * 4, sprite.height * 4)


def test_pixelize_output_without_extension_defaults_to_png(tmp_path, gradient_scene):
    # Checkpoint A/B field bug: `-o knight_v3` (no extension) crashed in PIL
    src = tmp_path / "in.png"
    gradient_scene.save(src)

    main(["pixelize", str(src), "-o", str(tmp_path / "knight_v3")])

    assert (tmp_path / "knight_v3.png").exists()


def test_pixelize_output_to_directory(tmp_path, gradient_scene):
    src = tmp_path / "in.png"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    gradient_scene.save(src)

    main(["pixelize", str(src), "-o", str(out_dir)])

    assert (out_dir / "sprite.png").exists()


def test_pixelize_default_size_is_64(tmp_path, gradient_scene):
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    gradient_scene.save(src)

    main(["pixelize", str(src), "-o", str(out)])

    assert max(Image.open(out).size) == 64


def test_pixelize_deterministic_across_runs(tmp_path, noise_image):
    src = tmp_path / "in.png"
    noise_image.save(src)
    outs = []
    for name in ("a.png", "b.png"):
        out = tmp_path / name
        main(["pixelize", str(src), "-o", str(out), "--size", "48", "--seed", "5"])
        outs.append(np.asarray(Image.open(out)))
    assert np.array_equal(outs[0], outs[1])
