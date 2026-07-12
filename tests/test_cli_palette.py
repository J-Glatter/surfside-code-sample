from __future__ import annotations

from PIL import Image

from spriteforge.cli import main
from spriteforge.palette import Palette

from .conftest import opaque_colors


def test_palette_extract_then_locked_pixelize(tmp_path, gradient_scene, noise_image):
    """The intended workflow: extract a palette from favourites, lock it for all."""
    ref1, ref2 = tmp_path / "ref1.png", tmp_path / "ref2.png"
    gradient_scene.save(ref1)
    noise_image.save(ref2)
    pal_path = tmp_path / "game.json"

    main(["palette", "extract", str(ref1), str(ref2), "-o", str(pal_path),
          "--colors", "8"])
    pal = Palette.load(pal_path)
    assert 1 <= len(pal) <= 8

    out1, out2 = tmp_path / "a.png", tmp_path / "b.png"
    main(["pixelize", str(ref1), "-o", str(out1), "--size", "48",
          "--palette", str(pal_path)])
    main(["pixelize", str(ref2), "-o", str(out2), "--size", "48",
          "--palette", str(pal_path)])

    palette_set = set(map(tuple, pal.colors))
    assert opaque_colors(Image.open(out1)) <= palette_set
    assert opaque_colors(Image.open(out2)) <= palette_set


def test_palette_show_with_swatch(tmp_path, capsys):
    pal_path = tmp_path / "p.hex"
    pal_path.write_text("0f380f\n9bbc0f\n")
    swatch = tmp_path / "sw.png"

    main(["palette", "show", str(pal_path), "--swatch", str(swatch), "--cell", "4"])

    out = capsys.readouterr().out
    assert "#0f380f" in out and "#9bbc0f" in out
    img = Image.open(swatch)
    assert img.size == (8, 4)  # 2 colours * cell 4 wide, cell 4 tall
