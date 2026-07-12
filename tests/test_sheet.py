from __future__ import annotations

import json

import pytest
from PIL import Image

from spriteforge.animate.sheet import pack_sheet, save_sheet


def _frames(n, size=(16, 16), color=(200, 30, 30, 255)):
    return [Image.new("RGBA", size, color) for _ in range(n)]


def test_grid_layout_and_metadata():
    sheet, meta = pack_sheet({"walk": _frames(8), "jump": _frames(6)})
    assert sheet.size == (8 * 16, 2 * 16)
    assert meta["cell_width"] == meta["cell_height"] == 16
    assert meta["columns"] == 8 and meta["rows"] == 2
    assert meta["actions"]["walk"] == {"row": 0, "frames": 8, "fps": 10}
    assert meta["actions"]["jump"] == {"row": 1, "frames": 6, "fps": 10}


def test_frames_placed_and_short_rows_transparent():
    sheet, _ = pack_sheet({
        "walk": _frames(3, color=(255, 0, 0, 255)),
        "jump": _frames(1, color=(0, 255, 0, 255)),
    })
    assert sheet.getpixel((8, 8)) == (255, 0, 0, 255)    # walk frame 0
    assert sheet.getpixel((40, 8)) == (255, 0, 0, 255)   # walk frame 2
    assert sheet.getpixel((8, 24)) == (0, 255, 0, 255)   # jump frame 0
    assert sheet.getpixel((40, 24)) == (0, 0, 0, 0)      # jump row padding


def test_mixed_sizes_centred_in_cell():
    big = Image.new("RGBA", (16, 16), (0, 0, 255, 255))
    small = Image.new("RGBA", (8, 8), (255, 255, 0, 255))
    sheet, meta = pack_sheet({"a": [big, small]})
    assert meta["cell_width"] == 16
    assert sheet.getpixel((24, 8)) == (255, 255, 0, 255)  # small frame centred
    assert sheet.getpixel((17, 1)) == (0, 0, 0, 0)        # its corner is padding


def test_explicit_cell_too_small_rejected():
    with pytest.raises(ValueError):
        pack_sheet({"a": _frames(1, size=(32, 32))}, cell=(16, 16))


def test_empty_rejected():
    with pytest.raises(ValueError):
        pack_sheet({})
    with pytest.raises(ValueError):
        pack_sheet({"walk": []})


def test_save_sheet_writes_png_and_sidecar(tmp_path):
    out = tmp_path / "hero.png"
    meta = save_sheet({"walk": _frames(2)}, out, fps={"walk": 12})
    assert out.exists()
    sidecar = json.loads((tmp_path / "hero.json").read_text())
    assert sidecar == meta
    assert sidecar["actions"]["walk"]["fps"] == 12
