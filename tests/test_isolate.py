from __future__ import annotations

import numpy as np
from PIL import Image

from spriteforge.isolate import isolate_subject


def _subject_on_bg(bg=(250, 250, 250), size=128, eye=True, noise_seed=0):
    """A green blob on a plain background; optional bg-coloured 'eye' inside
    the blob, which connectivity must protect from removal."""
    rng = np.random.default_rng(noise_seed)
    arr = np.empty((size, size, 3), dtype=np.uint8)
    arr[...] = bg
    # mild background noise, as real renders have
    arr = np.clip(arr.astype(int) + rng.integers(-3, 4, arr.shape), 0, 255)
    y, x = np.mgrid[0:size, 0:size]
    blob = (x - size / 2) ** 2 + (y - size / 2) ** 2 < (size * 0.3) ** 2
    arr[blob] = (60, 170, 80)
    if eye:
        eye_mask = (x - size / 2) ** 2 + (y - size / 2) ** 2 < (size * 0.06) ** 2
        arr[eye_mask] = bg  # background-coloured, but NOT border-connected
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def test_strips_plain_background():
    out, found = isolate_subject(_subject_on_bg())
    assert found
    alpha = np.asarray(out)[..., 3]
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0       # corners gone
    assert alpha[64, 40] == 255                           # blob kept


def test_interior_background_coloured_pixels_survive():
    out, found = isolate_subject(_subject_on_bg(eye=True))
    assert found
    alpha = np.asarray(out)[..., 3]
    assert alpha[64, 64] == 255   # the 'eye' is bg-coloured but not connected


def test_busy_scene_left_untouched():
    rng = np.random.default_rng(1)
    scene = Image.fromarray(
        rng.integers(0, 256, (128, 128, 3), dtype=np.uint8), "RGB")
    out, found = isolate_subject(scene)
    assert not found
    assert np.all(np.asarray(out)[..., 3] == 255)


def test_near_full_coverage_refused():
    # uniform image: "background" would be everything — refuse, don't eat it
    flat = Image.new("RGB", (64, 64), (240, 240, 240))
    out, found = isolate_subject(flat)
    assert not found
    assert np.all(np.asarray(out)[..., 3] == 255)


def test_deterministic():
    a, _ = isolate_subject(_subject_on_bg())
    b, _ = isolate_subject(_subject_on_bg())
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_flood_reports_method():
    _, method = isolate_subject(_subject_on_bg())
    assert method == "flood"


def test_rembg_failure_degrades_gracefully(monkeypatch):
    """A tier-2 crash (e.g. blocked weight download) must not kill the job."""
    import sys
    import types

    from spriteforge import isolate as iso

    fake = types.ModuleType("rembg")
    fake.remove = lambda img, session=None: img
    fake.new_session = lambda name: (_ for _ in ()).throw(OSError("403 blocked"))
    monkeypatch.setitem(sys.modules, "rembg", fake)
    monkeypatch.setattr(iso, "_REMBG_SESSION", None)

    rng = np.random.default_rng(3)
    busy = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
    out, method = iso.isolate_subject(busy)

    assert method is None
    assert np.all(np.asarray(out)[..., 3] == 255)   # untouched, not crashed


def test_rembg_fallback_wiring(monkeypatch):
    """Busy scene + a fake rembg module -> tier 2 is used and reported."""
    import sys
    import types

    from spriteforge import isolate as iso

    def fake_remove(img, session=None):
        arr = np.asarray(img.convert("RGBA")).copy()
        arr[:32, :, 3] = 0                      # "cut" the top half
        return Image.fromarray(arr, "RGBA")

    fake = types.ModuleType("rembg")
    fake.remove = fake_remove
    fake.new_session = lambda name: object()
    monkeypatch.setitem(sys.modules, "rembg", fake)
    monkeypatch.setattr(iso, "_REMBG_SESSION", None)

    rng = np.random.default_rng(2)
    busy = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
    out, method = iso.isolate_subject(busy)

    assert method == "rembg"
    alpha = np.asarray(out)[..., 3]
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 255
