from __future__ import annotations

import numpy as np
from PIL import Image

from spriteforge.isolate import isolate_subject, strip_cast_shadow


def _blob_with_shadow(size=64, shadow=True):
    """A saturated teal blob with (optionally) a grey cast-shadow band beneath
    it — the SDXL failure mode. Returns an already-isolated RGBA image."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    y, x = np.mgrid[0:size, 0:size]
    body = (x - size / 2) ** 2 + ((y - size * 0.42) / 1.1) ** 2 < (size * 0.30) ** 2
    arr[body] = (70, 190, 195, 255)                 # saturated teal body
    if shadow:
        band = (y >= int(size * 0.80)) & (np.abs(x - size / 2) < size * 0.28)
        arr[band] = (172, 168, 176, 255)            # desaturated grey shadow
    return Image.fromarray(arr, "RGBA")


def test_strip_cast_shadow_removes_grey_band():
    before = _blob_with_shadow()
    out, removed = strip_cast_shadow(before)
    assert removed > 0
    a = np.asarray(out)
    # the grey band is gone...
    grey_rows = a[int(64 * 0.80):]
    assert (grey_rows[..., 3] == 0).all() or (grey_rows[..., 3] > 128).sum() == 0
    # ...but the teal body is kept
    body = np.asarray(before)
    kept = (a[..., 3] > 128) & (body[..., :3].sum(axis=-1) > 300)
    assert kept.sum() > 100


def test_strip_cast_shadow_leaves_shadowless_sprite():
    before = _blob_with_shadow(shadow=False)
    out, removed = strip_cast_shadow(before)
    assert removed == 0.0
    assert np.array_equal(np.asarray(before), np.asarray(out))


def test_isolate_subject_deshadow_is_opt_in():
    # a teal blob with a grey shadow band, on a plain white background
    size = 64
    arr = np.full((size, size, 3), 250, dtype=np.uint8)
    y, x = np.mgrid[0:size, 0:size]
    body = (x - size / 2) ** 2 + ((y - size * 0.42) / 1.1) ** 2 < (size * 0.30) ** 2
    arr[body] = (70, 190, 195)
    band = (y >= int(size * 0.80)) & (np.abs(x - size / 2) < size * 0.28)
    arr[band] = (172, 168, 176)
    img = Image.fromarray(arr, "RGB")

    kept, _ = isolate_subject(img)                       # default: shadow retained
    stripped, _ = isolate_subject(img, trim_shadow=True)
    br = slice(int(size * 0.80), None)
    kept_band = (np.asarray(kept)[br, :, 3] > 128).sum()
    strip_band = (np.asarray(stripped)[br, :, 3] > 128).sum()
    assert kept_band > 0                                 # band opaque by default
    assert strip_band < kept_band                        # opt-in trim removed it


def test_strip_cast_shadow_spares_greyscale_subject():
    # a genuinely grey subject must not be eaten (body_chroma guard)
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[20:60, 20:44] = (120, 120, 120, 255)
    out, removed = strip_cast_shadow(Image.fromarray(arr, "RGBA"))
    assert removed == 0.0


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
