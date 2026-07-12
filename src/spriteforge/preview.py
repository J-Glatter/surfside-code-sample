"""Animation preview: locked frames -> a looping GIF at game speed.

Judging a cycle needs motion — a strip of stills hides jitter. Frames are
NN-upscaled (crisp pixels) and composited onto a flat background (GIF alpha is
too crude to trust for judging silhouettes).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .pixelize import upscale_preview

DEFAULT_BACKGROUND = (58, 58, 64)  # dark neutral, sprite-friendly


def make_gif(
    frames: list[Image.Image],
    out_path: str | Path,
    fps: int = 10,
    scale: int = 4,
    background: tuple[int, int, int] = DEFAULT_BACKGROUND,
) -> Path:
    """Write a looping GIF. Returns the output path."""
    if not frames:
        raise ValueError("no frames to preview")
    if fps <= 0 or scale <= 0:
        raise ValueError("fps and scale must be positive")

    rendered = []
    for frame in frames:
        frame = upscale_preview(frame.convert("RGBA"), scale)
        flat = Image.new("RGB", frame.size, background)
        flat.paste(frame, mask=frame.getchannel("A"))
        rendered.append(flat)

    out_path = Path(out_path)
    rendered[0].save(
        out_path,
        save_all=True,
        append_images=rendered[1:],
        duration=round(1000 / fps),
        loop=0,
    )
    return out_path


def gif_from_dir(
    frames_dir: str | Path,
    out_path: str | Path,
    fps: int = 10,
    scale: int = 4,
    background: tuple[int, int, int] = DEFAULT_BACKGROUND,
) -> Path:
    """GIF from a directory of frame PNGs (sorted by filename)."""
    paths = sorted(Path(frames_dir).glob("*.png"))
    if not paths:
        raise ValueError(f"no .png frames in {frames_dir}")
    return make_gif([Image.open(p) for p in paths], out_path,
                    fps=fps, scale=scale, background=background)
