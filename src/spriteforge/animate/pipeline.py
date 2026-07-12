"""The per-action animation loop: skeletons -> candidates -> pixelize -> select.

Orchestrates the GPU generation (frames.py) with the CPU selection
(selector.py). Candidates are pixelized as they are generated — selection
happens in pixelized space, and the raw 512px renders are discarded unless a
debug directory is given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..palette import Palette
from ..pixelize import DEFAULT_COLORS, pixelize
from .frames import generate_candidates
from .selector import Selection, select_frames
from .skeleton import ACTIONS, DEFAULT_FRAMES, render_openpose

# 100 candidates/frame is the handover's working number; generation is a
# fraction of a penny each, selection is what buys the smoothness.
DEFAULT_CANDIDATES = 100


def animate_action(
    pipe,
    action: str,
    prompt: str,
    size: int = 256,
    colors: int = DEFAULT_COLORS,
    palette: Palette | None = None,
    frames: int | None = None,
    n_candidates: int = DEFAULT_CANDIDATES,
    steps: int = 28,
    guidance: float = 7.0,
    controlnet_scale: float = 1.0,
    seed: int = 0,
    use_lora: bool = True,
    raw_dir: str | Path | None = None,
) -> tuple[list[Image.Image], Selection]:
    """Generate one animated action. Returns (locked pixelized frames, selection).

    Deterministic: candidate seeds are seed + frame_index * 10_000 + candidate_index.
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r} (have: {', '.join(ACTIONS)})")
    poses = ACTIONS[action](frames or DEFAULT_FRAMES[action])

    candidates_px: list[list[np.ndarray]] = []
    for k, pose in enumerate(poses):
        control = render_openpose(pose)
        raws = generate_candidates(
            pipe, control, prompt,
            n=n_candidates, steps=steps, guidance=guidance,
            controlnet_scale=controlnet_scale,
            base_seed=seed + k * 10_000, use_lora=use_lora,
        )
        if raw_dir is not None:
            d = Path(raw_dir) / f"{action}_{k:02d}"
            d.mkdir(parents=True, exist_ok=True)
            for j, raw in enumerate(raws):
                raw.save(d / f"{j:03d}.png")
        candidates_px.append([
            np.asarray(pixelize(raw, size=size, colors=colors, palette=palette))
            for raw in raws
        ])

    selection = select_frames(candidates_px)
    locked = [Image.fromarray(arr, "RGBA") for arr in selection.frames]
    return locked, selection
