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
from ..pixelize import DEFAULT_COLORS, DEFAULT_SIZE, pixelize
from .frames import generate_candidates
from .selector import Selection, select_frames

# 100 candidates/frame is the handover's working number; generation is a
# fraction of a penny each, selection is what buys the smoothness.
DEFAULT_CANDIDATES = 100


def rig_for(body: str):
    """(actions, default_frames, renderer) for a body type."""
    if body == "humanoid":
        from . import skeleton as rig

        return rig.ACTIONS, rig.DEFAULT_FRAMES, rig.render_openpose
    if body == "quadruped":
        from . import skeleton_quadruped as rig

        return rig.ACTIONS, rig.DEFAULT_FRAMES, rig.render_quadruped
    raise ValueError(f"unknown body type {body!r} (have: humanoid, quadruped)")


def animate_action(
    pipe,
    action: str,
    prompt: str,
    size: int = DEFAULT_SIZE,
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
    body: str = "humanoid",
    backend=None,
    trigger: str | None = None,
    isolate: bool = True,
) -> tuple[list[Image.Image], Selection]:
    """Generate one animated action. Returns (locked pixelized frames, selection).

    Deterministic: candidate seeds are seed + frame_index * 10_000 + candidate_index.
    `body` selects the rig: humanoid (COCO-18) or quadruped (AP-10K) — pair the
    pipe with the matching ControlNet (frames.CONTROLNET_BY_BODY).
    """
    actions, default_frames, render = rig_for(body)
    if action not in actions:
        raise ValueError(
            f"unknown action {action!r} for {body} (have: {', '.join(actions)})")
    poses = actions[action](frames or default_frames[action])

    candidates_px: list[list[np.ndarray]] = []
    for k, pose in enumerate(poses):
        control = render(pose)
        raws = generate_candidates(
            pipe, control, prompt,
            n=n_candidates, steps=steps, guidance=guidance,
            controlnet_scale=controlnet_scale,
            base_seed=seed + k * 10_000, use_lora=use_lora, backend=backend,
            trigger=trigger,
        )
        if raw_dir is not None:
            d = Path(raw_dir) / f"{action}_{k:02d}"
            d.mkdir(parents=True, exist_ok=True)
            for j, raw in enumerate(raws):
                raw.save(d / f"{j:03d}.png")
        if isolate:                          # strip the background SD paints in
            from ..isolate import isolate_subject

            raws = [isolate_subject(r)[0] for r in raws]
        candidates_px.append([
            np.asarray(pixelize(raw, size=size, colors=colors, palette=palette))
            for raw in raws
        ])

    selection = select_frames(candidates_px)
    locked = [Image.fromarray(arr, "RGBA") for arr in selection.frames]
    return locked, selection
