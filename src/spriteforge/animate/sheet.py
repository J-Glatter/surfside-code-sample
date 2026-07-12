"""Sprite-sheet packing: locked frames -> one game-ready PNG + JSON metadata.

Layout: one row per action, fixed cell grid sized to the largest frame,
transparent padding, frames left-aligned per row. The JSON sidecar carries
everything an engine needs to slice it.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

DEFAULT_FPS = {"walk": 10, "run": 12, "jump": 10}


def pack_sheet(
    actions: dict[str, list[Image.Image]],
    cell: tuple[int, int] | None = None,
    fps: dict[str, int] | None = None,
) -> tuple[Image.Image, dict]:
    """Pack {action: frames} into (sheet image, metadata dict).

    Rows follow the dict's insertion order. `cell` defaults to the smallest
    box fitting every frame; frames are centred in their cells.
    """
    if not actions or any(len(f) == 0 for f in actions.values()):
        raise ValueError("every action needs at least one frame")

    all_frames = [f for frames in actions.values() for f in frames]
    if cell is None:
        cell = (max(f.width for f in all_frames), max(f.height for f in all_frames))
    cw, ch = cell
    if any(f.width > cw or f.height > ch for f in all_frames):
        raise ValueError(f"a frame exceeds the cell size {cell}")

    columns = max(len(f) for f in actions.values())
    sheet = Image.new("RGBA", (columns * cw, len(actions) * ch), (0, 0, 0, 0))

    meta_actions = {}
    for row, (name, frames) in enumerate(actions.items()):
        for col, frame in enumerate(frames):
            frame = frame.convert("RGBA")
            x = col * cw + (cw - frame.width) // 2
            y = row * ch + (ch - frame.height) // 2
            sheet.paste(frame, (x, y))
        meta_actions[name] = {
            "row": row,
            "frames": len(frames),
            "fps": (fps or DEFAULT_FPS).get(name, 10),
        }

    metadata = {
        "cell_width": cw,
        "cell_height": ch,
        "columns": columns,
        "rows": len(actions),
        "actions": meta_actions,
    }
    return sheet, metadata


def save_sheet(
    actions: dict[str, list[Image.Image]],
    out_png: str | Path,
    out_json: str | Path | None = None,
    cell: tuple[int, int] | None = None,
    fps: dict[str, int] | None = None,
) -> dict:
    """Pack and write the sheet PNG + JSON sidecar (defaults to <sheet>.json)."""
    out_png = Path(out_png)
    sheet, metadata = pack_sheet(actions, cell=cell, fps=fps)
    sheet.save(out_png)
    sidecar = Path(out_json) if out_json else out_png.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
