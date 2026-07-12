"""Procedural sprite animation — squash & stretch, no diffusion required.

For blobs, slimes, coins, pickups and other limbless things, runtime-style
scaling of one good sprite is how real games animate them — cheaper and
cleaner than generated frames, and immune to the temporal-consistency problem
entirely. Limbed characters still go through the skeletal pipeline.

Pure CPU / PIL. All cycles are deterministic and loop seamlessly.
"""

from __future__ import annotations

import math

from PIL import Image


def _scaled(sprite: Image.Image, sx: float, sy: float) -> Image.Image:
    return sprite.resize(
        (max(1, round(sprite.width * sx)), max(1, round(sprite.height * sy))),
        Image.NEAREST,
    )


def _stage(sprite: Image.Image, frame: Image.Image, canvas: tuple[int, int],
           altitude: float = 0.0) -> Image.Image:
    """Place a frame on a transparent canvas, bottom-anchored, centred, lifted
    by `altitude` pixels."""
    w, h = canvas
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(frame, ((w - frame.width) // 2, h - frame.height - round(altitude)))
    return out


def bounce_cycle(
    sprite: Image.Image,
    frames: int = 12,
    jump: float | None = None,
    squash: float = 0.30,
    stretch: float = 0.22,
) -> list[Image.Image]:
    """A full bounce: anticipation squash -> launch stretch -> parabolic arc ->
    round at the apex -> landing splat. `jump` is the peak height in pixels
    (default: the sprite's own height)."""
    sprite = sprite.convert("RGBA")
    jump = sprite.height if jump is None else jump
    canvas = (
        math.ceil(sprite.width * (1 + squash) + 2),
        math.ceil(sprite.height * (1 + stretch) + jump + 2),
    )

    out = []
    for k in range(frames):
        t = k / frames
        if t < 0.18:                          # anticipation: press down
            u = t / 0.18
            sx, sy = 1 + squash * u, 1 - squash * u
            alt = 0.0
        else:                                 # airborne: parabolic arc
            u = (t - 0.18) / 0.82
            alt = jump * (1 - (2 * u - 1) ** 2)
            v = abs(2 * u - 1)                # |vertical speed|: 1 launch/land, 0 apex
            sx, sy = 1 - stretch * 0.8 * v, 1 + stretch * v
            if u > 0.92:                      # touchdown splat
                sx, sy = 1 + squash * 0.9, 1 - squash * 0.9
        out.append(_stage(sprite, _scaled(sprite, sx, sy), canvas, alt))
    return out


def idle_cycle(
    sprite: Image.Image,
    frames: int = 6,
    amount: float = 0.04,
) -> list[Image.Image]:
    """Gentle breathing: slow sinusoidal squash & stretch in place."""
    sprite = sprite.convert("RGBA")
    canvas = (
        math.ceil(sprite.width * (1 + amount) + 2),
        math.ceil(sprite.height * (1 + amount) + 2),
    )
    out = []
    for k in range(frames):
        p = 2 * math.pi * k / frames
        s = amount * math.sin(p)
        out.append(_stage(sprite, _scaled(sprite, 1 - s, 1 + s), canvas))
    return out


def sway_cycle(
    sprite: Image.Image,
    frames: int = 8,
    amount: float = 0.10,
) -> list[Image.Image]:
    """Wind sway: a sinusoidal horizontal shear pivoting at the base row —
    the trunk stays planted while the top leans. `amount` is the top's peak
    lean as a fraction of sprite height. For trees, flags, plants, lanterns."""
    sprite = sprite.convert("RGBA")
    w, h = sprite.size
    reach = math.ceil(amount * (h - 1)) + 1      # how far the top can lean
    canvas = (w + 2 * reach, h)

    out = []
    for k in range(frames):
        s = amount * math.sin(2 * math.pi * k / frames)
        staged = _stage(sprite, sprite, canvas)
        # affine maps output->input: pixels at height y shift by s*(H-1-y),
        # zero at the base row, maximal at the top
        sheared = staged.transform(
            canvas, Image.AFFINE,
            (1, s, -s * (h - 1), 0, 1, 0),
            resample=Image.NEAREST,
        )
        out.append(sheared)
    return out


PROCEDURAL_ACTIONS = {
    "bounce": bounce_cycle,
    "idle": idle_cycle,
    "sway": sway_cycle,
}
