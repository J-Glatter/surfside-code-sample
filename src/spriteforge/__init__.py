"""spriteforge — prompt -> palette-coherent, game-ready pixel-art sprites.

Pipeline: generate (SD 1.5 + LoRA, GPU) -> pixelize (CPU, deterministic) -> serve.
See PLAN.md and reference/HANDOVER.md for the full design.
"""

from .pixelize import DEFAULT_COLORS, DEFAULT_SIZE, pixelize, upscale_preview

__version__ = "0.1.0"
__all__ = ["DEFAULT_COLORS", "DEFAULT_SIZE", "pixelize", "upscale_preview", "__version__"]
