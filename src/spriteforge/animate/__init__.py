"""Animation (handover §12-13): pose-controlled generation + brute-force selection.

    skeleton  — canonical walk/run/jump keypoint cycles -> OpenPose conditioning images
    frames    — SD 1.5 + ControlNet(openpose) candidate generation (GPU)
    selector  — continuity-scored brute-force frame selection (CPU)
    sheet     — pack locked frames into game-ready sprite sheets
    pipeline  — ties the loop together per action
"""

from .selector import select_frames
from .sheet import pack_sheet, save_sheet
from .skeleton import ACTIONS, Pose, render_openpose

__all__ = ["ACTIONS", "Pose", "render_openpose", "select_frames", "pack_sheet",
           "save_sheet"]
