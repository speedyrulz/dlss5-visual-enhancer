from .capabilities import probe_frame_interpolation_capabilities
from .models import (
    ENGINE_CHOICES,
    FPS_CHOICES,
    FPS_RATES,
    FrameInterpolationBatchResult,
    FrameInterpolationCapabilities,
    FrameInterpolationOptions,
    FrameInterpolationResult,
    resolve_target_rate,
)
from .scheduler import choose_interpolation_plan, output_frame_count, select_grid_timestamps


def interpolate_video(*args, **kwargs):
    from .pipeline import interpolate_video as implementation

    return implementation(*args, **kwargs)


def interpolate_videos(*args, **kwargs):
    from .pipeline import interpolate_videos as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "ENGINE_CHOICES",
    "FPS_CHOICES",
    "FPS_RATES",
    "FrameInterpolationBatchResult",
    "FrameInterpolationCapabilities",
    "FrameInterpolationOptions",
    "FrameInterpolationResult",
    "choose_interpolation_plan",
    "interpolate_video",
    "interpolate_videos",
    "output_frame_count",
    "probe_frame_interpolation_capabilities",
    "resolve_target_rate",
    "select_grid_timestamps",
]
