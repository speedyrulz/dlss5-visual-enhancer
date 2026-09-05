"""Portable DLSS 5 Visual Enhancer for images and video."""

from .core.ffmpeg import probe_video
from .image import (
    ImageBatchResult,
    ImageConversionFailure,
    ImageConversionOptions,
    ImageConversionResult,
    convert_image,
    convert_images,
    probe_image,
)
from .frame_interpolation import (
    FrameInterpolationBatchResult,
    FrameInterpolationCapabilities,
    FrameInterpolationOptions,
    FrameInterpolationResult,
    interpolate_video,
    interpolate_videos,
    probe_frame_interpolation_capabilities,
)
from .video import (
    ConversionOptions,
    ConversionResult,
    VideoBatchResult,
    VideoConversionFailure,
    VideoConversionSuccess,
    convert_video,
    convert_videos,
)

__all__ = [
    "ConversionOptions",
    "ConversionResult",
    "VideoBatchResult",
    "VideoConversionFailure",
    "VideoConversionSuccess",
    "ImageBatchResult",
    "ImageConversionOptions",
    "ImageConversionResult",
    "ImageConversionFailure",
    "FrameInterpolationBatchResult",
    "FrameInterpolationCapabilities",
    "FrameInterpolationOptions",
    "FrameInterpolationResult",
    "convert_image",
    "convert_images",
    "convert_video",
    "convert_videos",
    "probe_image",
    "probe_video",
    "interpolate_video",
    "interpolate_videos",
    "probe_frame_interpolation_capabilities",
]
