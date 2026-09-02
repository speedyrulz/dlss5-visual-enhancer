from __future__ import annotations

import threading
import traceback
import sys
from dataclasses import replace
from pathlib import Path

import gradio as gr
from PIL import Image

from src.images import (
    IMAGE_FORMATS,
    RAW_EXTENSIONS,
    ImageConversionOptions,
    convert_images,
    decode_image,
    take_image_preview,
)
from src.naming import RENAME_MODES
from src.ffmpeg import probe_nvenc_codecs, probe_video
from src.frame_interpolation import (
    ENGINE_CHOICES,
    FPS_CHOICES,
    FrameInterpolationOptions,
    choose_interpolation_plan,
    interpolate_video,
    interpolate_videos,
    probe_frame_interpolation_capabilities,
)
from src.runtime import (
    LOGS,
    OUTPUTS,
    cancel_active_job,
    gpu_choice_label,
    validate_gpu_runtime,
)
from src.prepare import prepare_runtime
from src.settings import (
    CODEC_CHOICES,
    CONTAINER_CHOICES,
    DEFAULT_SETTINGS,
    QUALITY_CHOICES,
    UISettings,
    export_settings_preset,
    import_settings_preset,
    load_settings,
    preset_filename,
    save_settings,
)
from src.video import (
    ConversionOptions,
    DLSS_MODEL_PRESETS,
    NR_PRESETS,
    NR_STYLES,
    UPSCALING_CHOICES,
    convert_video,
    convert_videos,
)


CONFIG_PATH = Path(__file__).resolve().with_name("config.ini")
PREVIEW_SECONDS = 3.0
AUTOMATIC_MASK_CHOICES = ("Off", "On")
_CONFIG_LOCK = threading.Lock()
_CURRENT_SETTINGS: UISettings | None = None

APP_CSS = """
/* Keep the header links identical even when one URL has been visited. */
#app-title a,
#app-title a:visited,
#app-title a:hover,
#app-title a:active {
    color: #00bfff !important;
    opacity: 1 !important;
}

/* Keep large upload batches compact: roughly three file rows, then scroll. */
#image-upload-list .file-preview-holder,
#image-output-list .file-preview-holder,
#video-upload-list .file-preview-holder,
#video-output-list .file-preview-holder,
#frame-interpolation-upload-list .file-preview-holder,
#frame-interpolation-output-list .file-preview-holder {
    max-height: 210px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    overscroll-behavior: contain;
    scrollbar-gutter: stable;
}

#image-upload-list .file-preview,
#image-output-list .file-preview,
#video-upload-list .file-preview,
#video-output-list .file-preview,
#frame-interpolation-upload-list .file-preview,
#frame-interpolation-output-list .file-preview {
    max-height: none !important;
}

/* A single input or output preview is a full 16:9 viewport without scrolling. */
#image-input-preview:has(.gallery-item:only-child),
#image-output-preview:has(.gallery-item:only-child) {
    aspect-ratio: 16 / 9;
    height: auto !important;
    min-height: 0 !important;
}

#image-input-preview:has(.gallery-item:only-child) .gallery-container,
#image-input-preview:has(.gallery-item:only-child) .grid-wrap,
#image-input-preview:has(.gallery-item:only-child) .grid-container,
#image-input-preview:has(.gallery-item:only-child) .gallery-item,
#image-input-preview:has(.gallery-item:only-child) .thumbnail-lg,
#image-output-preview:has(.gallery-item:only-child) .gallery-container,
#image-output-preview:has(.gallery-item:only-child) .grid-wrap,
#image-output-preview:has(.gallery-item:only-child) .grid-container,
#image-output-preview:has(.gallery-item:only-child) .gallery-item,
#image-output-preview:has(.gallery-item:only-child) .thumbnail-lg {
    box-sizing: border-box;
    height: 100% !important;
    min-height: 0 !important;
}

#image-input-preview:has(.gallery-item:only-child) .grid-wrap,
#image-output-preview:has(.gallery-item:only-child) .grid-wrap {
    overflow: hidden !important;
}

#image-input-preview:has(.gallery-item:only-child) .grid-container,
#image-output-preview:has(.gallery-item:only-child) .grid-container {
    grid-template-rows: minmax(0, 1fr) !important;
    grid-auto-rows: minmax(0, 1fr) !important;
}

#image-input-preview:has(.gallery-item:only-child) img,
#image-output-preview:has(.gallery-item:only-child) img {
    height: 100% !important;
    width: 100% !important;
    object-fit: contain !important;
}
"""


def _automatic_mask_choice(enabled: bool) -> str:
    return "On" if enabled else "Off"


def _parse_automatic_mask(value: str) -> bool:
    if value not in AUTOMATIC_MASK_CHOICES:
        choices = ", ".join(AUTOMATIC_MASK_CHOICES)
        raise ValueError(f"Automatic Mask must be one of: {choices}.")
    return value == "On"


def rename_suffix_update(mode: str):
    return gr.update(interactive=mode == "Custom")


def _neural_values(
    settings: UISettings,
) -> tuple[str, str, float, float, float, float, float, str]:
    return (
        settings.nr_preset,
        settings.nr_style,
        settings.nr_intensity,
        settings.local_tone_strength,
        settings.local_structure_strength,
        settings.skin_structure_strength,
        settings.upscaling_factor,
        _automatic_mask_choice(settings.automatic_mask),
    )


def _shared_dlss_values(
    settings: UISettings,
) -> tuple[str, str, float, float, float, float, float, str, str]:
    return (*_neural_values(settings), settings.dlss_model_preset)


def persist_image_settings(
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    dlss_model_preset: str,
    image_format: str,
    image_quality: float,
    rename_mode: str,
    custom_suffix: str,
) -> tuple[str, str, float, float, float, float, float, str, str]:
    global _CURRENT_SETTINGS
    with _CONFIG_LOCK:
        current = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
        settings = replace(
            current,
            nr_preset=nr_preset,
            nr_style=nr_style,
            nr_intensity=nr_intensity,
            local_tone_strength=local_tone_strength,
            local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength,
            upscaling_factor=upscaling_factor,
            automatic_mask=_parse_automatic_mask(automatic_mask),
            dlss_model_preset=dlss_model_preset,
            image_format=image_format,
            image_quality=int(image_quality),
            image_rename_mode=rename_mode,
            image_custom_suffix=custom_suffix,
        )
        if settings != current:
            save_settings(CONFIG_PATH, settings)
        _CURRENT_SETTINGS = settings
    return _shared_dlss_values(settings)


def persist_video_settings(
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    dlss_model_preset: str,
    codec: str,
    container: str,
    quality: str,
    rename_mode: str,
    custom_suffix: str,
) -> tuple[str, str, float, float, float, float, float, str, str]:
    global _CURRENT_SETTINGS
    with _CONFIG_LOCK:
        current = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
        settings = replace(
            current,
            nr_preset=nr_preset,
            nr_style=nr_style,
            nr_intensity=nr_intensity,
            local_tone_strength=local_tone_strength,
            local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength,
            upscaling_factor=upscaling_factor,
            automatic_mask=_parse_automatic_mask(automatic_mask),
            dlss_model_preset=dlss_model_preset,
            codec=codec,
            container=container,
            quality=quality,
            video_rename_mode=rename_mode,
            video_custom_suffix=custom_suffix,
        )
        if settings != current:
            save_settings(CONFIG_PATH, settings)
        _CURRENT_SETTINGS = settings
    return _shared_dlss_values(settings)


def persist_frame_interpolation_settings(
    target_fps: str,
    engine: str,
    codec: str,
    container: str,
    quality: str,
    rename_mode: str,
    custom_suffix: str,
) -> None:
    global _CURRENT_SETTINGS
    with _CONFIG_LOCK:
        current = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
        settings = replace(
            current,
            frame_interpolation_target_fps=target_fps,
            frame_interpolation_engine=engine,
            frame_interpolation_codec=codec,
            frame_interpolation_container=container,
            frame_interpolation_quality=quality,
            frame_interpolation_rename_mode=rename_mode,
            frame_interpolation_custom_suffix=custom_suffix,
        )
        if settings != current:
            save_settings(CONFIG_PATH, settings)
        _CURRENT_SETTINGS = settings


def _settings_component_values(settings: UISettings) -> tuple:
    shared = _shared_dlss_values(settings)
    return (
        *shared,
        *shared,
        settings.image_format,
        settings.image_quality,
        settings.image_rename_mode,
        gr.update(
            value=settings.image_custom_suffix,
            interactive=settings.image_rename_mode == "Custom",
        ),
        settings.codec,
        settings.container,
        settings.quality,
        settings.video_rename_mode,
        gr.update(
            value=settings.video_custom_suffix,
            interactive=settings.video_rename_mode == "Custom",
        ),
        settings.frame_interpolation_target_fps,
        settings.frame_interpolation_engine,
        settings.frame_interpolation_codec,
        settings.frame_interpolation_container,
        settings.frame_interpolation_quality,
        settings.frame_interpolation_rename_mode,
        gr.update(
            value=settings.frame_interpolation_custom_suffix,
            interactive=settings.frame_interpolation_rename_mode == "Custom",
        ),
        settings.ai_gpu_uuid,
        settings.video_gpu_uuid,
    )


def _processing_gpu_settings() -> tuple[str, str]:
    with _CONFIG_LOCK:
        settings = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
    return settings.ai_gpu_uuid, settings.video_gpu_uuid


def _gpu_choices(prepared) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    automatic = [("Automatic (first compatible)", "auto")]
    ai = []
    for gpu in prepared.gpus:
        if not gpu.get("ai_compatible") or not gpu.get("adapter_luid"):
            continue
        try:
            validate_gpu_runtime(gpu, prepared.runtime_bundle)
        except RuntimeError:
            continue
        ai.append((gpu_choice_label(gpu), str(gpu["uuid"])))
    video = [
        (gpu_choice_label(gpu), str(gpu["uuid"]))
        for gpu in prepared.gpus
        if gpu.get("cuda_ordinal") is not None
        and probe_nvenc_codecs(int(gpu["cuda_ordinal"]))
    ]
    return automatic + ai, automatic + video


def _normalize_gpu_settings(settings: UISettings, prepared) -> tuple[UISettings, str]:
    ai_choices, video_choices = _gpu_choices(prepared)
    ai_values = {value for _label, value in ai_choices}
    video_values = {value for _label, value in video_choices}
    warnings = []
    ai_uuid = settings.ai_gpu_uuid
    video_uuid = settings.video_gpu_uuid
    if ai_uuid not in ai_values:
        warnings.append("Saved AI Processing GPU is unavailable; using Automatic.")
        ai_uuid = "auto"
    if video_uuid not in video_values:
        warnings.append("Saved Video Processing GPU is unavailable; using Automatic.")
        video_uuid = "auto"
    return replace(settings, ai_gpu_uuid=ai_uuid, video_gpu_uuid=video_uuid), " ".join(warnings)


def persist_gpu_settings(ai_gpu_uuid: str, video_gpu_uuid: str) -> str:
    global _CURRENT_SETTINGS
    prepared = prepare_runtime()
    ai_choices, video_choices = _gpu_choices(prepared)
    if ai_gpu_uuid not in {value for _label, value in ai_choices}:
        raise gr.Error("Choose an available AI Processing GPU.")
    if video_gpu_uuid not in {value for _label, value in video_choices}:
        raise gr.Error("Choose an available Video Processing GPU.")
    with _CONFIG_LOCK:
        current = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
        settings = replace(
            current, ai_gpu_uuid=ai_gpu_uuid, video_gpu_uuid=video_gpu_uuid
        )
        if settings != current:
            save_settings(CONFIG_PATH, settings)
        _CURRENT_SETTINGS = settings
    ai_name = "Automatic" if ai_gpu_uuid == "auto" else next(
        label for label, value in ai_choices if value == ai_gpu_uuid
    )
    video_name = "Automatic" if video_gpu_uuid == "auto" else next(
        label for label, value in video_choices if value == video_gpu_uuid
    )
    return f"AI Processing: {ai_name}\n\nVideo Processing: {video_name}"


def reset_saved_settings() -> tuple:
    global _CURRENT_SETTINGS
    with _CONFIG_LOCK:
        save_settings(CONFIG_PATH, DEFAULT_SETTINGS)
        _CURRENT_SETTINGS = DEFAULT_SETTINGS
    message = "All Image, Video, and Frame Interpolation settings were reset to defaults."
    return (
        *_settings_component_values(DEFAULT_SETTINGS),
        message,
        message,
        message,
        "AI Processing and Video Processing GPU selections were reset to Automatic.",
    )


def settings_preset_download(name: str) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    with _CONFIG_LOCK:
        current = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
    try:
        path = export_settings_preset(name, current)
    except ValueError:
        return None
    return str(path)


def settings_preset_export_status(name: str) -> str:
    try:
        filename = preset_filename(name)
    except ValueError as exc:
        return f"Export failed: {exc}"
    return f"Exported preset {name.strip()!r} as {filename}."


def apply_settings_preset(
    uploaded_path: str | None, current_name: str
) -> tuple:
    global _CURRENT_SETTINGS
    with _CONFIG_LOCK:
        current = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
        try:
            if not uploaded_path:
                raise ValueError("Choose a JSON preset file before importing.")
            name, imported = import_settings_preset(uploaded_path, current)
            imported, gpu_warning = _normalize_gpu_settings(
                imported, prepare_runtime()
            )
            save_settings(CONFIG_PATH, imported)
        except (OSError, ValueError) as exc:
            return (
                current_name,
                *_settings_component_values(current),
                f"Import failed: {exc}",
            )
        _CURRENT_SETTINGS = imported
    return (
        name,
        *_settings_component_values(imported),
        f"Imported preset {name!r}; all tabs and saved startup settings were updated."
        + (f" {gpu_warning}" if gpu_warning else ""),
    )


def _process_video(
    input_path: str | None,
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    dlss_model_preset: str,
    codec: str,
    container: str,
    quality: str,
    progress,
    preview_seconds: float | None,
    preview_frames: int | None,
) -> tuple[str | None, str]:
    if not input_path:
        raise gr.Error("Choose a video first.")
    is_preview = preview_seconds is not None or preview_frames is not None
    ai_gpu_uuid, video_gpu_uuid = _processing_gpu_settings()
    options = ConversionOptions(
        ai_gpu_uuid=ai_gpu_uuid,
        video_gpu_uuid=video_gpu_uuid,
        nr_preset=nr_preset,
        nr_style=nr_style,
        nr_intensity=nr_intensity,
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        skin_structure_strength=skin_structure_strength,
        automatic_mask=_parse_automatic_mask(automatic_mask),
        dlss_model_preset=dlss_model_preset,
        upscaling_factor=upscaling_factor,
        codec="H.264" if is_preview else codec,
        container="MP4" if is_preview else container,
        quality=quality,
        preview_seconds=preview_seconds,
        preview_frames=preview_frames,
    )

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        result = convert_video(input_path, options, progress=report)
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc)) from exc
    output_preview = result.output_path if options.container == "MP4" else None
    source_name = Path(input_path).name
    if preview_frames is not None:
        return output_preview, (
            f"One-frame preview complete for {source_name} on {result.gpu} "
            f"in {result.elapsed_seconds:.1f}s. "
            f"DLSS {result.dlss_mode}: {result.render_width}×{result.render_height} → "
            f"{result.output_width}×{result.output_height}. Signed feature 18 confirmed."
        )
    if is_preview:
        return output_preview, (
            f"Preview complete for {source_name}: {result.frames} frames from the first "
            f"{PREVIEW_SECONDS:g} seconds processed "
            f"on {result.gpu} in {result.elapsed_seconds:.1f}s. DLSS {result.dlss_mode}: "
            f"{result.render_width}×{result.render_height} → {result.output_width}×{result.output_height}. "
            "All frames returned success with signed feature 18 confirmed."
        )
    status = (
        f"Complete: {result.frames} frames processed on {result.gpu} in {result.elapsed_seconds:.1f}s. "
        f"All {result.nr_count_evidence} frames returned success with signed feature 18 confirmed. "
        f"DLSS {result.dlss_mode}: {result.render_width}×{result.render_height} → "
        f"{result.output_width}×{result.output_height}."
    )
    if container != "MP4":
        status += f" {container} output was created successfully, but browser preview is unavailable."
    return output_preview, status


def render_video(
    input_path: str,
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    dlss_model_preset: str,
    codec: str,
    container: str,
    quality: str,
    progress=gr.Progress(track_tqdm=False),
):
    return _process_video(
        input_path, nr_preset, nr_style, nr_intensity, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask, dlss_model_preset,
        codec, container, quality,
        progress, None, None
    )


def _normalize_video_paths(paths: list[str] | str | None) -> list[str]:
    if not paths:
        return []
    return [paths] if isinstance(paths, str) else list(paths)


def first_video_path(paths: list[str] | str | None) -> str | None:
    normalized = _normalize_video_paths(paths)
    return normalized[0] if normalized else None


def update_video_preview_mode(paths: list[str] | str | None):
    normalized = _normalize_video_paths(paths)
    single = len(normalized) == 1
    input_value = normalized[0] if single else None
    return (
        gr.update(value=input_value, visible=single),
        gr.update(value=None, visible=single),
        gr.update(visible=single),
        gr.update(visible=single),
    )


def frame_interpolation_capability_text() -> str:
    ai_gpu_uuid, _video_gpu_uuid = _processing_gpu_settings()
    capabilities = probe_frame_interpolation_capabilities(ai_gpu_uuid)
    hags = "Enabled" if capabilities.hags_enabled else "Disabled"
    native = (
        f"{capabilities.native_multiplier}× "
        f"({capabilities.native_generated_frame_max} generated frame per evaluation)"
    )
    cascade = "Available" if capabilities.cascade_available else "Unavailable"
    state = "Ready" if capabilities.available else "Unavailable"
    detail = f"\n{capabilities.detail}" if capabilities.detail else ""
    return (
        f"{state} — GPU: {capabilities.gpu} | Driver: {capabilities.driver} | "
        f"HAGS: {hags}\nNative maximum: {native} | Experimental cascade: {cascade} | "
        f"DLSSG runtime: {capabilities.runtime_version}{detail}"
    )


def describe_frame_interpolation_plan(
    paths: list[str] | str | None,
    target_fps: str,
    engine: str,
) -> str:
    selected = first_video_path(paths)
    if not selected:
        return "Choose a video to preflight its DLSSG path and temporal precision."
    try:
        metadata = probe_video(selected, count_mode="metadata")
        ai_gpu_uuid, _video_gpu_uuid = _processing_gpu_settings()
        capabilities = probe_frame_interpolation_capabilities(ai_gpu_uuid)
        plan = choose_interpolation_plan(
            metadata["rate"],
            FrameInterpolationOptions(target_fps=target_fps).target_rate,
            engine,
            capabilities.native_multiplier,
            cfr=bool(metadata.get("cfr", True)),
        )
        if plan.cascade_stages:
            precision = (
                f"≤ {float(plan.maximum_temporal_error) * 1000:.3f} ms "
                f"(1/{2 * plan.grid_multiplier} source-frame interval)"
            )
        elif plan.path == "Native DLSSG":
            precision = "Exact native temporal grid"
        else:
            precision = "Nearest real source frame"
        return (
            f"{Path(selected).name}: {metadata['rate']} FPS → {target_fps} FPS | "
            f"Path: {plan.path} | Cascade stages: {plan.cascade_stages} | "
            f"Internal grid: {plan.grid_multiplier}× | Precision: {precision}"
        )
    except Exception as exc:
        return f"Preflight: {exc}"


def update_frame_interpolation_preview_mode(
    paths: list[str] | str | None,
    target_fps: str,
    engine: str,
):
    normalized = _normalize_video_paths(paths)
    single = len(normalized) == 1
    return (
        gr.update(value=normalized[0] if single else None, visible=single),
        gr.update(value=None, visible=single),
        gr.update(visible=single),
        describe_frame_interpolation_plan(paths, target_fps, engine),
    )


def render_frame_interpolation_batch(
    input_paths: list[str] | str | None,
    target_fps: str,
    engine: str,
    codec: str,
    container: str,
    quality: str,
    rename_mode: str,
    custom_suffix: str,
    progress=gr.Progress(track_tqdm=False),
):
    paths = _normalize_video_paths(input_paths)
    if not paths:
        raise gr.Error("Choose at least one video first.")
    options = FrameInterpolationOptions(
        ai_gpu_uuid=_processing_gpu_settings()[0],
        video_gpu_uuid=_processing_gpu_settings()[1],
        target_fps=target_fps,
        engine=engine,
        codec=codec,
        container=container,
        quality=quality,
        rename_mode=rename_mode,
        custom_suffix=custom_suffix,
    )

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        result = interpolate_videos(paths, options, report)
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc)) from exc
    ordered: list[tuple[int, list[str]]] = []
    for item in result.successes:
        value = item.result
        details = (
            f"{value.output_frames} frames; {value.selected_path}; "
            f"native {value.native_multiplier}×; cascade stages {value.cascade_stages}; "
            f"copied {value.copied_frames}, DLSSG {value.generated_frames}, "
            f"cuts {value.scene_cuts}; report: {value.report_path}"
        )
        ordered.append(
            (item.index, [Path(item.input_path).name, "Complete", Path(value.output_path).name, details])
        )
    for item in result.failures:
        state = "Skipped" if item.error == "Cancelled before rendering." else (
            "Cancelled" if item.cancelled else "Failed"
        )
        ordered.append((item.index, [Path(item.input_path).name, state, "", item.error]))
    rows = [row for _index, row in sorted(ordered, key=lambda entry: entry[0])]
    files = [item.result.output_path for item in result.successes]
    preview = None
    if len(paths) == 1 and result.successes and Path(files[0]).suffix.lower() == ".mp4":
        preview = files[0]
    status = (
        f"{'Cancelled' if result.cancelled else 'Complete'}: "
        f"{len(result.successes)} completed, {len(result.failures)} failed/cancelled. "
        f"Batch manifest: {result.manifest_path}"
    )
    return gr.update(value=preview, visible=len(paths) == 1), files, rows, status


def preview_frame_interpolation(
    input_paths: list[str] | str | None,
    target_fps: str,
    engine: str,
    codec: str,
    container: str,
    quality: str,
    progress=gr.Progress(track_tqdm=False),
):
    selected = first_video_path(input_paths)
    if not selected:
        raise gr.Error("Choose one video first.")
    options = FrameInterpolationOptions(
        ai_gpu_uuid=_processing_gpu_settings()[0],
        video_gpu_uuid=_processing_gpu_settings()[1],
        target_fps=target_fps,
        engine=engine,
        codec="H.264",
        container="MP4",
        quality=quality,
        preview_seconds=PREVIEW_SECONDS,
    )
    try:
        result = interpolate_video(
            selected,
            options,
            lambda value, message: progress(value, desc=message),
        )
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc)) from exc
    return result.output_path, (
        f"Preview complete: {result.output_frames} frames, {result.selected_path}, "
        f"{result.cascade_stages} cascade stage(s), {result.generated_frames} DLSSG-selected "
        f"frames in {result.elapsed_seconds:.1f}s. Report: {result.report_path}"
    )


def render_video_batch(
    input_paths: list[str] | str | None,
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    dlss_model_preset: str,
    codec: str,
    container: str,
    quality: str,
    rename_mode: str,
    custom_suffix: str,
    progress=gr.Progress(track_tqdm=False),
) -> tuple[object, list[str], list[list[str]], str]:
    paths = _normalize_video_paths(input_paths)
    if not paths:
        raise gr.Error("Choose at least one video first.")
    options = ConversionOptions(
        ai_gpu_uuid=_processing_gpu_settings()[0],
        video_gpu_uuid=_processing_gpu_settings()[1],
        nr_preset=nr_preset,
        nr_style=nr_style,
        nr_intensity=nr_intensity,
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        skin_structure_strength=skin_structure_strength,
        automatic_mask=_parse_automatic_mask(automatic_mask),
        dlss_model_preset=dlss_model_preset,
        upscaling_factor=upscaling_factor,
        codec=codec,
        container=container,
        quality=quality,
        rename_mode=rename_mode,
        custom_suffix=custom_suffix,
    )

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        result = convert_videos(paths, options, progress=report)
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc)) from exc

    ordered_rows: list[tuple[int, list[str]]] = []
    for item in result.successes:
        conversion = item.result
        details = (
            f"{conversion.frames} frames in {conversion.elapsed_seconds:.1f}s; "
            f"DLSS {conversion.dlss_mode}: "
            f"{conversion.render_width}×{conversion.render_height} → "
            f"{conversion.output_width}×{conversion.output_height}; "
            f"report: {conversion.report_path}"
        )
        ordered_rows.append(
            (
                item.index,
                [
                    Path(item.input_path).name,
                    "Complete",
                    Path(conversion.output_path).name,
                    details,
                ],
            )
        )
    for item in result.failures:
        state = "Skipped" if item.error == "Cancelled before rendering." else (
            "Cancelled" if item.cancelled else "Failed"
        )
        ordered_rows.append(
            (item.index, [Path(item.input_path).name, state, "", item.error])
        )
    rows = [row for _index, row in sorted(ordered_rows, key=lambda entry: entry[0])]
    files = [item.result.output_path for item in result.successes]
    output_preview = None
    if len(paths) == 1 and result.successes:
        candidate = result.successes[0].result.output_path
        if Path(candidate).suffix.lower() == ".mp4":
            output_preview = candidate
    failed_count = sum(not item.cancelled for item in result.failures)
    cancelled_count = sum(
        item.cancelled and item.error != "Cancelled before rendering."
        for item in result.failures
    )
    skipped_count = sum(
        item.error == "Cancelled before rendering." for item in result.failures
    )
    state = "Cancelled" if result.cancelled else "Complete"
    status = (
        f"{state}: {len(result.successes)} completed, {failed_count} failed, "
        f"{cancelled_count} cancelled, {skipped_count} skipped. "
        f"Batch manifest: {result.manifest_path}"
    )
    return gr.update(value=output_preview, visible=len(paths) == 1), files, rows, status


def preview_video(
    input_path: list[str] | str | None,
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    dlss_model_preset: str,
    codec: str,
    container: str,
    quality: str,
    progress=gr.Progress(track_tqdm=False),
):
    selected = first_video_path(input_path)
    return _process_video(
        selected, nr_preset, nr_style, nr_intensity, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask, dlss_model_preset,
        codec, container, quality,
        progress, PREVIEW_SECONDS, None
    )


def preview_one_frame(
    input_path: list[str] | str | None,
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    dlss_model_preset: str,
    codec: str,
    container: str,
    quality: str,
    progress=gr.Progress(track_tqdm=False),
):
    selected = first_video_path(input_path)
    return _process_video(
        selected, nr_preset, nr_style, nr_intensity, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask, dlss_model_preset,
        codec, container, quality,
        progress, None, 1
    )


def preview_input_images(paths: list[str] | str | None):
    if not paths:
        return []
    if isinstance(paths, str):
        paths = [paths]
    previews = []
    for raw_path in paths:
        try:
            decoded = decode_image(raw_path)
            image = Image.fromarray(decoded.rgba, mode="RGBA")
            image.thumbnail((1200, 900), Image.Resampling.LANCZOS)
            previews.append((image, Path(raw_path).name))
        except Exception:
            continue
    return previews


def render_image_batch(
    input_paths: list[str] | str | None,
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    dlss_model_preset: str,
    image_format: str,
    image_quality: float,
    rename_mode: str,
    custom_suffix: str,
    progress=gr.Progress(track_tqdm=False),
):
    if not input_paths:
        raise gr.Error("Choose at least one image first.")
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    options = ImageConversionOptions(
        ai_gpu_uuid=_processing_gpu_settings()[0],
        nr_preset=nr_preset,
        nr_style=nr_style,
        nr_intensity=nr_intensity,
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        skin_structure_strength=skin_structure_strength,
        automatic_mask=_parse_automatic_mask(automatic_mask),
        dlss_model_preset=dlss_model_preset,
        upscaling_factor=upscaling_factor,
        output_format=image_format,
        quality=int(image_quality),
        rename_mode=rename_mode,
        custom_suffix=custom_suffix,
    )

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        result = convert_images(input_paths, options, progress=report)
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc)) from exc

    gallery = []
    for item in result.successes:
        preview = take_image_preview(item.output_path)
        if preview is None:
            with Image.open(item.output_path) as output:
                preview = output.convert("RGBA")
                preview.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
                preview = preview.copy()
        gallery.append((preview, Path(item.output_path).name))
    files = [item.output_path for item in result.successes]
    rows = [
        [Path(item.input_path).name, "Complete", Path(item.output_path).name, "; ".join(item.warnings)]
        for item in result.successes
    ]
    rows.extend(
        [Path(item.input_path).name, "Failed", "", item.error] for item in result.failures
    )
    state = "Cancelled" if result.cancelled else "Complete"
    status = (
        f"{state}: {len(result.successes)} image(s) rendered, {len(result.failures)} failed. "
        "Every successful output returned feature-18 success and has a diagnostic report."
    )
    return gallery, files, result.zip_path, rows, status


def build_neural_controls(settings: UISettings):
    nr_preset = gr.Dropdown(
        list(NR_PRESETS), value=settings.nr_preset, label="NR Preset",
        info="Experimental content-dependent neural-rendering model hint. Default is recommended."
    )
    nr_style = gr.Radio(
        list(NR_STYLES), value=settings.nr_style, label="NR Style",
        info="Selects the native neural-rendering style."
    )
    upscaling_factor = gr.Dropdown(
        choices=list(UPSCALING_CHOICES), value=settings.upscaling_factor,
        label="Upscaling factor", info="Uses NVIDIA's fixed DLSS modes. DLAA keeps source resolution."
    )
    with gr.Row():
        nr_intensity = gr.Slider(
            0.0, 2.0, value=settings.nr_intensity, step=0.05, precision=2,
            label="NR Intensity", info="Overall neural-rendering strength.", buttons=["reset"]
        )
        local_tone_strength = gr.Slider(
            0.0, 2.0, value=settings.local_tone_strength, step=0.05, precision=2,
            label="Local Tone Strength", info="Local tone and contrast enhancement.", buttons=["reset"]
        )
    with gr.Row():
        local_structure_strength = gr.Slider(
            0.0, 2.0, value=settings.local_structure_strength, step=0.05, precision=2,
            label="Local Structure Strength", info="Local detail and texture structure.", buttons=["reset"]
        )
        skin_structure_strength = gr.Slider(
            -1.0, 2.0, value=settings.skin_structure_strength, step=0.05, precision=2,
            label="Skin Structure Strength", info="Skin-specific structure; -1.00 is the native default.",
            buttons=["reset"]
        )
    automatic_mask = gr.Radio(
        choices=AUTOMATIC_MASK_CHOICES,
        value=_automatic_mask_choice(settings.automatic_mask),
        label="Automatic Mask",
        info=(
            "Experimental runtime-generated mask that changes where Neural Rendering is "
            "applied; it may cause flicker or inconsistent results."
        ),
    )
    return [
        nr_preset, nr_style, nr_intensity, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask
    ]


def build_dlss_model_control(settings: UISettings):
    return gr.Dropdown(
        choices=list(DLSS_MODEL_PRESETS),
        value=settings.dlss_model_preset,
        label="DLSS Model Preset",
        info=(
            "Default lets NVIDIA select its normal mode-specific presets. "
            "J, K, L, or M forces that model preset for every DLSS scaling mode."
        ),
    )


def build_app() -> gr.Blocks:
    """Build the UI from the cached settings without rewriting configuration."""
    global _CURRENT_SETTINGS
    prepared = prepare_runtime()
    settings = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
    settings, gpu_warning = _normalize_gpu_settings(settings, prepared)
    if settings != (_CURRENT_SETTINGS or load_settings(CONFIG_PATH)):
        save_settings(CONFIG_PATH, settings)
    _CURRENT_SETTINGS = settings
    ai_gpu_choices, video_gpu_choices = _gpu_choices(prepared)
    upload_types = ["image", ".svg", ".heic", ".heif", *sorted(RAW_EXTENSIONS)]

    with gr.Blocks(title="DLSS 5 Visual Enhancer") as demo:
        gr.Markdown(
            "# DLSS 5 Visual Enhancer\n"
            "[Support on Patreon](https://www.patreon.com/MM744) | "
            "[GitHub](https://github.com/Merserk/dlss5-visual-enhancer)",
            elem_id="app-title",
        )
        with gr.Tabs(selected="image"):
            with gr.Tab("Image", id="image"):
                with gr.Row():
                    with gr.Column(scale=3):
                        image_sources = gr.File(
                            label="Input image(s)",
                            file_count="multiple",
                            file_types=upload_types,
                            type="filepath",
                            allow_reordering=True,
                            elem_id="image-upload-list",
                        )
                        image_input_gallery = gr.Gallery(
                            label="Input preview",
                            columns=3,
                            height=320,
                            object_fit="contain",
                            interactive=False,
                            buttons=["fullscreen"],
                            elem_id="image-input-preview",
                        )
                        with gr.Accordion(
                            "DLSS 5 Neural Rendering Settings", open=True
                        ):
                            image_neural = build_neural_controls(settings)
                        with gr.Accordion("DLSS 5 Settings", open=True):
                            image_model_preset = build_dlss_model_control(settings)
                        with gr.Row():
                            image_format = gr.Dropdown(
                                list(IMAGE_FORMATS),
                                value=settings.image_format,
                                label="Output format",
                                info=(
                                    "PNG and TIFF are lossless; JPEG composites transparency "
                                    "over white."
                                ),
                            )
                            image_quality = gr.Slider(
                                1,
                                100,
                                value=settings.image_quality,
                                step=1,
                                precision=0,
                                label="Lossy quality",
                                info=(
                                    "Used by JPEG, WebP, and AVIF; ignored by PNG/TIFF."
                                ),
                            )
                        with gr.Row():
                            image_rename_mode = gr.Radio(
                                RENAME_MODES,
                                value=settings.image_rename_mode,
                                label="Rename",
                                info=(
                                    "Auto adds the current DLSS5 timestamp; Copy keeps the "
                                    "original base name; Custom appends your suffix."
                                ),
                            )
                            image_custom_suffix = gr.Textbox(
                                value=settings.image_custom_suffix,
                                label="Custom suffix",
                                placeholder="_DLSS5",
                                interactive=settings.image_rename_mode == "Custom",
                            )
                        gr.Markdown(
                            "Images are processed as 8-bit SDR sRGB. Animated and multipage "
                            "files use their first frame/page."
                        )
                        with gr.Row():
                            image_render = gr.Button("Render image(s)", variant="primary")
                            image_stop = gr.Button("Stop", variant="stop")
                            image_reset = gr.Button("Reset settings")
                        image_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=3):
                        image_output_gallery = gr.Gallery(
                            label="Enhanced previews",
                            columns=2,
                            height=520,
                            object_fit="contain",
                            interactive=False,
                            buttons=["download", "download_all", "fullscreen"],
                            elem_id="image-output-preview",
                        )
                        image_output_files = gr.File(
                            label="Rendered image files",
                            file_count="multiple",
                            interactive=False,
                            elem_id="image-output-list",
                        )
                        image_zip = gr.DownloadButton(
                            "Download successful images as ZIP"
                        )
                        image_results = gr.Dataframe(
                            headers=["Input", "Result", "Output", "Details"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            label="Batch results",
                            wrap=True,
                        )

            with gr.Tab("Video", id="video"):
                with gr.Row():
                    with gr.Column(scale=3):
                        video_sources = gr.File(
                            label="Input video(s)",
                            file_count="multiple",
                            file_types=["video"],
                            type="filepath",
                            allow_reordering=True,
                            elem_id="video-upload-list",
                        )
                        video_input_preview = gr.Video(
                            label="Input video preview",
                            interactive=False,
                            visible=False,
                        )
                        with gr.Accordion(
                            "DLSS 5 Neural Rendering Settings", open=True
                        ):
                            video_neural = build_neural_controls(settings)
                        with gr.Accordion("DLSS 5 Settings", open=True):
                            video_model_preset = build_dlss_model_control(settings)
                        video_quality = gr.Radio(
                            QUALITY_CHOICES,
                            value=settings.quality,
                            label="Encoding quality",
                            info=(
                                "Auto uses output resolution, FPS, and codec. Good = Auto×2, "
                                "Best = Auto×4, Max = CQ 0."
                            ),
                        )
                        with gr.Row():
                            video_codec = gr.Dropdown(
                                CODEC_CHOICES,
                                value=settings.codec,
                                label="Video codec",
                                info=(
                                    "ProRes Proxy uses 10-bit 4:2:2 and requires MOV or MKV."
                                ),
                            )
                            video_container = gr.Dropdown(
                                CONTAINER_CHOICES,
                                value=settings.container,
                                label="Container",
                            )
                        with gr.Row():
                            video_rename_mode = gr.Radio(
                                RENAME_MODES,
                                value=settings.video_rename_mode,
                                label="Rename",
                                info=(
                                    "Auto adds the current DLSS5 timestamp; Copy keeps the "
                                    "original base name; Custom appends your suffix."
                                ),
                            )
                            video_custom_suffix = gr.Textbox(
                                value=settings.video_custom_suffix,
                                label="Custom suffix",
                                placeholder="_DLSS5",
                                interactive=settings.video_rename_mode == "Custom",
                            )
                        gr.Checkbox(
                            value=False,
                            interactive=False,
                            label=(
                                "Preserve HDR (disabled: verified feature-18 path is RGBA8; "
                                "HDR safely outputs as SDR)"
                            ),
                        )
                        with gr.Row():
                            video_preview_frame = gr.Button(
                                "Preview 1 frame", visible=False
                            )
                            video_preview = gr.Button("Preview 3 sec", visible=False)
                            video_render = gr.Button("Render video(s)", variant="primary")
                            video_stop = gr.Button("Stop", variant="stop")
                            video_reset = gr.Button("Reset settings")
                        video_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=3):
                        output_video = gr.Video(
                            label="Output video",
                            interactive=False,
                            visible=False,
                        )
                        video_output_files = gr.File(
                            label="Rendered video files",
                            file_count="multiple",
                            interactive=False,
                            elem_id="video-output-list",
                        )
                        video_results = gr.Dataframe(
                            headers=["Input", "Result", "Output", "Details"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            label="Batch results",
                            wrap=True,
                        )

            with gr.Tab("Frame Interpolation", id="frame-interpolation"):
                with gr.Row():
                    with gr.Column(scale=3):
                        frame_interpolation_sources = gr.File(
                            label="Input video(s)",
                            file_count="multiple",
                            file_types=["video"],
                            type="filepath",
                            allow_reordering=True,
                            elem_id="frame-interpolation-upload-list",
                        )
                        frame_interpolation_input_preview = gr.Video(
                            label="Input video preview",
                            interactive=False,
                            visible=False,
                        )
                        with gr.Accordion("DLSS Frame Generation Settings", open=True):
                            with gr.Row():
                                frame_interpolation_target_fps = gr.Dropdown(
                                    FPS_CHOICES,
                                    value=settings.frame_interpolation_target_fps,
                                    label="Output FPS",
                                    info="Fractional choices use exact 1001-based rates.",
                                )
                                frame_interpolation_engine = gr.Radio(
                                    ENGINE_CHOICES,
                                    value=settings.frame_interpolation_engine,
                                    label="DLSS engine",
                                    info=(
                                        "Auto uses a supported exact native grid, then the "
                                        "2× cascade when required."
                                    ),
                                )
                            frame_interpolation_capabilities = gr.Textbox(
                                value=frame_interpolation_capability_text(),
                                label="GPU / HAGS / driver / DLSSG capability",
                                interactive=False,
                                lines=3,
                            )
                            frame_interpolation_plan = gr.Textbox(
                                value=(
                                    "Choose a video to preflight its DLSSG path and temporal "
                                    "precision."
                                ),
                                label="Per-file interpolation plan",
                                interactive=False,
                                lines=2,
                            )
                        frame_interpolation_quality = gr.Radio(
                            QUALITY_CHOICES,
                            value=settings.frame_interpolation_quality,
                            label="Encoding quality",
                            info="Auto uses output resolution, selected FPS, and codec.",
                        )
                        with gr.Row():
                            frame_interpolation_codec = gr.Dropdown(
                                CODEC_CHOICES,
                                value=settings.frame_interpolation_codec,
                                label="Video codec",
                            )
                            frame_interpolation_container = gr.Dropdown(
                                CONTAINER_CHOICES,
                                value=settings.frame_interpolation_container,
                                label="Container",
                            )
                        with gr.Row():
                            frame_interpolation_rename_mode = gr.Radio(
                                RENAME_MODES,
                                value=settings.frame_interpolation_rename_mode,
                                label="Rename",
                                info=(
                                    "Auto adds a DLSSFG timestamp; Copy keeps the original base "
                                    "name; Custom appends your suffix."
                                ),
                            )
                            frame_interpolation_custom_suffix = gr.Textbox(
                                value=settings.frame_interpolation_custom_suffix,
                                label="Custom suffix",
                                placeholder="_DLSSFG",
                                interactive=(
                                    settings.frame_interpolation_rename_mode == "Custom"
                                ),
                            )
                        gr.Markdown(
                            "Pure NVIDIA DLSSG: CPU optical flow supplies guide vectors only; "
                            "every invented image is returned by signed DLSSG 310.7. HDR "
                            "PQ/HLG input is rejected in v1. Maximum output/source rate is 6×."
                        )
                        with gr.Row():
                            frame_interpolation_preview = gr.Button(
                                "Preview 3 sec", visible=False
                            )
                            frame_interpolation_render = gr.Button(
                                "Interpolate video(s)", variant="primary"
                            )
                            frame_interpolation_stop = gr.Button("Stop", variant="stop")
                            frame_interpolation_reset = gr.Button("Reset settings")
                        frame_interpolation_status = gr.Textbox(
                            label="Status", interactive=False
                        )
                    with gr.Column(scale=3):
                        frame_interpolation_output_video = gr.Video(
                            label="Interpolated output",
                            interactive=False,
                            visible=False,
                        )
                        frame_interpolation_output_files = gr.File(
                            label="Interpolated video files",
                            file_count="multiple",
                            interactive=False,
                            elem_id="frame-interpolation-output-list",
                        )
                        frame_interpolation_results = gr.Dataframe(
                            headers=["Input", "Result", "Output", "Details"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            label="Batch results",
                            wrap=True,
                        )

            with gr.Tab("Settings", id="settings"):
                gr.Markdown(
                    "## GPU Selection\n"
                    "AI Processing controls DLSS and Frame Generation. Video Processing "
                    "controls FFmpeg NVENC only; ProRes and mux-only work remain CPU-based."
                )
                with gr.Row():
                    ai_gpu_selector = gr.Dropdown(
                        choices=ai_gpu_choices,
                        value=settings.ai_gpu_uuid,
                        label="AI Processing GPU",
                        info="Used for every DLSS Neural Rendering and Frame Generation operation.",
                    )
                    video_gpu_selector = gr.Dropdown(
                        choices=video_gpu_choices,
                        value=settings.video_gpu_uuid,
                        label="Video Processing GPU",
                        info="Used for FFmpeg H.264, HEVC, and AV1 NVENC encoding only.",
                    )
                gpu_status = gr.Markdown(
                    gpu_warning or "GPU selections are ready.",
                    elem_id="gpu-selection-status",
                )
                gr.Markdown(
                    "## Settings Presets\n"
                    "Export every adjustable option from the Image, Video, and Frame "
                    "Interpolation tabs, or import a preset to apply it everywhere."
                )
                with gr.Row():
                    preset_name = gr.Textbox(
                        label="Preset name",
                        placeholder="Cinematic 2",
                        info="The exported file uses this name; the original name is stored in JSON.",
                        scale=3,
                    )
                    preset_export = gr.DownloadButton(
                        "Export preset",
                        value=settings_preset_download,
                        inputs=preset_name,
                        variant="primary",
                        scale=1,
                    )
                    preset_import = gr.UploadButton(
                        "Import preset",
                        file_count="single",
                        file_types=[".json"],
                        type="filepath",
                        variant="primary",
                        scale=1,
                    )
                preset_status = gr.Markdown(
                    "",
                    elem_id="preset-status",
                )

        image_inputs = [
            image_sources,
            *image_neural,
            image_model_preset,
            image_format,
            image_quality,
            image_rename_mode,
            image_custom_suffix,
        ]
        video_inputs = [
            video_sources,
            *video_neural,
            video_model_preset,
            video_codec,
            video_container,
            video_quality,
            video_rename_mode,
            video_custom_suffix,
        ]
        video_preview_inputs = [
            video_sources,
            *video_neural,
            video_model_preset,
            video_codec,
            video_container,
            video_quality,
        ]
        image_settings_inputs = [
            *image_neural,
            image_model_preset,
            image_format,
            image_quality,
            image_rename_mode,
            image_custom_suffix,
        ]
        video_settings_inputs = [
            *video_neural,
            video_model_preset,
            video_codec,
            video_container,
            video_quality,
            video_rename_mode,
            video_custom_suffix,
        ]
        frame_interpolation_inputs = [
            frame_interpolation_sources,
            frame_interpolation_target_fps,
            frame_interpolation_engine,
            frame_interpolation_codec,
            frame_interpolation_container,
            frame_interpolation_quality,
            frame_interpolation_rename_mode,
            frame_interpolation_custom_suffix,
        ]
        frame_interpolation_preview_inputs = [
            frame_interpolation_sources,
            frame_interpolation_target_fps,
            frame_interpolation_engine,
            frame_interpolation_codec,
            frame_interpolation_container,
            frame_interpolation_quality,
        ]
        frame_interpolation_settings_inputs = [
            frame_interpolation_target_fps,
            frame_interpolation_engine,
            frame_interpolation_codec,
            frame_interpolation_container,
            frame_interpolation_quality,
            frame_interpolation_rename_mode,
            frame_interpolation_custom_suffix,
        ]
        settings_component_outputs = [
            *image_neural,
            image_model_preset,
            *video_neural,
            video_model_preset,
            image_format,
            image_quality,
            image_rename_mode,
            image_custom_suffix,
            video_codec,
            video_container,
            video_quality,
            video_rename_mode,
            video_custom_suffix,
            frame_interpolation_target_fps,
            frame_interpolation_engine,
            frame_interpolation_codec,
            frame_interpolation_container,
            frame_interpolation_quality,
            frame_interpolation_rename_mode,
            frame_interpolation_custom_suffix,
            ai_gpu_selector,
            video_gpu_selector,
        ]

        image_sources.change(
            preview_input_images,
            inputs=image_sources,
            outputs=image_input_gallery,
            queue=False,
            show_progress="hidden",
        )
        video_sources.change(
            update_video_preview_mode,
            inputs=video_sources,
            outputs=[
                video_input_preview,
                output_video,
                video_preview_frame,
                video_preview,
            ],
            queue=False,
            show_progress="hidden",
        )
        frame_interpolation_sources.change(
            update_frame_interpolation_preview_mode,
            inputs=[
                frame_interpolation_sources,
                frame_interpolation_target_fps,
                frame_interpolation_engine,
            ],
            outputs=[
                frame_interpolation_input_preview,
                frame_interpolation_output_video,
                frame_interpolation_preview,
                frame_interpolation_plan,
            ],
            queue=False,
            show_progress="hidden",
        )
        image_rename_mode.change(
            rename_suffix_update,
            inputs=image_rename_mode,
            outputs=image_custom_suffix,
            queue=False,
        )
        video_rename_mode.change(
            rename_suffix_update,
            inputs=video_rename_mode,
            outputs=video_custom_suffix,
            queue=False,
        )
        frame_interpolation_rename_mode.change(
            rename_suffix_update,
            inputs=frame_interpolation_rename_mode,
            outputs=frame_interpolation_custom_suffix,
            queue=False,
        )
        for component in [frame_interpolation_target_fps, frame_interpolation_engine]:
            component.change(
                describe_frame_interpolation_plan,
                inputs=[
                    frame_interpolation_sources,
                    frame_interpolation_target_fps,
                    frame_interpolation_engine,
                ],
                outputs=frame_interpolation_plan,
                queue=False,
                show_progress="hidden",
            )
        for component in image_settings_inputs:
            component.input(
                persist_image_settings,
                inputs=image_settings_inputs,
                outputs=[*video_neural, video_model_preset],
                queue=False,
            )
        for component in video_settings_inputs:
            component.input(
                persist_video_settings,
                inputs=video_settings_inputs,
                outputs=[*image_neural, image_model_preset],
                queue=False,
            )
        for component in frame_interpolation_settings_inputs:
            component.input(
                persist_frame_interpolation_settings,
                inputs=frame_interpolation_settings_inputs,
                queue=False,
            )

        preset_export.click(
            settings_preset_export_status,
            inputs=preset_name,
            outputs=preset_status,
            queue=False,
        )
        preset_import_event = preset_import.upload(
            apply_settings_preset,
            inputs=[preset_import, preset_name],
            outputs=[preset_name, *settings_component_outputs, preset_status],
            queue=False,
        )
        preset_import_event.then(
            persist_gpu_settings,
            inputs=[ai_gpu_selector, video_gpu_selector],
            outputs=gpu_status,
            queue=False,
        ).then(
            frame_interpolation_capability_text,
            outputs=frame_interpolation_capabilities,
            queue=False,
            show_progress="hidden",
        ).then(
            describe_frame_interpolation_plan,
            inputs=[
                frame_interpolation_sources,
                frame_interpolation_target_fps,
                frame_interpolation_engine,
            ],
            outputs=frame_interpolation_plan,
            queue=False,
            show_progress="hidden",
        )

        for selector in [ai_gpu_selector, video_gpu_selector]:
            selector.input(
                persist_gpu_settings,
                inputs=[ai_gpu_selector, video_gpu_selector],
                outputs=gpu_status,
                queue=False,
            ).then(
                frame_interpolation_capability_text,
                outputs=frame_interpolation_capabilities,
                queue=False,
                show_progress="hidden",
            ).then(
                describe_frame_interpolation_plan,
                inputs=[
                    frame_interpolation_sources,
                    frame_interpolation_target_fps,
                    frame_interpolation_engine,
                ],
                outputs=frame_interpolation_plan,
                queue=False,
                show_progress="hidden",
            )

        image_render.click(
            render_image_batch,
            inputs=image_inputs,
            outputs=[
                image_output_gallery,
                image_output_files,
                image_zip,
                image_results,
                image_status,
            ],
            concurrency_limit=1,
        )
        image_stop.click(cancel_active_job, outputs=image_status, queue=False)

        video_render.click(
            render_video_batch,
            inputs=video_inputs,
            outputs=[output_video, video_output_files, video_results, video_status],
            concurrency_limit=1,
        )
        video_preview.click(
            preview_video,
            inputs=video_preview_inputs,
            outputs=[output_video, video_status],
            concurrency_limit=1,
        )
        video_preview_frame.click(
            preview_one_frame,
            inputs=video_preview_inputs,
            outputs=[output_video, video_status],
            concurrency_limit=1,
        )
        video_stop.click(cancel_active_job, outputs=video_status, queue=False)

        frame_interpolation_render.click(
            render_frame_interpolation_batch,
            inputs=frame_interpolation_inputs,
            outputs=[
                frame_interpolation_output_video,
                frame_interpolation_output_files,
                frame_interpolation_results,
                frame_interpolation_status,
            ],
            concurrency_limit=1,
        )
        frame_interpolation_preview.click(
            preview_frame_interpolation,
            inputs=frame_interpolation_preview_inputs,
            outputs=[frame_interpolation_output_video, frame_interpolation_status],
            concurrency_limit=1,
        )
        frame_interpolation_stop.click(
            cancel_active_job, outputs=frame_interpolation_status, queue=False
        )

        reset_outputs = [
            *settings_component_outputs,
            image_status,
            video_status,
            frame_interpolation_status,
            gpu_status,
        ]
        image_reset.click(reset_saved_settings, outputs=reset_outputs, queue=False)
        video_reset.click(reset_saved_settings, outputs=reset_outputs, queue=False)
        frame_interpolation_reset.click(
            reset_saved_settings, outputs=reset_outputs, queue=False
        )
    return demo


def main() -> None:
    print("Preparing DLSS, GPU, image, and FFmpeg runtime before launching the UI...", flush=True)
    try:
        prepared = prepare_runtime()
    except Exception as exc:
        print(f"Startup preparation failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    print(
        f"Runtime ready on {prepared.gpu['display_name']}; launching Gradio.",
        flush=True,
    )
    OUTPUTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    demo = build_app()
    demo.queue(default_concurrency_limit=1).launch(
        css=APP_CSS,
        theme=gr.themes.Ocean(),
        server_name="127.0.0.1",
        inbrowser=True,
        share=False,
        allowed_paths=[str(OUTPUTS.resolve())],
        show_error=True,
    )


if __name__ == "__main__":
    main()
