from __future__ import annotations

import configparser
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from .ffmpeg import ENCODING_QUALITIES
from .frame_interpolation.models import ENGINE_CHOICES, FPS_CHOICES
from .naming import RENAME_MODES, validate_rename
from .video import (
    DLSS_MODEL_PRESETS,
    NR_PRESETS,
    NR_STYLES,
    ConversionOptions,
    resolve_native_settings,
    resolve_upscaling_mode,
)


QUALITY_CHOICES = ENCODING_QUALITIES
CODEC_CHOICES = ("H.264", "HEVC", "AV1", "ProRes Proxy")
CONTAINER_CHOICES = ("MP4", "MKV", "MOV")
IMAGE_FORMAT_CHOICES = ("PNG", "JPEG", "WebP", "AVIF", "TIFF")
CONFIG_SECTION = "Settings"
PRESET_FORMAT = "dlss5-visual-enhancer-settings-preset"
PRESET_SCHEMA_VERSION = 1
MAX_PRESET_BYTES = 1024 * 1024
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True, slots=True)
class UISettings:
    ai_gpu_uuid: str = "auto"
    video_gpu_uuid: str = "auto"
    nr_style: str = "Default"
    nr_intensity: float = 1.0
    local_tone_strength: float = 1.0
    local_structure_strength: float = 1.0
    skin_structure_strength: float = -1.0
    upscaling_factor: float = 1.0
    codec: str = "H.264"
    container: str = "MP4"
    quality: str = "Auto (Default)"
    image_format: str = "PNG"
    image_quality: int = 95
    nr_preset: str = "Default"
    automatic_mask: bool = False
    image_rename_mode: str = "Auto"
    image_custom_suffix: str = "_DLSS5"
    video_rename_mode: str = "Auto"
    video_custom_suffix: str = "_DLSS5"
    dlss_model_preset: str = "Default"
    frame_interpolation_target_fps: str = "60"
    frame_interpolation_engine: str = "Auto"
    frame_interpolation_codec: str = "H.264"
    frame_interpolation_container: str = "MP4"
    frame_interpolation_quality: str = "Auto (Default)"
    frame_interpolation_rename_mode: str = "Auto"
    frame_interpolation_custom_suffix: str = "_DLSSFG"

    def component_values(
        self,
    ) -> tuple[str, str, float, float, float, float, float, bool, str, str, str, str]:
        return (
            self.nr_preset,
            self.nr_style,
            self.nr_intensity,
            self.local_tone_strength,
            self.local_structure_strength,
            self.skin_structure_strength,
            self.upscaling_factor,
            self.automatic_mask,
            self.codec,
            self.container,
            self.quality,
            self.dlss_model_preset,
        )


DEFAULT_SETTINGS = UISettings()


def _validate(settings: UISettings) -> UISettings:
    for label, value in (
        ("AI Processing GPU", settings.ai_gpu_uuid),
        ("Video Processing GPU", settings.video_gpu_uuid),
    ):
        if not isinstance(value, str) or not value.strip() or len(value) > 160:
            raise ValueError(f"{label} selection must be Automatic or a valid GPU UUID.")
    resolve_native_settings(
        ConversionOptions(
            nr_preset=settings.nr_preset,
            nr_style=settings.nr_style,
            nr_intensity=settings.nr_intensity,
            local_tone_strength=settings.local_tone_strength,
            local_structure_strength=settings.local_structure_strength,
            skin_structure_strength=settings.skin_structure_strength,
            automatic_mask=settings.automatic_mask,
            upscaling_factor=settings.upscaling_factor,
            dlss_model_preset=settings.dlss_model_preset,
        )
    )
    resolve_upscaling_mode(settings.upscaling_factor)
    if not isinstance(settings.automatic_mask, bool):
        raise ValueError("Automatic Mask must be a boolean value.")
    allowed = {
        "Video codec": (settings.codec, CODEC_CHOICES),
        "Container": (settings.container, CONTAINER_CHOICES),
        "Encoding quality": (settings.quality, QUALITY_CHOICES),
        "Image format": (settings.image_format, IMAGE_FORMAT_CHOICES),
        "Frame Interpolation FPS": (
            settings.frame_interpolation_target_fps,
            FPS_CHOICES,
        ),
        "Frame Interpolation engine": (
            settings.frame_interpolation_engine,
            ENGINE_CHOICES,
        ),
        "Frame Interpolation codec": (
            settings.frame_interpolation_codec,
            CODEC_CHOICES,
        ),
        "Frame Interpolation container": (
            settings.frame_interpolation_container,
            CONTAINER_CHOICES,
        ),
        "Frame Interpolation quality": (
            settings.frame_interpolation_quality,
            QUALITY_CHOICES,
        ),
    }
    for label, (value, choices) in allowed.items():
        if value not in choices:
            raise ValueError(f"Unknown {label}: {value!r}.")
    if isinstance(settings.image_quality, bool) or not 1 <= int(settings.image_quality) <= 100:
        raise ValueError("Image quality must be an integer from 1 to 100.")
    if int(settings.image_quality) != settings.image_quality:
        raise ValueError("Image quality must be an integer from 1 to 100.")
    validate_rename(settings.image_rename_mode, settings.image_custom_suffix)
    validate_rename(settings.video_rename_mode, settings.video_custom_suffix)
    validate_rename(
        settings.frame_interpolation_rename_mode,
        settings.frame_interpolation_custom_suffix,
    )
    return settings


def _preset_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Preset name must be text.")
    name = value.strip()
    if not name:
        raise ValueError("Enter a preset name before exporting.")
    if len(name) > 120:
        raise ValueError("Preset name must be 120 characters or fewer.")
    if any(ord(character) < 32 for character in name):
        raise ValueError("Preset name cannot contain control characters.")
    return name


def preset_filename(name: str) -> str:
    """Return a portable JSON filename while preserving the display name in the file."""
    display_name = _preset_name(name)
    characters = [
        character if character.isalnum() or character in "-_" else "_"
        for character in display_name
    ]
    stem = re.sub(r"_+", "_", "".join(characters)).strip("-_")[:80].rstrip("-_")
    if not stem:
        raise ValueError("Preset name must contain at least one letter or number.")
    if stem.upper() in _WINDOWS_RESERVED_NAMES:
        stem += "_preset"
    return f"{stem}.json"


def preset_document(name: str, settings: UISettings) -> dict[str, Any]:
    """Build the versioned user-facing preset document."""
    display_name = _preset_name(name)
    _validate(settings)
    return {
        "format": PRESET_FORMAT,
        "schema_version": PRESET_SCHEMA_VERSION,
        "name": display_name,
        "settings": asdict(settings),
    }


def export_settings_preset(name: str, settings: UISettings) -> Path:
    """Write a validated preset to an isolated temporary download directory."""
    document = preset_document(name, settings)
    filename = preset_filename(document["name"])
    directory = Path(tempfile.mkdtemp(prefix="dlss5-settings-preset-"))
    path = directory / filename
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def _coerce_preset_value(field_name: str, value: Any, current: UISettings) -> Any:
    expected = getattr(current, field_name)
    if isinstance(expected, bool):
        if not isinstance(value, bool):
            raise ValueError(f"Preset setting {field_name!r} must be a boolean.")
        return value
    if isinstance(expected, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Preset setting {field_name!r} must be an integer.")
        return value
    if isinstance(expected, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Preset setting {field_name!r} must be a number.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Preset setting {field_name!r} must be finite.")
        return number
    if isinstance(expected, str):
        if not isinstance(value, str):
            raise ValueError(f"Preset setting {field_name!r} must be text.")
        return value
    raise ValueError(f"Preset setting {field_name!r} has an unsupported type.")


def import_settings_preset(
    path: str | os.PathLike[str], current: UISettings
) -> tuple[str, UISettings]:
    """Load, compatibly merge, and atomically validate a version-1 preset."""
    preset_path = Path(path)
    if preset_path.suffix.casefold() != ".json":
        raise ValueError("Choose a JSON preset file.")
    try:
        size = preset_path.stat().st_size
    except OSError as exc:
        raise ValueError("The selected preset file cannot be read.") from exc
    if size > MAX_PRESET_BYTES:
        raise ValueError("Preset file is too large; the maximum size is 1 MiB.")
    try:
        document = json.loads(preset_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Preset file is not valid UTF-8 JSON.") from exc
    if not isinstance(document, dict):
        raise ValueError("Preset JSON must contain an object at its top level.")
    if document.get("format") != PRESET_FORMAT:
        raise ValueError("This JSON file is not a DLSS 5 Visual Enhancer settings preset.")
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Preset schema_version must be an integer.")
    if version != PRESET_SCHEMA_VERSION:
        direction = "newer" if version > PRESET_SCHEMA_VERSION else "unsupported"
        raise ValueError(
            f"Preset schema version {version} is {direction}; this build supports version "
            f"{PRESET_SCHEMA_VERSION}."
        )
    name = _preset_name(document.get("name"))
    imported = document.get("settings")
    if not isinstance(imported, dict):
        raise ValueError("Preset settings must be a JSON object.")

    known_names = {field.name for field in fields(UISettings)}
    changes = {
        key: _coerce_preset_value(key, value, current)
        for key, value in imported.items()
        if key in known_names
    }
    merged = replace(current, **changes)
    return name, _validate(merged)


def load_settings(path: str | os.PathLike[str]) -> UISettings:
    config_path = Path(path)
    parser = configparser.ConfigParser()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, configparser.Error):
        return DEFAULT_SETTINGS

    section = parser[CONFIG_SECTION] if parser.has_section(CONFIG_SECTION) else {}

    def choice(key: str, choices: tuple[str, ...], default: str) -> str:
        value = section.get(key, default)
        return value if value in choices else default

    def number(key: str, minimum: float, maximum: float, default: float) -> float:
        try:
            value = float(section.get(key, str(default)))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) and minimum <= value <= maximum else default

    def upscaling_factor() -> float:
        try:
            return resolve_upscaling_mode(float(section.get("upscaling_factor", "1.0")))[0]
        except (TypeError, ValueError):
            return DEFAULT_SETTINGS.upscaling_factor

    def image_quality() -> int:
        value = number("image_quality", 1, 100, DEFAULT_SETTINGS.image_quality)
        return int(value) if float(value).is_integer() else DEFAULT_SETTINGS.image_quality

    def boolean(key: str, default: bool) -> bool:
        raw_value = section.get(key)
        if raw_value is None:
            return default
        parsed = configparser.ConfigParser.BOOLEAN_STATES.get(str(raw_value).casefold())
        return parsed if parsed is not None else default

    def frame_interpolation_engine() -> str:
        value = section.get(
            "frame_interpolation_engine",
            DEFAULT_SETTINGS.frame_interpolation_engine,
        )
        if value == "Experimental Cascade":
            return "Cascade"
        return value if value in ENGINE_CHOICES else DEFAULT_SETTINGS.frame_interpolation_engine

    image_rename_mode = choice(
        "image_rename_mode", RENAME_MODES, DEFAULT_SETTINGS.image_rename_mode
    )
    image_custom_suffix = section.get(
        "image_custom_suffix", DEFAULT_SETTINGS.image_custom_suffix
    )
    video_rename_mode = choice(
        "video_rename_mode", RENAME_MODES, DEFAULT_SETTINGS.video_rename_mode
    )
    video_custom_suffix = section.get(
        "video_custom_suffix", DEFAULT_SETTINGS.video_custom_suffix
    )
    frame_interpolation_rename_mode = choice(
        "frame_interpolation_rename_mode",
        RENAME_MODES,
        DEFAULT_SETTINGS.frame_interpolation_rename_mode,
    )
    frame_interpolation_custom_suffix = section.get(
        "frame_interpolation_custom_suffix",
        DEFAULT_SETTINGS.frame_interpolation_custom_suffix,
    )
    try:
        validate_rename(image_rename_mode, image_custom_suffix)
    except ValueError:
        image_rename_mode = DEFAULT_SETTINGS.image_rename_mode
        image_custom_suffix = DEFAULT_SETTINGS.image_custom_suffix
    try:
        validate_rename(video_rename_mode, video_custom_suffix)
    except ValueError:
        video_rename_mode = DEFAULT_SETTINGS.video_rename_mode
        video_custom_suffix = DEFAULT_SETTINGS.video_custom_suffix
    try:
        validate_rename(
            frame_interpolation_rename_mode,
            frame_interpolation_custom_suffix,
        )
    except ValueError:
        frame_interpolation_rename_mode = DEFAULT_SETTINGS.frame_interpolation_rename_mode
        frame_interpolation_custom_suffix = DEFAULT_SETTINGS.frame_interpolation_custom_suffix

    return UISettings(
        ai_gpu_uuid=section.get("ai_gpu_uuid", DEFAULT_SETTINGS.ai_gpu_uuid).strip()
        or DEFAULT_SETTINGS.ai_gpu_uuid,
        video_gpu_uuid=section.get("video_gpu_uuid", DEFAULT_SETTINGS.video_gpu_uuid).strip()
        or DEFAULT_SETTINGS.video_gpu_uuid,
        nr_preset=choice("nr_preset", tuple(NR_PRESETS), DEFAULT_SETTINGS.nr_preset),
        nr_style=choice("nr_style", tuple(NR_STYLES), DEFAULT_SETTINGS.nr_style),
        nr_intensity=number("nr_intensity", 0.0, 2.0, DEFAULT_SETTINGS.nr_intensity),
        local_tone_strength=number(
            "local_tone_strength", 0.0, 2.0, DEFAULT_SETTINGS.local_tone_strength
        ),
        local_structure_strength=number(
            "local_structure_strength", 0.0, 2.0, DEFAULT_SETTINGS.local_structure_strength
        ),
        skin_structure_strength=number(
            "skin_structure_strength", -1.0, 2.0, DEFAULT_SETTINGS.skin_structure_strength
        ),
        automatic_mask=boolean("automatic_mask", DEFAULT_SETTINGS.automatic_mask),
        upscaling_factor=upscaling_factor(),
        codec=choice("codec", CODEC_CHOICES, DEFAULT_SETTINGS.codec),
        container=choice("container", CONTAINER_CHOICES, DEFAULT_SETTINGS.container),
        quality=choice("quality", QUALITY_CHOICES, DEFAULT_SETTINGS.quality),
        image_format=choice(
            "image_format", IMAGE_FORMAT_CHOICES, DEFAULT_SETTINGS.image_format
        ),
        image_quality=image_quality(),
        dlss_model_preset=choice(
            "dlss_model_preset",
            tuple(DLSS_MODEL_PRESETS),
            DEFAULT_SETTINGS.dlss_model_preset,
        ),
        image_rename_mode=image_rename_mode,
        image_custom_suffix=image_custom_suffix,
        video_rename_mode=video_rename_mode,
        video_custom_suffix=video_custom_suffix,
        frame_interpolation_target_fps=choice(
            "frame_interpolation_target_fps",
            FPS_CHOICES,
            DEFAULT_SETTINGS.frame_interpolation_target_fps,
        ),
        frame_interpolation_engine=frame_interpolation_engine(),
        frame_interpolation_codec=choice(
            "frame_interpolation_codec",
            CODEC_CHOICES,
            DEFAULT_SETTINGS.frame_interpolation_codec,
        ),
        frame_interpolation_container=choice(
            "frame_interpolation_container",
            CONTAINER_CHOICES,
            DEFAULT_SETTINGS.frame_interpolation_container,
        ),
        frame_interpolation_quality=choice(
            "frame_interpolation_quality",
            QUALITY_CHOICES,
            DEFAULT_SETTINGS.frame_interpolation_quality,
        ),
        frame_interpolation_rename_mode=frame_interpolation_rename_mode,
        frame_interpolation_custom_suffix=frame_interpolation_custom_suffix,
    )


def save_settings(path: str | os.PathLike[str], settings: UISettings) -> None:
    settings = _validate(settings)
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    parser = configparser.ConfigParser()
    parser[CONFIG_SECTION] = {
        "ai_gpu_uuid": settings.ai_gpu_uuid,
        "video_gpu_uuid": settings.video_gpu_uuid,
        "nr_preset": settings.nr_preset,
        "nr_style": settings.nr_style,
        "nr_intensity": f"{settings.nr_intensity:.2f}",
        "local_tone_strength": f"{settings.local_tone_strength:.2f}",
        "local_structure_strength": f"{settings.local_structure_strength:.2f}",
        "skin_structure_strength": f"{settings.skin_structure_strength:.2f}",
        "automatic_mask": str(settings.automatic_mask).lower(),
        "upscaling_factor": f"{settings.upscaling_factor:g}",
        "codec": settings.codec,
        "container": settings.container,
        "quality": settings.quality,
        "image_format": settings.image_format,
        "image_quality": str(settings.image_quality),
        "image_rename_mode": settings.image_rename_mode,
        "image_custom_suffix": settings.image_custom_suffix,
        "video_rename_mode": settings.video_rename_mode,
        "video_custom_suffix": settings.video_custom_suffix,
        "dlss_model_preset": settings.dlss_model_preset,
        "frame_interpolation_target_fps": settings.frame_interpolation_target_fps,
        "frame_interpolation_engine": settings.frame_interpolation_engine,
        "frame_interpolation_codec": settings.frame_interpolation_codec,
        "frame_interpolation_container": settings.frame_interpolation_container,
        "frame_interpolation_quality": settings.frame_interpolation_quality,
        "frame_interpolation_rename_mode": settings.frame_interpolation_rename_mode,
        "frame_interpolation_custom_suffix": settings.frame_interpolation_custom_suffix,
    }

    temporary = config_path.with_name(f".{config_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            parser.write(stream)
        os.replace(temporary, config_path)
    finally:
        if temporary.exists():
            temporary.unlink()
