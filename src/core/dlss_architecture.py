from __future__ import annotations

"""Selectable DLSS 5 Neural Rendering architecture builds.

``bin/runtime/dlssnr/`` holds one ``nvngx_dlssnr.dll`` per GPU-architecture
floor (extracted from the ``bin/*.zip`` archives)::

    dlssnr/Turing+/nvngx_dlssnr.dll
    dlssnr/Ada Lovelace+/nvngx_dlssnr.dll
    dlssnr/Blackwell+/nvngx_dlssnr.dll

The native worker (``bin/runtime/host/nvngx.dll``) loads the NR runtime
through the NGX API by filename convention from ``host/`` (and ``dlss/``),
with no architecture gating of its own — so selecting an architecture is a
pure file swap of ``host/nvngx_dlssnr.dll`` (which RenoDX requires to sit
next to itself). No native recompilation is involved.

Policy (per product decision):
- ``Auto`` maps RTX 20/30 (Turing/Ampere) to ``Turing+``, RTX 40 (Ada) to
  ``Ada Lovelace+``, RTX 50 (Blackwell) to ``Blackwell+``; unknown GPUs fall
  back to ``Turing+`` (the historical default build).
- Applied at startup (before ``prepare_runtime()`` warms the DLL) and again
  before each render. The worker is a per-job subprocess, so swapping
  between jobs cannot race a running render. Identical content is skipped.
- Any failure warns (``logs/dlssnr_architecture.log``) and keeps the
  current host DLL; renders proceed. These helpers never raise.
"""

import hashlib
import time
import zipfile
from pathlib import Path
from typing import Any

from .paths import DLSSNR_DIR, LOGS, NEURAL_RUNTIME, ROOT

DLL_FILENAME = "nvngx_dlssnr.dll"
LOG_FILENAME = "dlssnr_architecture.log"

DLSS_ARCHITECTURE_CHOICES = (
    "Auto",
    "Turing and higher",
    "Ada Lovelace and higher",
    "Blackwell and higher",
)
DEFAULT_DLSS_ARCHITECTURE = "Auto"

_CHOICE_SUBFOLDERS: dict[str, str | None] = {
    "Auto": None,
    "Turing and higher": "Turing+",
    "Ada Lovelace and higher": "Ada Lovelace+",
    "Blackwell and higher": "Blackwell+",
}

_SUBFOLDER_ZIPS: dict[str, str] = {
    "Turing+": "Turing+.zip",
    "Ada Lovelace+": "Ada Lovelace+.zip",
    "Blackwell+": "Blackwell+.zip",
}


def _warn(message: str) -> None:
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOGS / LOG_FILENAME, "a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")
    except OSError:
        pass


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def subfolder_for_choice(choice: str) -> str | None:
    """Return the explicit arch subfolder, or None for ``Auto``."""
    return _CHOICE_SUBFOLDERS.get(choice)


def subfolder_for_gpu(gpu: dict[str, Any] | None) -> str:
    """Map a detected GPU to its ``Auto`` arch subfolder (Turing+ default)."""
    architecture = ""
    generation: Any = None
    if isinstance(gpu, dict):
        architecture = str(gpu.get("architecture") or "")
        generation = gpu.get("generation")
    if architecture == "Blackwell" or generation == 50:
        return "Blackwell+"
    if architecture == "Ada" or generation == 40:
        return "Ada Lovelace+"
    return "Turing+"


def resolve_subfolder(choice: str, gpu: dict[str, Any] | None) -> str:
    """Resolve a DLSS Architecture setting (possibly ``Auto``) to a folder."""
    if choice not in DLSS_ARCHITECTURE_CHOICES:
        _warn(f"Unknown DLSS Architecture {choice!r}; using {DEFAULT_DLSS_ARCHITECTURE!r}.")
        choice = DEFAULT_DLSS_ARCHITECTURE
    explicit = subfolder_for_choice(choice)
    if explicit is not None:
        return explicit
    return subfolder_for_gpu(gpu)


def arch_dll_path(subfolder: str) -> Path:
    return DLSSNR_DIR / subfolder / DLL_FILENAME


def ensure_dlssnr_dlls() -> dict[str, bool]:
    """Extract any missing arch DLL from its ``bin/*.zip``. Never raises.

    Returns ``{subfolder: available}``.
    """
    availability: dict[str, bool] = {}
    for subfolder, zip_name in _SUBFOLDER_ZIPS.items():
        destination = arch_dll_path(subfolder)
        try:
            if destination.is_file():
                availability[subfolder] = True
                continue
            zip_path = ROOT / "bin" / zip_name
            if not zip_path.is_file():
                _warn(
                    f"DLSS Architecture DLL missing: {destination} "
                    f"(archive {zip_path} not found)."
                )
                availability[subfolder] = False
                continue
            with zipfile.ZipFile(zip_path) as archive:
                if archive.namelist() != [DLL_FILENAME]:
                    _warn(
                        f"Archive {zip_path} has unexpected contents; "
                        f"expected only {DLL_FILENAME}."
                    )
                    availability[subfolder] = False
                    continue
                data = archive.read(DLL_FILENAME)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{DLL_FILENAME}.tmp")
            temporary.write_bytes(data)
            temporary.replace(destination)
            availability[subfolder] = True
        except Exception as exc:  # noqa: BLE001 - warn-and-continue policy
            _warn(f"Could not provide {destination}: {exc}.")
            availability[subfolder] = False
    return availability


def apply_dlss_architecture(choice: str, gpu: dict[str, Any] | None) -> str:
    """Copy the resolved arch DLL over ``host/nvngx_dlssnr.dll``.

    Returns the applied subfolder name, or ``""`` when nothing was applied
    (missing source or copy failure — the current host DLL is kept and the
    caller should proceed). Never raises.
    """
    try:
        ensure_dlssnr_dlls()
        subfolder = resolve_subfolder(choice, gpu)
        source = arch_dll_path(subfolder)
        if not source.is_file():
            _warn(
                f"DLSS Architecture {choice!r} unavailable "
                f"({source} missing); keeping current host DLL."
            )
            return ""
        source_hash = _sha256(source)
        if source_hash is not None and source_hash == _sha256(NEURAL_RUNTIME):
            return subfolder
        # prepare_runtime() memory-maps the host DLL for warming and holds
        # the mapping for the process lifetime; on Windows an open mapping
        # blocks replacement, so release it before swapping (next
        # prepare_runtime() call transparently re-warms the new file).
        try:
            from .runtime import close_prepared_runtime

            close_prepared_runtime()
        except Exception:  # noqa: BLE001 - best effort
            pass
        NEURAL_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        temporary = NEURAL_RUNTIME.with_name(f".{DLL_FILENAME}.tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(NEURAL_RUNTIME)
        return subfolder
    except Exception as exc:  # noqa: BLE001 - warn-and-continue policy
        _warn(f"Could not apply DLSS Architecture {choice!r}: {exc}.")
        return ""


def _current_choice_and_gpu() -> tuple[str, dict[str, Any] | None]:
    from ..settings.storage import CONFIG_PATH, SETTINGS_STATE, load_settings

    from .gpu_detection import detect_gpus
    from .gpu_selection import resolve_runtime_ai_gpu

    with SETTINGS_STATE.lock:
        settings = SETTINGS_STATE.current or load_settings(CONFIG_PATH)
    choice = getattr(settings, "dlss_architecture", DEFAULT_DLSS_ARCHITECTURE)
    if choice not in DLSS_ARCHITECTURE_CHOICES:
        choice = DEFAULT_DLSS_ARCHITECTURE
    try:
        gpus = detect_gpus()
    except Exception as exc:  # noqa: BLE001 - no GPU info available
        _warn(f"GPU detection unavailable for DLSS Architecture: {exc}.")
        return choice, None
    try:
        gpu = resolve_runtime_ai_gpu(
            gpus, {}, getattr(settings, "ai_gpu_uuid", "auto")
        )
    except Exception as exc:  # noqa: BLE001 - e.g. no RTX GPU
        _warn(f"AI GPU resolution unavailable for DLSS Architecture: {exc}.")
        return choice, None
    return choice, gpu


def init_dlssnr_runtime() -> str:
    """Ensure arch DLLs and apply the saved setting. Call before ``prepare_runtime()``.

    Returns the applied subfolder name or ``""``. Never raises.
    """
    try:
        choice, gpu = _current_choice_and_gpu()
        return apply_dlss_architecture(choice, gpu)
    except Exception as exc:  # noqa: BLE001 - startup must never block
        _warn(f"DLSS Architecture init failed: {exc}.")
        return ""


def apply_current_dlss_architecture(gpu: dict[str, Any] | None) -> str:
    """Re-apply the live setting before a render. Call with the job's GPU.

    Returns the applied subfolder name or ``""``. Never raises.
    """
    try:
        from ..settings.storage import CONFIG_PATH, SETTINGS_STATE, load_settings

        with SETTINGS_STATE.lock:
            settings = SETTINGS_STATE.current or load_settings(CONFIG_PATH)
        choice = getattr(settings, "dlss_architecture", DEFAULT_DLSS_ARCHITECTURE)
        return apply_dlss_architecture(choice, gpu)
    except Exception as exc:  # noqa: BLE001 - render must never block
        _warn(f"DLSS Architecture pre-render apply failed: {exc}.")
        return ""
