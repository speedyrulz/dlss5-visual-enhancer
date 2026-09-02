from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

from ..runtime import ROOT, detect_gpu
from .models import FrameInterpolationCapabilities


RUNTIME_DIR = ROOT / "bin" / "runtime" / "frame_interpolation" / "dlssg"
DLSSG_RUNTIME = RUNTIME_DIR / "nvngx_dlssg.dll"
DLSSG_WORKER = RUNTIME_DIR / "dlssg-worker.exe"
MANIFEST = RUNTIME_DIR / "manifest.json"
EXPECTED_RUNTIME_SHA256 = "135EAF0733C1E37381A8C28ABCF7A862404A54132B81787C04E35D09EFC5E36F"
EXPECTED_WORKER_SHA256 = "8A747F9ED613842D5B8B34A811AD43BC1A9466540E2E5A0C8EF4005F0DB9E384"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _hags_enabled() -> bool:
    if os.name != "nt":
        return False
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers",
        ) as key:
            value, _kind = winreg.QueryValueEx(key, "HwSchMode")
        return int(value) == 2
    except (OSError, ValueError):
        return False


def _authenticode_status(path: Path) -> str:
    if os.name != "nt":
        return "Unavailable"
    escaped = str(path.resolve()).replace("'", "''")
    try:
        process = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Import-Module \"$env:WINDIR\\System32\\WindowsPowerShell\\v1.0\\Modules\\"
                "Microsoft.PowerShell.Security\\Microsoft.PowerShell.Security.psd1\"; "
                f"[string](Get-AuthenticodeSignature -LiteralPath '{escaped}').Status",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "Unavailable"
    return process.stdout.strip() if process.returncode == 0 else "Unavailable"


def _probe_worker(adapter_luid: str | None = None) -> dict:
    if not DLSSG_WORKER.is_file():
        raise RuntimeError(f"DLSSG worker is missing: {DLSSG_WORKER}")
    command = [str(DLSSG_WORKER), "--probe"]
    if adapter_luid:
        command.extend(["--adapter-luid", adapter_luid])
    process = subprocess.run(
        command,
        cwd=str(RUNTIME_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(detail or f"DLSSG capability probe exited with {process.returncode}.")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("DLSSG capability probe returned no result.")
    return json.loads(lines[-1])


@lru_cache(maxsize=32)
def probe_frame_interpolation_capabilities(
    ai_gpu_uuid: str = "auto",
) -> FrameInterpolationCapabilities:
    gpu = detect_gpu(ai_gpu_uuid)
    gpu_name = str(gpu.get("display_name") or gpu.get("name") or "NVIDIA RTX GPU")
    driver = str(gpu.get("driver") or "unknown")
    hags = _hags_enabled()
    if not DLSSG_RUNTIME.is_file():
        return FrameInterpolationCapabilities(
            False, gpu_name, driver, hags, 0, 1, False, "missing", "", "missing",
            "Missing", f"Signed NVIDIA DLSSG runtime is missing: {DLSSG_RUNTIME}",
        )
    runtime_hash = _sha256(DLSSG_RUNTIME)
    signature_status = _authenticode_status(DLSSG_RUNTIME)
    if runtime_hash != EXPECTED_RUNTIME_SHA256:
        return FrameInterpolationCapabilities(
            False, gpu_name, driver, hags, 0, 1, False, "unknown", runtime_hash,
            "unknown", "Hash mismatch",
            "The staged nvngx_dlssg.dll does not match the pinned NVIDIA 310.7 release.",
        )
    if signature_status != "Valid":
        return FrameInterpolationCapabilities(
            False, gpu_name, driver, hags, 0, 1, False, "310.7.0.0", runtime_hash,
            "unknown", signature_status,
            "Windows did not validate the NVIDIA Authenticode signature on nvngx_dlssg.dll.",
        )
    if not DLSSG_WORKER.is_file() or _sha256(DLSSG_WORKER) != EXPECTED_WORKER_SHA256:
        return FrameInterpolationCapabilities(
            False, gpu_name, driver, hags, 0, 1, False, "310.7.0.0", runtime_hash,
            "unknown", "Worker hash mismatch",
            "The direct D3D12 worker is missing or does not match its source build manifest.",
        )
    try:
        probed = _probe_worker(gpu.get("adapter_luid"))
        generated_max = max(0, int(probed.get("multi_frame_count_max", 0)))
        multiplier = generated_max + 1 if generated_max else 1
        available = bool(probed.get("available", False)) and generated_max >= 1 and hags
        detail = str(probed.get("detail") or "")
    except Exception as exc:
        generated_max = 0
        multiplier = 1
        available = False
        detail = str(exc)
        probed = {}
    return FrameInterpolationCapabilities(
        available=available,
        gpu=gpu_name,
        driver=driver,
        hags_enabled=hags,
        native_generated_frame_max=generated_max,
        native_multiplier=multiplier,
        cascade_available=available and multiplier >= 2,
        runtime_version=str(probed.get("runtime_version") or "310.7.0.0"),
        runtime_sha256=runtime_hash,
        worker_version=str(probed.get("worker_version") or "1"),
        signature_status=f"{signature_status} (NVIDIA; pinned hash)",
        detail=detail,
        gpu_uuid=str(gpu.get("uuid") or ai_gpu_uuid),
    )


def clear_capability_cache() -> None:
    probe_frame_interpolation_capabilities.cache_clear()
