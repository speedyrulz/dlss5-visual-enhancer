from __future__ import annotations

import hashlib
import json
import ctypes
import re
import struct
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "bin" / "runtime"
FFMPEG = ROOT / "bin" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = ROOT / "bin" / "ffmpeg" / "bin" / "ffprobe.exe"
WORKER = RUNTIME / "nvngx.dll"  # Signed-snippet caller checks require this image name.
ADDON = RUNTIME / "renodx-dlss5.addon64"
NEURAL_RUNTIME = RUNTIME / "nvngx_dlssnr.dll"
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
JOBS = ROOT / "jobs"

EXPECTED_AMPERE_ADDON_SHA256 = (
    "D5ADF82EB44B065F4C590AC91FE824BAB07AFEA0EB9F994BDE936710C8593952"
)
EXPECTED_AMPERE_NEURAL_SHA256 = (
    "6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927"
)

_FINGERPRINT_LOCK = threading.Lock()
_FINGERPRINT_CACHE: dict[tuple[str, int, int], str] = {}

VIDEO_MAGIC = 0x34563544
SETUP_MAGIC = 0x34505553
FRAME_MAGIC = 0x314D5246
OUT_MAGIC = 0x3154554F
VIDEO_HEADER_FORMAT = "<14I4f"
SETUP_RESPONSE_FORMAT = "<12I"


def _file_sha256(path: Path) -> str:
    """Hash a runtime component, cached until its size or modification time changes."""
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    with _FINGERPRINT_LOCK:
        cached = _FINGERPRINT_CACHE.get(key)
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest().upper()
    with _FINGERPRINT_LOCK:
        stale = [entry for entry in _FINGERPRINT_CACHE if entry[0] == str(resolved)]
        for entry in stale:
            _FINGERPRINT_CACHE.pop(entry, None)
        _FINGERPRINT_CACHE[key] = value
    return value


def inspect_runtime_bundle(
    addon_path: Path | None = None,
    neural_path: Path | None = None,
) -> dict[str, Any]:
    """Return reproducible component identities for compatibility checks and reports."""
    addon = addon_path or ADDON
    neural = neural_path or NEURAL_RUNTIME
    addon_hash = _file_sha256(addon)
    neural_hash = _file_sha256(neural)
    worker_hash = _file_sha256(WORKER)
    known_pair = (
        addon_hash == EXPECTED_AMPERE_ADDON_SHA256
        and neural_hash == EXPECTED_AMPERE_NEURAL_SHA256
    )
    return {
        "known_ampere_pair": known_pair,
        "addon": {
            "path": str(addon.resolve()),
            "sha256": addon_hash,
            "expected_sha256": EXPECTED_AMPERE_ADDON_SHA256,
            "matches_expected": addon_hash == EXPECTED_AMPERE_ADDON_SHA256,
            "version": "4.70",
            "release": "RenoDX DLSS5 v4.70",
        },
        "neural_runtime": {
            "path": str(neural.resolve()),
            "sha256": neural_hash,
            "expected_sha256": EXPECTED_AMPERE_NEURAL_SHA256,
            "matches_expected": neural_hash == EXPECTED_AMPERE_NEURAL_SHA256,
            "version": "310.8.SF-v2",
            "release": "DLSS NR 310.8.SF-v2",
        },
        "worker": {
            "path": str(WORKER.resolve()),
            "sha256": worker_hash,
        },
    }


def validate_gpu_runtime(
    gpu: dict[str, Any], bundle: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Require the tested community component pair for experimental Ampere use."""
    inspected = bundle or inspect_runtime_bundle()
    if int(gpu.get("generation", 0)) == 30 and not inspected["known_ampere_pair"]:
        addon_hash = inspected["addon"]["sha256"]
        neural_hash = inspected["neural_runtime"]["sha256"]
        raise RuntimeError(
            f"{gpu.get('name', 'RTX 30-series GPU')} requires the tested experimental Ampere "
            "runtime pair: RenoDX DLSS5 v4.70 plus DLSS NR 310.8.SF-v2. "
            f"Installed hashes are add-on {addon_hash} and neural runtime {neural_hash}. "
            "Restore the matching runtime files before rendering."
        )
    return inspected


def classify_worker_failure(
    *,
    worker_code: int,
    frame_index: int,
    worker_logs: list[str],
    reshade_lines: list[str],
    gpu: dict[str, Any],
    runtime_bundle: dict[str, Any],
) -> str:
    """Translate native/add-on failures into an actionable feature-18 diagnosis."""
    evidence = "\n".join([*worker_logs, *reshade_lines])
    access_violation = (
        "evaluate raised 0xC0000005" in evidence
        or "feature 18 evaluate raised an exception" in evidence
    )
    generation = int(gpu.get("generation", 0))
    if access_violation and generation == 30:
        summary = (
            "DLSS 5 feature 18 hit access violation 0xC0000005 on the experimental "
            f"RTX 30-series path before frame {frame_index} completed. Ordinary D3D12/NGX "
            "initialization may still have succeeded; the failure is inside the patched neural "
            "runtime/add-on evaluation."
        )
        if runtime_bundle.get("known_ampere_pair"):
            summary += (
                " The tested v4.70/SF-v2 component pair is installed. Update to the latest "
                "NVIDIA driver; if the failure remains, this closed community runtime is "
                "incompatible with this Ampere system and there is no truthful non-neural fallback."
            )
    elif access_violation:
        summary = (
            f"DLSS 5 feature 18 raised access violation 0xC0000005 before frame {frame_index} "
            "completed inside the neural runtime/add-on evaluation."
        )
    else:
        summary = (
            f"Native DLSS worker exited with code {worker_code} before frame {frame_index} "
            "completed."
        )

    addon_hash = runtime_bundle.get("addon", {}).get("sha256", "unavailable")
    neural_hash = runtime_bundle.get("neural_runtime", {}).get("sha256", "unavailable")
    details = [
        summary,
        f"GPU: {gpu.get('name', 'unknown')} | driver: {gpu.get('driver', 'unknown')}",
        f"RenoDX add-on SHA-256: {addon_hash}",
        f"DLSSNR runtime SHA-256: {neural_hash}",
    ]
    if worker_logs:
        details.append("Worker log:\n" + "\n".join(worker_logs[-60:]))
    if reshade_lines:
        details.append("ReShade feature-18 log:\n" + "\n".join(reshade_lines[-60:]))
    return "\n".join(details)


def write_failure_report(
    *,
    operation: str,
    source: str,
    error: BaseException | str,
    gpu: dict[str, Any] | None,
    runtime_bundle: dict[str, Any] | None,
    worker_code: int | None = None,
    worker_logs: list[str] | None = None,
    reshade_lines: list[str] | None = None,
    encoder_logs: list[str] | None = None,
    logs_dir: Path | None = None,
) -> Path:
    """Persist diagnostics even when the incomplete media output is removed."""
    destination = logs_dir or LOGS
    destination.mkdir(parents=True, exist_ok=True)
    safe_operation = re.sub(r"[^A-Za-z0-9_.-]+", "-", operation).strip("-") or "render"
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
    report_path = destination / f"{safe_operation}-failure-{stamp}.json"
    report = {
        "status": "failure",
        "operation": operation,
        "input": source,
        "error": str(error),
        "gpu": gpu,
        "runtime_bundle": runtime_bundle,
        "worker_exit_code": worker_code,
        "worker_log": list(worker_logs or []),
        "reshade_feature_18_log": list(reshade_lines or []),
        "encoder_log": list(encoder_logs or []),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path.resolve()


class Cancelled(RuntimeError):
    pass


class JobController:
    """Own cancellation state and subprocesses for one render."""

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self._processes: list[subprocess.Popen] = []

    def register(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.append(process)
            cancelled = self.cancel.is_set()
        if cancelled and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    def unregister(self, process: subprocess.Popen) -> None:
        with self._lock:
            if process in self._processes:
                self._processes.remove(process)

    def stop(self) -> None:
        self.cancel.set()
        self.terminate_processes()

    def terminate_processes(self) -> None:
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass


_RENDER_LOCK = threading.Lock()
_ACTIVE_LOCK = threading.Lock()
_ACTIVE: JobController | None = None


@contextmanager
def active_job() -> Iterator[JobController]:
    """Claim the single GPU render slot and always release its resources."""
    global _ACTIVE
    if not _RENDER_LOCK.acquire(blocking=False):
        raise RuntimeError("Another GPU render is already running.")
    controller = JobController()
    with _ACTIVE_LOCK:
        _ACTIVE = controller
    try:
        yield controller
    finally:
        controller.terminate_processes()
        with _ACTIVE_LOCK:
            if _ACTIVE is controller:
                _ACTIVE = None
        _RENDER_LOCK.release()


def cancel_active_job() -> str:
    with _ACTIVE_LOCK:
        controller = _ACTIVE
    if controller is None:
        return "No render is running."
    controller.stop()
    return "Stop requested; incomplete output will be removed and completed batch files retained."


def drain_text(stream, lines: list[str]) -> None:
    for raw in iter(stream.readline, b""):
        lines.append(raw.decode("utf-8", "replace").rstrip())


class BoundedLogBuffer:
    """Keep diagnostic evidence and a bounded tail instead of every frame log."""

    _IMPORTANT = (
        "profile applied",
        "model preset",
        "DLSS 5 add-on",
        "carrier ready",
        "stream source",
        "optimal settings",
        "complete:",
        "failed",
        "error",
        "exception",
    )

    def __init__(self, max_tail: int = 500, max_important: int = 100) -> None:
        self._tail: deque[str] = deque(maxlen=max_tail)
        self._important: list[str] = []
        self._max_important = max_important
        self._seen = 0
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self._seen += 1
            self._tail.append(line)
            lowered = line.casefold()
            if (
                len(self._important) < self._max_important
                and any(marker.casefold() in lowered for marker in self._IMPORTANT)
            ):
                self._important.append(line)

    def snapshot(self) -> list[str]:
        with self._lock:
            important = list(self._important)
            tail = list(self._tail)
        seen: set[str] = set()
        return [line for line in [*important, *tail] if not (line in seen or seen.add(line))]

    @property
    def dropped_lines(self) -> int:
        with self._lock:
            return max(0, self._seen - len(self._tail))


def drain_bounded_text(stream, buffer: BoundedLogBuffer) -> None:
    for raw in iter(stream.readline, b""):
        buffer.append(raw.decode("utf-8", "replace").rstrip())


def resize_fit(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = rgba.shape[:2]
    if source_width == width and source_height == height:
        return np.ascontiguousarray(rgba, dtype=np.uint8)
    scale = min(width / source_width, height / source_height)
    fit_width = max(1, min(width, int(round(source_width * scale))))
    fit_height = max(1, min(height, int(round(source_height * scale))))
    resized = cv2.resize(rgba, (fit_width, fit_height), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    canvas[..., 3] = 255
    x = (width - fit_width) // 2
    y = (height - fit_height) // 2
    canvas[y : y + fit_height, x : x + fit_width] = resized
    return canvas


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return np.ascontiguousarray(np.rot90(frame, 3))
    if rotation == 180:
        return np.ascontiguousarray(np.rot90(frame, 2))
    if rotation == 270:
        return np.ascontiguousarray(np.rot90(frame, 1))
    return frame


_RTX_ARCHITECTURES: dict[tuple[int, int], tuple[str, int]] = {
    (8, 0): ("Ampere", 30),
    (8, 6): ("Ampere", 30),
    (8, 7): ("Ampere", 30),
    (8, 8): ("Ampere", 30),
    (8, 9): ("Ada", 40),
    (10, 0): ("Blackwell", 50),
    (10, 3): ("Blackwell", 50),
    (11, 0): ("Blackwell", 50),
    (12, 0): ("Blackwell", 50),
    (12, 1): ("Blackwell", 50),
}


def _classify_rtx_architecture(name: str, capability: str) -> tuple[str, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)", capability.strip())
    if match is None:
        raise RuntimeError(
            f"{name} reported malformed compute capability {capability!r}; "
            "a current NVIDIA driver is required."
        )
    compute_capability = (int(match.group(1)), int(match.group(2)))
    classified = _RTX_ARCHITECTURES.get(compute_capability)
    if classified is None:
        raise RuntimeError(
            f"{name} (compute capability {capability}) is not a supported RTX architecture. "
            "This build supports Ampere, Ada, and Blackwell RTX GPUs only."
        )
    return classified


def _normalize_pci_bus_id(value: str) -> str:
    match = re.fullmatch(
        r"(?:([0-9A-Fa-f]{4,8}):)?([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])",
        value.strip(),
    )
    if match is None:
        return value.strip().upper()
    domain = int(match.group(1) or "0", 16)
    return f"{domain:04X}:{match.group(2).upper()}:{match.group(3).upper()}.{match.group(4)}"


def _cuda_device_identities() -> dict[str, dict[str, Any]]:
    """Map normalized PCI bus IDs to the CUDA ordinal and Windows DXGI LUID."""
    try:
        loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
        cuda = loader("nvcuda.dll")
    except (AttributeError, OSError):
        return {}

    c_int_p = ctypes.POINTER(ctypes.c_int)
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGetCount.argtypes = [c_int_p]
    cuda.cuDeviceGetCount.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [c_int_p, ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    cuda.cuDeviceGetPCIBusId.restype = ctypes.c_int
    get_luid = getattr(cuda, "cuDeviceGetLuid", None)
    if get_luid is not None:
        get_luid.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_int]
        get_luid.restype = ctypes.c_int

    if cuda.cuInit(0) != 0:
        return {}
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return {}
    identities: dict[str, dict[str, Any]] = {}
    for ordinal in range(max(0, count.value)):
        device = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(device), ordinal) != 0:
            continue
        bus_buffer = ctypes.create_string_buffer(32)
        if cuda.cuDeviceGetPCIBusId(bus_buffer, len(bus_buffer), device.value) != 0:
            continue
        luid_hex: str | None = None
        if get_luid is not None:
            luid = (ctypes.c_ubyte * 8)()
            node_mask = ctypes.c_uint()
            if get_luid(luid, ctypes.byref(node_mask), device.value) == 0:
                luid_hex = bytes(luid).hex()
        identities[_normalize_pci_bus_id(bus_buffer.value.decode("ascii", "replace"))] = {
            "cuda_ordinal": ordinal,
            "adapter_luid": luid_hex,
        }
    return identities


@lru_cache(maxsize=1)
def detect_gpus() -> tuple[dict[str, Any], ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "NVIDIA driver tools are unavailable; an RTX GPU and current driver are required."
        ) from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        message = "nvidia-smi failed while detecting an RTX GPU"
        raise RuntimeError(f"{message}: {detail}" if detail else f"{message}.")
    cuda_identities = _cuda_device_identities()
    devices: list[dict[str, Any]] = []
    for fallback_index, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if not parts or not any(parts):
            continue
        # Four-column rows preserve compatibility with isolated detector tests and
        # older nvidia-smi builds; production drivers use the seven-column query.
        if len(parts) == 4:
            legacy_row = True
            name, driver, memory, capability = parts
            index, uuid, pci_bus_id = str(fallback_index), f"index:{fallback_index}", ""
        elif len(parts) == 7:
            legacy_row = False
            index, uuid, pci_bus_id, name, driver, memory, capability = parts
        else:
            name = parts[3] if len(parts) > 3 else parts[0] or "NVIDIA GPU"
            raise RuntimeError(
                f"{name} returned incomplete nvidia-smi data; expected name, driver, "
                "memory, compute capability, UUID, and PCI bus ID."
            )
        if any(not value for value in (name, driver, memory, capability)):
            raise RuntimeError(
                f"{name or 'NVIDIA GPU'} returned incomplete nvidia-smi data; expected name, "
                "driver, memory, and compute capability."
            )
        try:
            memory_mb = int(memory)
            smi_index = int(index)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} reported malformed index or memory capacity through nvidia-smi."
            ) from exc
        normalized_pci = _normalize_pci_bus_id(pci_bus_id) if pci_bus_id else ""
        identity = cuda_identities.get(normalized_pci, {})
        architecture: str | None = None
        generation: int | None = None
        compatibility_error = ""
        if "RTX" in name.upper():
            try:
                architecture, generation = _classify_rtx_architecture(name, capability)
            except RuntimeError as exc:
                compatibility_error = str(exc)
        else:
            compatibility_error = "The device name does not identify an RTX GPU."
        devices.append(
            {
                "index": smi_index,
                "uuid": uuid,
                "pci_bus_id": normalized_pci,
                "name": name,
                "display_name": (
                    f"{name} (experimental RTX 30 path; may be very slow)"
                    if generation == 30
                    else name
                ),
                "driver": driver,
                "memory_mb": memory_mb,
                "compute_capability": capability,
                "architecture": architecture,
                "generation": generation,
                "beta": generation == 30,
                "ai_compatible": generation is not None,
                "compatibility_error": compatibility_error,
                "cuda_ordinal": identity.get("cuda_ordinal")
                if not legacy_row
                else smi_index,
                "adapter_luid": identity.get("adapter_luid"),
            }
        )
    if not devices:
        raise RuntimeError("No NVIDIA GPU was detected.")
    return tuple(devices)


def query_free_vram_mb() -> dict[str, int]:
    """Fresh per-GPU free VRAM by UUID; empty when the query is unavailable."""
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    free: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                free[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return free


def gpu_choice_label(gpu: dict[str, Any]) -> str:
    memory_gib = float(gpu.get("memory_mb", 0)) / 1024
    location = gpu.get("pci_bus_id") or f"index {gpu.get('index', '?')}"
    return f"{gpu.get('name', 'NVIDIA GPU')} | {memory_gib:.0f} GB | PCI {location}"


def resolve_ai_gpu(gpus: tuple[dict[str, Any], ...], gpu_uuid: str = "auto") -> dict[str, Any]:
    compatible = [gpu for gpu in gpus if gpu.get("ai_compatible")]
    if gpu_uuid != "auto":
        selected = next((gpu for gpu in compatible if gpu.get("uuid") == gpu_uuid), None)
        if selected is not None:
            return dict(selected)
        raise RuntimeError("The selected AI Processing GPU is unavailable or incompatible.")
    if not compatible:
        details = "; ".join(
            str(gpu.get("compatibility_error")) for gpu in gpus if gpu.get("compatibility_error")
        )
        message = "No supported NVIDIA RTX GPU was detected."
        raise RuntimeError(f"{message} {details}" if details else message)
    return dict(compatible[0])


def resolve_runtime_ai_gpu(
    gpus: tuple[dict[str, Any], ...],
    runtime_bundle: dict[str, Any],
    gpu_uuid: str = "auto",
) -> dict[str, Any]:
    """Resolve an AI device and enforce architecture-specific runtime pairing."""
    candidates = [
        gpu
        for gpu in gpus
        if gpu.get("ai_compatible") and gpu.get("adapter_luid")
    ]
    if gpu_uuid != "auto":
        candidates = [gpu for gpu in candidates if gpu.get("uuid") == gpu_uuid]
        if not candidates:
            raise RuntimeError("The selected AI Processing GPU is unavailable or incompatible.")
    errors: list[str] = []
    for candidate in candidates:
        try:
            validate_gpu_runtime(candidate, runtime_bundle)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        return dict(candidate)
    if errors:
        raise RuntimeError("No compatible AI Processing GPU/runtime pair is available. " + " ".join(errors))
    if not candidates:
        raise RuntimeError(
            "No compatible AI Processing GPU could be mapped to a Windows adapter LUID. "
            "Update or reinstall the NVIDIA display driver."
        )
    return resolve_ai_gpu(gpus, gpu_uuid)


@lru_cache(maxsize=32)
def _detect_gpu_cached(gpu_uuid: str = "auto") -> dict:
    return resolve_ai_gpu(detect_gpus(), gpu_uuid)


def detect_gpu(gpu_uuid: str = "auto") -> dict:
    return _detect_gpu_cached(gpu_uuid)


def _clear_gpu_detection_cache() -> None:
    _detect_gpu_cached.cache_clear()
    detect_gpus.cache_clear()


detect_gpu.cache_clear = _clear_gpu_detection_cache  # type: ignore[attr-defined]


def validate_runtime_files() -> None:
    required = [
        FFMPEG,
        FFPROBE,
        WORKER,
        RUNTIME / "dxgi.dll",
        ADDON,
        RUNTIME / "nvngx_dlss.dll",
        NEURAL_RUNTIME,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Portable runtime is incomplete:\n" + "\n".join(missing))


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = stream.read(size - len(chunks))
        if not block:
            raise EOFError(f"Native worker stopped after {len(chunks)} of {size} output bytes")
        chunks.extend(block)
    return bytes(chunks)


def _read_exact_into(stream, target: np.ndarray) -> None:
    view = memoryview(target).cast("B")
    offset = 0
    while offset < len(view):
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError(
                f"Native worker stopped after {offset} of {len(view)} output bytes"
            )
        offset += count


def _array_bytes(array: np.ndarray, dtype: np.dtype) -> memoryview:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return memoryview(contiguous).cast("B")


class DLSSFrameSession:
    """A reusable native DLSSNR feature-18 frame stream."""

    def __init__(
        self,
        *,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        frame_count: int,
        warmup_frames: int,
        factor: float,
        mode: dict[str, str | int],
        native_settings: dict[str, int | float],
        gpu: dict[str, Any],
        runtime_bundle: dict[str, Any],
        controller: JobController,
    ) -> None:
        self.controller = controller
        self._worker_log_buffer = BoundedLogBuffer()
        self.closed = False
        self.factor = factor
        self.mode = mode
        self.native_settings = native_settings
        self.gpu = gpu
        self.runtime_bundle = runtime_bundle
        reshade_log_path = RUNTIME / "ReShade.log"
        self._reshade_log_baseline_size = 0
        self._reshade_log_baseline_tail = b""
        if reshade_log_path.exists():
            self._reshade_log_baseline_size = reshade_log_path.stat().st_size
            with reshade_log_path.open("rb") as stream:
                tail_size = min(256, self._reshade_log_baseline_size)
                stream.seek(self._reshade_log_baseline_size - tail_size)
                self._reshade_log_baseline_tail = stream.read(tail_size)
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        worker_command = [str(WORKER), "--video"]
        if gpu.get("adapter_luid"):
            worker_command.extend(["--adapter-luid", str(gpu["adapter_luid"])])
        self.worker = subprocess.Popen(
            worker_command,
            cwd=RUNTIME,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
        )
        controller.register(self.worker)
        assert self.worker.stderr is not None
        self.worker_thread = threading.Thread(
            target=drain_bounded_text,
            args=(self.worker.stderr, self._worker_log_buffer),
            daemon=True,
        )
        self.worker_thread.start()
        native = native_settings
        header = struct.pack(
            VIDEO_HEADER_FORMAT,
            VIDEO_MAGIC,
            input_width,
            input_height,
            output_width,
            output_height,
            int(warmup_frames),
            frame_count,
            int(mode["perf_quality"]),
            int(native["dlss_model_preset"]),
            native["profile"],
            native["preset"],
            native["style"],
            native["auto_mask"],
            native["ui_correction"],
            native["intensity"],
            native["local_tone"],
            native["local_structure"],
            native["skin_structure"],
        )
        assert self.worker.stdin is not None and self.worker.stdout is not None
        try:
            self.worker.stdin.write(header)
            self.worker.stdin.flush()
            try:
                setup_data = _read_exact(
                    self.worker.stdout, struct.calcsize(SETUP_RESPONSE_FORMAT)
                )
            except EOFError as exc:
                worker_code = self.worker.wait(timeout=10)
                self.worker_thread.join(timeout=2)
                details = (
                    "\n".join(self.worker_logs[-60:])
                    or "The worker produced no diagnostic output."
                )
                raise RuntimeError(
                    "The native worker is incompatible with the version-4 model-preset protocol "
                    f"or failed during DLSS setup (exit {worker_code}):\n{details}"
                ) from exc
            (
                setup_magic,
                setup_ok,
                self.setup_result,
                self.render_width,
                self.render_height,
                negotiated_output_width,
                negotiated_output_height,
                self.minimum_width,
                self.minimum_height,
                self.maximum_width,
                self.maximum_height,
                self.applied_dlss_model_preset,
            ) = struct.unpack(SETUP_RESPONSE_FORMAT, setup_data)
            if setup_magic != SETUP_MAGIC:
                raise RuntimeError(
                    "The installed native worker does not support the version-4 model-preset protocol. "
                    "Rebuild it."
                )
            if not setup_ok:
                details = "\n".join(self.worker_logs[-60:])
                raise RuntimeError(
                    f"DLSS {mode['name']} is unavailable for {output_width}×{output_height} "
                    f"(NGX 0x{self.setup_result:08X}). Choose a lower upscaling factor or update "
                    "the NVIDIA driver."
                    + (f"\n{details}" if details else "")
                )
            if (negotiated_output_width, negotiated_output_height) != (
                output_width,
                output_height,
            ):
                raise RuntimeError(
                    "The native worker returned output dimensions different from the request."
                )
            requested_model_preset = int(native["dlss_model_preset"])
            if self.applied_dlss_model_preset != requested_model_preset:
                raise RuntimeError(
                    "The native worker acknowledged DLSS model preset "
                    f"{self.applied_dlss_model_preset} instead of the requested "
                    f"{requested_model_preset}."
                )
            if self.render_width < 64 or self.render_height < 64:
                raise RuntimeError(
                    f"DLSS returned an unsupported render size: "
                    f"{self.render_width}×{self.render_height}; both dimensions must be at least "
                    "64 pixels."
                )
            self.output_width = output_width
            self.output_height = output_height
            self.reconcile_bound_adapter()
        except Exception:
            self.abort()
            raise

    def reshade_log_text(self) -> str:
        """Return only ReShade output created during this worker session."""
        path = RUNTIME / "ReShade.log"
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            size = path.stat().st_size
            can_seek = size >= self._reshade_log_baseline_size
            tail = self._reshade_log_baseline_tail
            if can_seek and tail:
                stream.seek(self._reshade_log_baseline_size - len(tail))
                can_seek = stream.read(len(tail)) == tail
            stream.seek(self._reshade_log_baseline_size if can_seek else 0)
            current = stream.read()
        text = current.decode("utf-8", errors="replace")
        process_marker = f"[{self.worker.pid}]"
        process_lines = [line for line in text.splitlines() if process_marker in line]
        return "\n".join(process_lines) if process_lines else text

    @property
    def worker_logs(self) -> list[str]:
        return self._worker_log_buffer.snapshot()

    @property
    def worker_log_dropped_lines(self) -> int:
        return self._worker_log_buffer.dropped_lines

    def reshade_diagnostics(self, limit: int = 300) -> list[str]:
        """Return new log lines from this worker, favoring feature-18 evidence."""
        lines = self.reshade_log_text().splitlines()
        relevant = [
            line
            for line in lines
            if "DLSS 5 Neural Rendering" in line
            or "DLSSNR" in line
            or "feature 18" in line
            or "exception" in line.lower()
            or "failed" in line.lower()
        ]
        return (relevant or lines)[-limit:]

    def _worker_failure(self, frame_index: int, cause: BaseException) -> RuntimeError:
        worker_code = self.worker.wait(timeout=10)
        self.worker_thread.join(timeout=2)
        details = classify_worker_failure(
            worker_code=worker_code,
            frame_index=frame_index,
            worker_logs=self.worker_logs,
            reshade_lines=self.reshade_diagnostics(),
            gpu=self.gpu,
            runtime_bundle=self.runtime_bundle,
        )
        error = RuntimeError(details)
        error.__cause__ = cause
        return error

    def bound_adapter_name(self, timeout: float = 5.0) -> str | None:
        """The adapter the native worker actually bound, from its own startup log."""
        deadline = time.monotonic() + timeout
        while True:
            for line in self.worker_logs:
                match = re.search(r"\[host\] adapter \d+: (.+?) vendor=", line)
                if match:
                    return match.group(1).strip()
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def reconcile_bound_adapter(self) -> None:
        """Verify the worker honored the requested adapter; correct the report.

        --adapter-luid steers the worker, but a missing LUID (cuDeviceGetLuid
        unavailable) or an older worker leaves it on the first NVIDIA adapter.
        """
        bound = self.bound_adapter_name()
        if not bound:
            return
        self.gpu["bound_adapter"] = bound
        if bound.casefold() == str(self.gpu.get("name", "")).casefold():
            return
        if self.gpu.get("requested"):
            raise RuntimeError(
                f"The native DLSS worker bound {bound}, not the selected "
                f"{self.gpu['name']}. The worker could not honor the requested "
                "adapter; pick the GPU it bound, or update the runtime files."
            )
        replacement = next(
            (
                entry
                for entry in detect_gpus()
                if entry["name"].casefold() == bound.casefold()
            ),
            None,
        )
        if replacement is None:
            self.gpu["name"] = bound
            self.gpu["display_name"] = bound
            return
        self.gpu.update(replacement)
        validate_gpu_runtime(self.gpu, self.runtime_bundle)

    def send_frame(
        self,
        *,
        index: int,
        rgba: np.ndarray,
        motion: np.ndarray,
        reset: bool,
        pts: int,
    ) -> None:
        """Queue one frame into the worker without waiting for its result."""
        if self.controller.cancel.is_set():
            raise Cancelled("Render stopped by user.")
        assert self.worker.stdin is not None
        frame_header = struct.pack("<4Iq", FRAME_MAGIC, index, int(reset), 0, pts)
        try:
            self.worker.stdin.write(frame_header)
            self.worker.stdin.write(_array_bytes(rgba, np.dtype(np.uint8)))
            self.worker.stdin.write(_array_bytes(motion, np.dtype(np.float16)))
            self.worker.stdin.flush()
        except OSError as exc:
            raise self._worker_failure(index, exc)

    def receive_frame(self, index: int) -> tuple[np.ndarray, int]:
        """Read the result for the oldest queued frame; results arrive in order."""
        assert self.worker.stdout is not None
        try:
            result_header = _read_exact(self.worker.stdout, struct.calcsize("<5Iq"))
        except EOFError as exc:
            raise self._worker_failure(index, exc)
        magic, out_index, ok, byte_count, ngx_result, out_pts = struct.unpack(
            "<5Iq", result_header
        )
        expected = self.output_width * self.output_height * 4
        if magic != OUT_MAGIC or not ok or out_index != index or byte_count != expected:
            raise RuntimeError(f"Invalid native worker response for frame {index}")
        if ngx_result != 1:
            raise RuntimeError(
                f"Direct feature-18 evaluation failed on frame {index}: 0x{ngx_result:08X}"
            )
        output = np.empty((self.output_height, self.output_width, 4), dtype=np.uint8)
        _read_exact_into(self.worker.stdout, output)
        return output, out_pts

    def process(
        self,
        *,
        index: int,
        rgba: np.ndarray,
        motion: np.ndarray,
        reset: bool,
        pts: int,
    ) -> tuple[np.ndarray, int]:
        self.send_frame(index=index, rgba=rgba, motion=motion, reset=reset, pts=pts)
        return self.receive_frame(index)

    def close(self) -> None:
        if self.closed:
            return
        if self.worker.stdin and not self.worker.stdin.closed:
            self.worker.stdin.close()
        worker_code = self.worker.wait(timeout=60)
        self.worker_thread.join(timeout=2)
        self.controller.unregister(self.worker)
        self.closed = True
        if worker_code:
            raise RuntimeError(
                "Native DLSS worker failed:\n" + "\n".join(self.worker_logs[-40:])
            )

    def abort(self) -> None:
        if self.closed:
            return
        if self.worker.poll() is None:
            try:
                self.worker.terminate()
                self.worker.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.worker.kill()
                except OSError:
                    pass
        self.worker_thread.join(timeout=2)
        self.controller.unregister(self.worker)
        self.closed = True


def verify_feature_18(
    worker_logs: list[str], reshade_log: str | None = None
) -> dict[str, object]:
    if reshade_log is None:
        reshade_log_path = RUNTIME / "ReShade.log"
        reshade_log = (
            reshade_log_path.read_text(encoding="utf-8", errors="replace")
            if reshade_log_path.exists()
            else ""
        )
    feature_created = "feature 18 created via the signed snippet" in reshade_log
    feature_evaluated = "inline feature 18 evaluation succeeded" in reshade_log
    runtime_initialized = "signed DLSSNR 310.8.0 D3D12 runtime initialized" in reshade_log
    if not (runtime_initialized and feature_created and feature_evaluated):
        evidence = "\n".join(
            line
            for line in reshade_log.splitlines()
            if "DLSS 5 Neural Rendering" in line
            or "DLSSNR" in line
            or "feature 18" in line
        )
        raise RuntimeError(
            "The carrier render completed, but signed DLSSNR feature-18 execution was not "
            "verified.\n"
            + (evidence[-6000:] or "ReShade produced no DLSSNR evidence.")
        )
    carrier_matches = re.findall(
        r"DLSS carrier ready:.*result=0x([0-9A-Fa-f]{8})", "\n".join(worker_logs)
    )
    return {
        "reshade_log": reshade_log,
        "nr_upscaling_active": "[upscaling]" in reshade_log and feature_evaluated,
        "nr_native_fallback": "NR upscaling fell back to native" in reshade_log,
        "carrier_create_result": (
            f"0x{carrier_matches[-1].upper()}" if carrier_matches else "unreported"
        ),
        "evidence": [
            line
            for line in reshade_log.splitlines()
            if "signed DLSSNR" in line
            or "feature 18 created" in line
            or "feature 18 evaluation succeeded" in line
            or "NR upscaling fell back" in line
        ],
    }
