from __future__ import annotations

import ctypes
import re
import subprocess
from functools import lru_cache
from typing import Any


_RTX_ARCHITECTURES: dict[tuple[int, int], tuple[str, int]] = {
    (7, 5): ("Turing", 20),
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


def _classify_rtx_architecture(name: str, capability: str) -> tuple[str, int | None]:
    """Best-effort RTX metadata; never use this to decide whether an RTX GPU may run."""
    match = re.fullmatch(r"(\d+)\.(\d+)", capability.strip())
    if match is not None:
        compute_capability = (int(match.group(1)), int(match.group(2)))
        classified = _RTX_ARCHITECTURES.get(compute_capability)
        if classified is not None:
            return classified

    name_match = re.search(r"\bRTX\s*(20|30|40|50)\d{2}\b", name.upper())
    if name_match is not None:
        generation = int(name_match.group(1))
        architecture = {20: "Turing", 30: "Ampere", 40: "Ada", 50: "Blackwell"}.get(
            generation, "Unknown"
        )
        return architecture, generation
    return "Unknown RTX", None


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
    """Map normalized PCI bus IDs to CUDA ordinals for optional NVENC selection."""
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
        identities[_normalize_pci_bus_id(bus_buffer.value.decode("ascii", "replace"))] = {
            "cuda_ordinal": ordinal,
        }
    return identities


@lru_cache(maxsize=1)
def detect_gpus() -> tuple[dict[str, Any], ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    fallback_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        used_fallback_query = False
        if result.returncode:
            result = subprocess.run(
                fallback_command, capture_output=True, text=True, timeout=10
            )
            used_fallback_query = True
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
        if len(parts) == 4:
            legacy_row = True
            name, driver, memory, capability = parts
            index, uuid, pci_bus_id = str(fallback_index), f"index:{fallback_index}", ""
        elif len(parts) == 6 and used_fallback_query:
            legacy_row = False
            index, uuid, pci_bus_id, name, driver, memory = parts
            capability = "unknown"
        elif len(parts) == 7:
            legacy_row = False
            index, uuid, pci_bus_id, name, driver, memory, capability = parts
        else:
            name = parts[3] if len(parts) > 3 else parts[0] or "NVIDIA GPU"
            raise RuntimeError(
                f"{name} returned incomplete nvidia-smi data; expected name, driver, "
                "memory, compute capability, UUID, and PCI bus ID."
            )
        if any(not value for value in (name, driver, memory)):
            raise RuntimeError(
                f"{name or 'NVIDIA GPU'} returned incomplete nvidia-smi data; expected name, "
                "driver, and memory capacity."
            )
        capability = capability or "unknown"
        try:
            memory_mb = int(memory)
            smi_index = int(index)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} reported malformed index or memory capacity through nvidia-smi."
            ) from exc
        normalized_pci = _normalize_pci_bus_id(pci_bus_id) if pci_bus_id else ""
        identity = cuda_identities.get(normalized_pci, {})
        is_rtx = "RTX" in name.upper()
        architecture: str | None = None
        generation: int | None = None
        compatibility_error = ""
        if is_rtx:
            architecture, generation = _classify_rtx_architecture(name, capability)
        else:
            compatibility_error = "The device name does not identify an RTX GPU."
        devices.append(
            {
                "index": smi_index,
                "uuid": uuid,
                "pci_bus_id": normalized_pci,
                "name": name,
                "display_name": name,
                "driver": driver,
                "memory_mb": memory_mb,
                "compute_capability": capability,
                "architecture": architecture,
                "generation": generation,
                "beta": False,
                "ai_compatible": is_rtx,
                "compatibility_error": compatibility_error,
                "cuda_ordinal": (
                    smi_index if legacy_row else identity.get("cuda_ordinal", smi_index)
                ),
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


def clear_gpu_detection_cache() -> None:
    detect_gpus.cache_clear()
