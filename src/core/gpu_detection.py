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


def _dxgi_adapter_luids() -> list[str]:
    """Adapter LUIDs in DXGI enumeration order (IDXGIFactory1::EnumAdapters1).

    The native worker creates its D3D12 device on the first NVIDIA adapter
    DXGI enumerates (the primary display's card comes first), which need not
    match nvidia-smi's order. This order predicts the GPU that will render.
    """
    try:
        dxgi = ctypes.WinDLL("dxgi")
    except (AttributeError, OSError):
        return []

    class _Guid(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_uint32),
            ("Data2", ctypes.c_uint16),
            ("Data3", ctypes.c_uint16),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class _Luid(ctypes.Structure):
        _fields_ = [("LowPart", ctypes.c_ulong), ("HighPart", ctypes.c_long)]

    class _AdapterDesc1(ctypes.Structure):
        _fields_ = [
            ("Description", ctypes.c_wchar * 128),
            ("VendorId", ctypes.c_uint),
            ("DeviceId", ctypes.c_uint),
            ("SubSysId", ctypes.c_uint),
            ("Revision", ctypes.c_uint),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuid", _Luid),
            ("Flags", ctypes.c_uint),
        ]

    # IID_IDXGIFactory1 = 770aae78-f26f-4dba-a829-253c83d1b387
    iid = _Guid(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(
        0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87
    ))

    def _method(instance, index: int, restype, *argtypes):
        vtable = ctypes.cast(instance, ctypes.POINTER(ctypes.c_void_p))[0]
        pointer = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(pointer)

    factory = ctypes.c_void_p()
    try:
        if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
            return []
    except (AttributeError, OSError):
        return []
    order: list[str] = []
    try:
        # IDXGIFactory1 vtable: IUnknown(3) + IDXGIObject(4) + IDXGIFactory(5)
        # puts EnumAdapters1 at slot 12; IDXGIAdapter1::GetDesc1 is slot 10.
        enum_adapters = _method(
            factory, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
        )
        position = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapters(factory, position, ctypes.byref(adapter)) != 0:
                break
            try:
                description = _AdapterDesc1()
                get_desc = _method(adapter, 10, ctypes.c_long, ctypes.POINTER(_AdapterDesc1))
                if get_desc(adapter, ctypes.byref(description)) == 0:
                    order.append(bytes(description.AdapterLuid).hex())
                else:
                    order.append("")
            finally:
                _method(adapter, 2, ctypes.c_ulong)(adapter)
            position += 1
    finally:
        _method(factory, 2, ctypes.c_ulong)(factory)
    return order


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
                "adapter_luid": identity.get("adapter_luid"),
                "d3d_adapter_index": None,
            }
        )
    if not devices:
        raise RuntimeError("No NVIDIA GPU was detected.")
    # Rank devices by the DXGI adapter order the native worker binds from.
    luid_order = _dxgi_adapter_luids()
    for device in devices:
        luid = device.get("adapter_luid")
        if luid and luid in luid_order:
            device["d3d_adapter_index"] = luid_order.index(luid)
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
