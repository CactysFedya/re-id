from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from typing import Any


def build_system_snapshot() -> dict[str, Any]:
    return {
        "cpu": _build_cpu_snapshot(),
        "gpu": _build_gpu_snapshot(),
        "ram": _build_ram_snapshot(),
    }


def _build_cpu_snapshot() -> dict[str, Any]:
    return {
        "name": _cpu_name(),
        "physical_cores": _physical_core_count(),
        "logical_cores": os.cpu_count(),
    }


def _build_gpu_snapshot() -> dict[str, Any]:
    torch_snapshot = _gpu_from_torch()
    if torch_snapshot is not None:
        return torch_snapshot

    nvidia_snapshot = _gpu_from_nvidia_smi()
    if nvidia_snapshot is not None:
        return nvidia_snapshot

    return {
        "backend": None,
        "cuda_available": False,
        "devices": [],
    }


def _build_ram_snapshot() -> dict[str, Any]:
    total_bytes = _total_memory_bytes()
    return {
        "total_bytes": total_bytes,
        "total_gb": _bytes_to_gb(total_bytes),
    }


def _gpu_from_torch() -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_bytes": int(props.total_memory),
                    "total_memory_gb": _bytes_to_gb(int(props.total_memory)),
                }
            )

    return {
        "backend": "torch",
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "device_count": len(devices),
        "devices": devices,
    }


def _gpu_from_nvidia_smi() -> dict[str, Any] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None

    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    devices: list[dict[str, Any]] = []
    for idx, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue

        name, memory_mb_text, driver_version = parts
        try:
            memory_mb = int(memory_mb_text)
        except ValueError:
            memory_mb = None

        devices.append(
            {
                "index": idx,
                "name": name,
                "total_memory_mb": memory_mb,
                "total_memory_gb": _bytes_to_gb(memory_mb * 1024 * 1024) if memory_mb is not None else None,
                "driver_version": driver_version,
            }
        )

    return {
        "backend": "nvidia-smi",
        "cuda_available": False,
        "device_count": len(devices),
        "devices": devices,
    }


def _cpu_name() -> str | None:
    if platform.system().lower() == "windows":
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
                capture_output=True,
                text=True,
                check=True,
            )
            value = result.stdout.strip()
            if value:
                return value
        except Exception:
            pass

    processor = platform.processor().strip()
    if processor:
        return processor

    if platform.system().lower() == "windows":
        return os.environ.get("PROCESSOR_IDENTIFIER")

    return None


def _physical_core_count() -> int | None:
    if platform.system().lower() != "windows":
        return None

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
            capture_output=True,
            text=True,
            check=True,
        )
        value = result.stdout.strip()
        return int(value) if value else None
    except Exception:
        return None


def _total_memory_bytes() -> int | None:
    if platform.system().lower() == "windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            return int(memory_status.ullTotalPhys)
        return None

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            return int(page_size * pages)
        except (ValueError, OSError):
            return None

    return None


def _bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 ** 3), 2)
