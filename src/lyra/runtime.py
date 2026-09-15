"""Shared platform checks for diagnostics and generation."""
from __future__ import annotations

import platform
import re


def platform_error(system: str, machine: str, macos: str, device_name: str = "") -> str | None:
    """Validate the platform without confusing M5 requirements with older Macs."""
    if system != "Darwin" or machine.lower() != "arm64":
        return "Native Apple Silicon Python on macOS is required (not Intel/Rosetta)"
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", macos)
    if match is None:
        return f"Cannot determine macOS version: {macos!r}"
    release = tuple(map(int, match.groups()))
    if release < (14, 2):
        return "mlx-Yue requires macOS >=14.2"
    if re.search(r"\bM5\b", device_name, re.IGNORECASE) and release < (26, 2):
        return "The MLX M5 runtime requires macOS >=26.2"
    return None


def runtime_status() -> dict:
    system, machine, macos = platform.system(), platform.machine(), platform.mac_ver()[0]
    error = platform_error(system, machine, macos)
    device = {}
    metal = False
    if error is None:
        import mlx.core as mx

        metal = mx.metal.is_available()
        if metal:
            device = mx.device_info()
            error = platform_error(system, machine, macos, device.get("device_name", ""))
        else:
            error = "MLX Metal is required"
    return {"supported": error is None, "error": error, "system": system,
            "machine": machine, "macos": macos, "metal": metal, "device": device}


def require_supported_runtime() -> dict:
    status = runtime_status()
    if not status["supported"]:
        raise RuntimeError(status["error"])
    return status
