"""Hardware and loaded-runtime diagnostics for local validation hosts."""

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import os
import platform
from typing import Any, Iterable, Sequence


_AUTO_IMPORT = object()
_LOGGER = logging.getLogger("tts_workbench.startup")


@dataclass(frozen=True, slots=True)
class XPUDevice:
    index: int
    name: str
    total_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class DeviceAvailability:
    cpu_available: bool
    cpu_name: str
    logical_cpu_count: int
    torch_available: bool
    torch_version: str | None
    xpu_available: bool
    xpu_devices: tuple[XPUDevice, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LoadedRuntime:
    engine_id: str
    model_id: str
    device: str
    device_name: str | None = None
    precision: str | None = None


@dataclass(frozen=True, slots=True)
class DeviceReport:
    availability: DeviceAvailability
    requested_device: str
    selected_device: str | None
    selection_error: str | None
    loaded_runtimes: tuple[LoadedRuntime, ...]

    def as_dict(self) -> dict[str, Any]:
        loaded = [asdict(runtime) for runtime in self.loaded_runtimes]
        first = loaded[0] if loaded else None
        return {
            "cpu": {
                "available": self.availability.cpu_available,
                "name": self.availability.cpu_name,
                "logical_count": self.availability.logical_cpu_count,
            },
            "torch": {
                "available": self.availability.torch_available,
                "version": self.availability.torch_version,
            },
            "xpu": {
                "available": self.availability.xpu_available,
                "count": len(self.availability.xpu_devices),
                "devices": [
                    asdict(device) for device in self.availability.xpu_devices
                ],
            },
            "requested_device": self.requested_device,
            "selected_device": self.selected_device,
            "selection_error": self.selection_error,
            "loaded_engine": first["engine_id"] if first else None,
            "loaded_model": first["model_id"] if first else None,
            "loaded_runtimes": loaded,
            "warnings": list(self.availability.warnings),
        }


def probe_device_availability(
    *,
    torch_module: Any = _AUTO_IMPORT,
) -> DeviceAvailability:
    """Probe CPU and torch.xpu without requiring Torch at import time."""
    cpu_name = platform.processor().strip() or platform.machine() or "unknown"
    logical_cpu_count = os.cpu_count() or 1
    warnings: list[str] = []

    if torch_module is _AUTO_IMPORT:
        try:
            import torch as imported_torch
        except ModuleNotFoundError:
            torch_module = None
        else:
            torch_module = imported_torch

    if torch_module is None:
        return DeviceAvailability(
            cpu_available=True,
            cpu_name=cpu_name,
            logical_cpu_count=logical_cpu_count,
            torch_available=False,
            torch_version=None,
            xpu_available=False,
            xpu_devices=(),
            warnings=("PyTorch is not installed; XPU could not be probed.",),
        )

    torch_version = str(getattr(torch_module, "__version__", "unknown"))
    xpu = getattr(torch_module, "xpu", None)
    if xpu is None:
        return DeviceAvailability(
            cpu_available=True,
            cpu_name=cpu_name,
            logical_cpu_count=logical_cpu_count,
            torch_available=True,
            torch_version=torch_version,
            xpu_available=False,
            xpu_devices=(),
            warnings=("This PyTorch build does not expose torch.xpu.",),
        )

    try:
        xpu_available = bool(xpu.is_available())
    except Exception as error:
        warnings.append(f"torch.xpu.is_available() failed: {error}")
        xpu_available = False

    devices: list[XPUDevice] = []
    if xpu_available:
        try:
            device_count = int(xpu.device_count())
        except Exception as error:
            warnings.append(f"torch.xpu.device_count() failed: {error}")
            device_count = 0
        for index in range(device_count):
            name = _xpu_device_name(xpu, index, warnings)
            total_memory = _xpu_total_memory(xpu, index, warnings)
            devices.append(XPUDevice(index, name, total_memory))
        if not devices:
            warnings.append("torch.xpu is available but reported no devices.")
            xpu_available = False

    return DeviceAvailability(
        cpu_available=True,
        cpu_name=cpu_name,
        logical_cpu_count=logical_cpu_count,
        torch_available=True,
        torch_version=torch_version,
        xpu_available=xpu_available,
        xpu_devices=tuple(devices),
        warnings=tuple(warnings),
    )


def select_device(
    availability: DeviceAvailability,
    requested_device: str = "auto",
    xpu_index: int = 0,
) -> str:
    """Resolve an explicit or automatic CPU/XPU selection."""
    requested_device = requested_device.lower()
    if requested_device == "cpu":
        return "cpu"
    if requested_device not in {"auto", "xpu"}:
        raise ValueError(f"unsupported device request: {requested_device!r}")
    if availability.xpu_available:
        indexes = {device.index for device in availability.xpu_devices}
        if xpu_index not in indexes:
            raise ValueError(
                f"XPU index {xpu_index} is unavailable; detected indexes: {sorted(indexes)}"
            )
        return f"xpu:{xpu_index}"
    if requested_device == "xpu":
        raise ValueError("XPU was requested but no usable XPU device was detected")
    return "cpu"


def collect_loaded_runtimes(registry: Iterable[Any] | None) -> tuple[LoadedRuntime, ...]:
    """Collect optional adapter status without extending EngineAdapter's contract."""
    if registry is None:
        return ()
    runtimes: list[LoadedRuntime] = []
    for adapter in registry:
        status_provider = getattr(adapter, "runtime_status", None)
        if not callable(status_provider):
            continue
        statuses = status_provider()
        for status in statuses:
            if not isinstance(status, LoadedRuntime):
                raise TypeError("adapter runtime_status() returned an invalid value")
            runtimes.append(status)
    return tuple(runtimes)


def build_device_report(
    *,
    registry: Iterable[Any] | None = None,
    requested_device: str = "auto",
    xpu_index: int = 0,
    torch_module: Any = _AUTO_IMPORT,
) -> DeviceReport:
    availability = probe_device_availability(torch_module=torch_module)
    try:
        selected_device = select_device(
            availability,
            requested_device=requested_device,
            xpu_index=xpu_index,
        )
        selection_error = None
    except ValueError as error:
        selected_device = None
        selection_error = str(error)
    return DeviceReport(
        availability=availability,
        requested_device=requested_device,
        selected_device=selected_device,
        selection_error=selection_error,
        loaded_runtimes=collect_loaded_runtimes(registry),
    )


def format_device_report(report: DeviceReport) -> str:
    availability = report.availability
    lines = [
        "TTS Workbench device diagnostics",
        f"CPU available: {'yes' if availability.cpu_available else 'no'}",
        f"CPU name: {availability.cpu_name}",
        f"Logical CPUs: {availability.logical_cpu_count}",
        f"PyTorch available: {'yes' if availability.torch_available else 'no'}",
        f"PyTorch version: {availability.torch_version or 'not installed'}",
        f"XPU available: {'yes' if availability.xpu_available else 'no'}",
        f"XPU count: {len(availability.xpu_devices)}",
    ]
    for device in availability.xpu_devices:
        memory = (
            f", {device.total_memory_bytes} bytes"
            if device.total_memory_bytes is not None
            else ""
        )
        lines.append(f"XPU {device.index}: {device.name}{memory}")
    lines.append(f"Requested device: {report.requested_device}")
    lines.append(f"Selected device: {report.selected_device or 'none'}")
    if report.selection_error:
        lines.append(f"Selection error: {report.selection_error}")
    if report.loaded_runtimes:
        for runtime in report.loaded_runtimes:
            lines.extend(
                [
                    f"Loaded engine: {runtime.engine_id}",
                    f"Loaded model: {runtime.model_id}",
                    f"Loaded device: {runtime.device}",
                ]
            )
    else:
        lines.extend(["Loaded engine: none", "Loaded model: none"])
    lines.extend(f"Warning: {warning}" for warning in availability.warnings)
    return "\n".join(lines)


def log_startup_report(
    report: DeviceReport,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Log concise startup hardware and runtime state."""
    logger = logger or _LOGGER
    availability = report.availability
    logger.info(
        "CPU available=%s name=%s logical_cpus=%s",
        availability.cpu_available,
        availability.cpu_name,
        availability.logical_cpu_count,
    )
    logger.info(
        "XPU available=%s count=%s torch=%s",
        availability.xpu_available,
        len(availability.xpu_devices),
        availability.torch_version or "not installed",
    )
    for device in availability.xpu_devices:
        logger.info(
            "XPU device index=%s name=%s total_memory_bytes=%s",
            device.index,
            device.name,
            device.total_memory_bytes,
        )
    logger.info("Selected device=%s", report.selected_device or "none")
    if report.selection_error:
        logger.error("Device selection error=%s", report.selection_error)
    if report.loaded_runtimes:
        for runtime in report.loaded_runtimes:
            logger.info(
                "Loaded engine=%s model=%s device=%s",
                runtime.engine_id,
                runtime.model_id,
                runtime.device,
            )
    else:
        logger.info("Loaded engine=none model=none")
    for warning in availability.warnings:
        logger.warning("Device diagnostic warning=%s", warning)


def _xpu_device_name(xpu: Any, index: int, warnings: list[str]) -> str:
    try:
        return str(xpu.get_device_name(index))
    except Exception as error:
        warnings.append(f"torch.xpu.get_device_name({index}) failed: {error}")
        return "unknown Intel XPU"


def _xpu_total_memory(xpu: Any, index: int, warnings: list[str]) -> int | None:
    try:
        properties = xpu.get_device_properties(index)
    except Exception as error:
        warnings.append(f"torch.xpu.get_device_properties({index}) failed: {error}")
        return None
    value = getattr(properties, "total_memory", None)
    return int(value) if value is not None else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report CPU/XPU availability")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu"),
        default="auto",
        help="Device selection to validate (default: auto)",
    )
    parser.add_argument(
        "--xpu-index",
        type=int,
        default=0,
        help="XPU index selected for auto/xpu requests (default: 0)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    from tts_workbench.bootstrap import create_default_registry

    report = build_device_report(
        registry=create_default_registry(),
        requested_device=args.device,
        xpu_index=args.xpu_index,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(format_device_report(report))
    return 2 if report.selection_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
