"""Engine-agnostic runtime lifecycle snapshots for UI and diagnostics."""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Iterable

from tts_workbench.device_diagnostics import DeviceAvailability, LoadedRuntime
from tts_workbench.domain import EngineDescriptor, ModelDescriptor, SynthesisResult


class RuntimePhase(StrEnum):
    """User-facing lifecycle states shared by every engine runtime."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    GENERATING = "generating"
    READY = "ready"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """One browser session's view of the selected engine runtime."""

    active_engine_id: str
    active_engine_name: str
    active_model_id: str
    active_model_name: str
    phase: RuntimePhase = RuntimePhase.UNLOADED
    device_backend: str | None = None
    device_name: str | None = None
    precision: str | None = None
    runtime_loaded: bool = False
    last_generation_seconds: float | None = None
    audio_duration_seconds: float | None = None
    active_runtime_engine_id: str | None = None
    active_runtime_model_id: str | None = None
    device_cleanup_status: str = "not required"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.active_engine_id:
            raise ValueError("active engine id cannot be empty")
        if not self.active_model_id:
            raise ValueError("active model id cannot be empty")
        if self.device_backend not in {None, "cpu", "xpu", "cuda"}:
            raise ValueError(f"unsupported runtime device: {self.device_backend!r}")
        if bool(self.active_runtime_engine_id) != bool(self.active_runtime_model_id):
            raise ValueError("active runtime engine and model must be set together")
        if not self.device_cleanup_status:
            raise ValueError("device cleanup status cannot be empty")
        for label, value in (
            ("generation duration", self.last_generation_seconds),
            ("audio duration", self.audio_duration_seconds),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{label} cannot be negative")

    @property
    def real_time_factor(self) -> float | None:
        return calculate_real_time_factor(
            self.last_generation_seconds,
            self.audio_duration_seconds,
        )


def calculate_real_time_factor(
    generation_seconds: float | None,
    audio_duration_seconds: float | None,
) -> float | None:
    """Return wall-clock generation time divided by generated audio duration."""
    if (
        generation_seconds is None
        or audio_duration_seconds is None
        or audio_duration_seconds <= 0
    ):
        return None
    if generation_seconds < 0:
        raise ValueError("generation duration cannot be negative")
    return generation_seconds / audio_duration_seconds


def status_for_selection(
    engine: EngineDescriptor,
    model: ModelDescriptor,
    *,
    loaded_runtimes: Iterable[LoadedRuntime] = (),
    availability: DeviceAvailability | None = None,
) -> RuntimeStatus:
    """Create an unloaded or ready status for an engine/model selection."""
    warnings = list(availability.warnings if availability is not None else ())
    if not engine.available and engine.unavailable_reason:
        warnings.append(engine.unavailable_reason)
    status = RuntimeStatus(
        active_engine_id=engine.id,
        active_engine_name=engine.name,
        active_model_id=model.id,
        active_model_name=model.name,
        warnings=tuple(warnings),
    )
    return status_after_runtime_load(
        status,
        loaded_runtimes,
        availability=availability,
    )


def status_before_generation(status: RuntimeStatus) -> RuntimeStatus:
    """Mark a request as loading or generating and clear the previous error."""
    phase = RuntimePhase.GENERATING if status.runtime_loaded else RuntimePhase.LOADING
    return replace(status, phase=phase, errors=())


def status_after_runtime_load(
    status: RuntimeStatus,
    loaded_runtimes: Iterable[LoadedRuntime],
    *,
    availability: DeviceAvailability | None = None,
) -> RuntimeStatus:
    """Merge the adapter's loaded session state into the active snapshot."""
    loaded = next(
        (
            runtime
            for runtime in loaded_runtimes
            if runtime.engine_id == status.active_engine_id
            and runtime.model_id == status.active_model_id
        ),
        None,
    )
    if loaded is None:
        return status

    backend = _device_backend(loaded.device)
    device_name = loaded.device_name or _available_device_name(
        loaded.device,
        availability,
    )
    phase = (
        RuntimePhase.GENERATING
        if status.phase in {RuntimePhase.LOADING, RuntimePhase.GENERATING}
        else RuntimePhase.READY
    )
    return replace(
        status,
        phase=phase,
        device_backend=backend,
        device_name=device_name,
        precision=loaded.precision,
        runtime_loaded=True,
        active_runtime_engine_id=loaded.engine_id,
        active_runtime_model_id=loaded.model_id,
    )


def status_with_runtime_lifecycle(
    status: RuntimeStatus,
    *,
    phase: RuntimePhase,
    active_runtime_engine_id: str | None,
    active_runtime_model_id: str | None,
    device_cleanup_status: str,
    cleanup_error: str | None = None,
) -> RuntimeStatus:
    """Merge generic manager state into the selected form's status snapshot."""
    active_matches_selection = (
        active_runtime_engine_id == status.active_engine_id
        and active_runtime_model_id == status.active_model_id
    )
    errors = (cleanup_error,) if cleanup_error else status.errors
    updates = {
        "phase": phase,
        "active_runtime_engine_id": active_runtime_engine_id,
        "active_runtime_model_id": active_runtime_model_id,
        "device_cleanup_status": device_cleanup_status,
        "errors": errors,
    }
    if not active_matches_selection:
        updates.update(
            {
                "device_backend": None,
                "device_name": None,
                "precision": None,
                "runtime_loaded": False,
            }
        )
    return replace(status, **updates)


def status_after_generation(
    status: RuntimeStatus,
    result: SynthesisResult,
) -> RuntimeStatus:
    """Record normalized result timings after successful generation."""
    if result.engine_id != status.active_engine_id:
        raise ValueError("generation result engine does not match runtime status")
    if result.model_id != status.active_model_id:
        raise ValueError("generation result model does not match runtime status")
    return replace(
        status,
        phase=RuntimePhase.READY,
        last_generation_seconds=result.generation_seconds,
        audio_duration_seconds=result.duration_seconds,
        errors=(),
    )


def status_after_error(status: RuntimeStatus, error: Exception | str) -> RuntimeStatus:
    """Record an expected validation or runtime failure without losing load state."""
    message = str(error).strip() or type(error).__name__
    return replace(status, phase=RuntimePhase.ERROR, errors=(message,))


def status_with_warning(status: RuntimeStatus, warning: str) -> RuntimeStatus:
    message = warning.strip()
    if not message or message in status.warnings:
        return status
    return replace(status, warnings=(*status.warnings, message))


def format_runtime_status(status: RuntimeStatus) -> str:
    """Render a compact Markdown table for the persistent Gradio panel."""
    generation = _format_seconds(status.last_generation_seconds)
    audio = _format_seconds(status.audio_duration_seconds)
    real_time_factor = (
        f"{status.real_time_factor:.3f}×"
        if status.real_time_factor is not None
        else "—"
    )
    lines = [
        "| Runtime | Value |",
        "| --- | --- |",
        (
            f"| Active engine | {status.active_engine_name} "
            f"(`{status.active_engine_id}`) |"
        ),
        (
            f"| Active model | {status.active_model_name} "
            f"(`{status.active_model_id}`) |"
        ),
        f"| Active runtime | {_format_active_runtime(status)} |",
        f"| State | {status.phase.value} |",
        f"| Device backend | {status.device_backend or '—'} |",
        f"| Device name | {status.device_name or '—'} |",
        f"| Precision | {status.precision or '—'} |",
        f"| Runtime loaded | {'yes' if status.runtime_loaded else 'no'} |",
        f"| Device cleanup | {status.device_cleanup_status} |",
        f"| Last generation | {generation} |",
        f"| Generated audio | {audio} |",
        f"| Real-time factor | {real_time_factor} |",
    ]
    if status.warnings:
        lines.extend(("", "**Warnings**"))
        lines.extend(f"- {warning}" for warning in status.warnings)
    if status.errors:
        lines.extend(("", "**Errors**"))
        lines.extend(f"- {error}" for error in status.errors)
    return "\n".join(lines)


def _device_backend(device: str) -> str:
    backend = device.partition(":")[0].lower()
    if backend not in {"cpu", "xpu", "cuda"}:
        raise ValueError(f"unsupported loaded runtime device: {device!r}")
    return backend


def _available_device_name(
    device: str,
    availability: DeviceAvailability | None,
) -> str | None:
    if availability is None:
        return None
    backend, _, index_value = device.partition(":")
    if backend == "cpu":
        return availability.cpu_name
    if backend != "xpu":
        return None
    index = int(index_value) if index_value else 0
    return next(
        (
            candidate.name
            for candidate in availability.xpu_devices
            if candidate.index == index
        ),
        None,
    )


def _format_seconds(value: float | None) -> str:
    return f"{value:.3f} s" if value is not None else "—"


def _format_active_runtime(status: RuntimeStatus) -> str:
    if status.active_runtime_engine_id is None:
        return "—"
    return (
        f"`{status.active_runtime_engine_id}` / "
        f"`{status.active_runtime_model_id}`"
    )
