"""Generic ownership, reuse, and cleanup for lazily loaded engine runtimes."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import gc
import logging
from threading import Condition, Thread
from time import monotonic
from typing import Any

from tts_workbench.device_diagnostics import LoadedRuntime, collect_loaded_runtimes
from tts_workbench.domain import SynthesisRequest, SynthesisResult
from tts_workbench.engines.base import EngineRuntimeError
from tts_workbench.engines.registry import EngineRegistry
from tts_workbench.runtime_status import RuntimePhase


_LOGGER = logging.getLogger(__name__)


class CleanupState(StrEnum):
    """Device cleanup progress exposed to diagnostics and the UI."""

    NOT_REQUIRED = "not required"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RuntimeKey:
    """Stable identity of an adapter-managed model runtime."""

    engine_id: str
    model_id: str


@dataclass(frozen=True, slots=True)
class RuntimeManagerSnapshot:
    """Thread-safe lifecycle view without owning model or tensor references."""

    selected: RuntimeKey | None
    active: RuntimeKey | None
    phase: RuntimePhase
    cleanup_state: CleanupState
    cleanup_detail: str | None = None

    @property
    def cleanup_status(self) -> str:
        if self.cleanup_detail:
            return f"{self.cleanup_state.value}: {self.cleanup_detail}"
        return self.cleanup_state.value


CleanupCallback = Callable[[RuntimeKey, tuple[object, ...]], None]
CleanupScheduler = Callable[[Callable[[], None]], None]


class RuntimeManager:
    """Own the one active engine/model runtime used by the application.

    Metadata selection is synchronous and descriptor-only. When it replaces a
    loaded runtime, cleanup runs on a background daemon so the UI selection event
    can return immediately. Generation waits for any prior cleanup before asking
    the selected adapter to load or reuse its session.
    """

    def __init__(
        self,
        registry: EngineRegistry,
        *,
        cleanup_callbacks: tuple[CleanupCallback, ...] = (),
        cleanup_scheduler: CleanupScheduler | None = None,
        garbage_collector: Callable[[], int] = gc.collect,
        torch_loader: Callable[[], Any] | None = None,
    ) -> None:
        self._registry = registry
        self._condition = Condition()
        self._selected: RuntimeKey | None = None
        self._active: RuntimeKey | None = None
        self._phase = RuntimePhase.UNLOADED
        self._cleanup_state = CleanupState.NOT_REQUIRED
        self._cleanup_detail: str | None = None
        self._cleanup_running = False
        self._cleanup_target: RuntimeKey | None = None
        self._generation_running = False
        self._generation_target: RuntimeKey | None = None
        self._cleanup_callbacks = list(cleanup_callbacks)
        self._cleanup_scheduler = cleanup_scheduler or _schedule_daemon
        self._garbage_collector = garbage_collector
        self._torch_loader = torch_loader or _load_torch

        for adapter in registry:
            adapter.set_runtime_loaded_callback(self._runtime_loaded)

    def register_cleanup_callback(self, callback: CleanupCallback) -> None:
        """Invoke callback after adapter references are released during unload."""
        if not callable(callback):
            raise TypeError("cleanup callback must be callable")
        with self._condition:
            self._cleanup_callbacks.append(callback)

    def select(self, engine_id: str, model_id: str) -> RuntimeManagerSnapshot:
        """Select metadata and schedule stale-runtime cleanup without loading."""
        selected = self._require_key(engine_id, model_id)
        cleanup_task: Callable[[], None] | None = None
        with self._condition:
            self._selected = selected
            stale_active = self._active is not None and self._active != selected
            stale_generation = (
                self._generation_running and self._generation_target != selected
            )
            if stale_active:
                cleanup_task = self._request_cleanup_locked(self._active)
            elif stale_generation:
                self._phase = RuntimePhase.UNLOADING
                self._cleanup_state = CleanupState.PENDING
                self._cleanup_detail = "waiting for the current generation"
            elif self._cleanup_running:
                self._phase = RuntimePhase.UNLOADING
            elif self._active == selected:
                self._phase = (
                    RuntimePhase.GENERATING
                    if self._generation_running
                    else RuntimePhase.READY
                )
            else:
                self._phase = RuntimePhase.UNLOADED
            snapshot = self._snapshot_locked()
        self._schedule(cleanup_task)
        return snapshot

    def generate(self, request: SynthesisRequest) -> SynthesisResult:
        """Unload an incompatible runtime, then load/reuse only on generation."""
        target = self._require_key(request.engine_id, request.model_id)
        cleanup_task: Callable[[], None] | None = None
        with self._condition:
            if self._selected is not None and self._selected != target:
                raise EngineRuntimeError(
                    "engine or model selection changed before generation started"
                )
            self._selected = target
            if self._active is not None and self._active != target:
                cleanup_task = self._request_cleanup_locked(self._active)
        self._schedule(cleanup_task)

        with self._condition:
            while self._cleanup_running or self._generation_running:
                self._condition.wait()
            reuse = self._active == target
            self._generation_running = True
            self._generation_target = target
            self._phase = RuntimePhase.GENERATING if reuse else RuntimePhase.LOADING
            self._condition.notify_all()

        adapter = self._registry.get(target.engine_id)
        cleanup_task = None
        try:
            result = adapter.generate(request)
        except Exception:
            with self._condition:
                self._generation_running = False
                self._generation_target = None
                if self._selected == target:
                    self._phase = RuntimePhase.ERROR
                elif self._active == target:
                    cleanup_task = self._request_cleanup_locked(target)
                else:
                    self._phase = RuntimePhase.UNLOADED
                    if self._cleanup_state is CleanupState.PENDING:
                        self._cleanup_state = CleanupState.NOT_REQUIRED
                        self._cleanup_detail = "no runtime was loaded"
                self._condition.notify_all()
            self._schedule(cleanup_task)
            raise

        with self._condition:
            self._generation_running = False
            self._generation_target = None
            # Adapters notify immediately after caching a real session. Keep the
            # manager useful for lightweight/fake adapters that omit the hook.
            if self._active is None:
                self._active = target
            if self._selected == target:
                self._phase = RuntimePhase.READY
            else:
                cleanup_task = self._request_cleanup_locked(target)
            self._condition.notify_all()
        self._schedule(cleanup_task)
        return result

    def unload_active(self, *, wait: bool = True) -> RuntimeManagerSnapshot:
        """Explicitly release the active runtime, optionally waiting for cleanup."""
        cleanup_task: Callable[[], None] | None = None
        with self._condition:
            if self._active is not None:
                cleanup_task = self._request_cleanup_locked(self._active)
            snapshot = self._snapshot_locked()
        self._schedule(cleanup_task)
        if wait:
            self.wait_for_idle()
            return self.snapshot()
        return snapshot

    def snapshot(self) -> RuntimeManagerSnapshot:
        with self._condition:
            return self._snapshot_locked()

    def loaded_runtimes(self) -> tuple[LoadedRuntime, ...]:
        """Read adapter status without constructing a runtime."""
        return collect_loaded_runtimes(self._registry)

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait for generation and cleanup; intended for shutdown and tests."""
        deadline = monotonic() + timeout if timeout is not None else None
        with self._condition:
            while self._cleanup_running or self._generation_running:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def _require_key(self, engine_id: str, model_id: str) -> RuntimeKey:
        adapter = self._registry.get(engine_id)
        adapter.require_model(model_id)
        return RuntimeKey(engine_id, model_id)

    def _runtime_loaded(self, runtime: Any) -> None:
        if not isinstance(runtime, LoadedRuntime):
            raise TypeError("runtime-loaded callback requires LoadedRuntime")
        loaded = RuntimeKey(runtime.engine_id, runtime.model_id)
        cleanup_task: Callable[[], None] | None = None
        with self._condition:
            self._active = loaded
            if self._selected == loaded:
                self._phase = (
                    RuntimePhase.GENERATING
                    if self._generation_running
                    else RuntimePhase.READY
                )
            else:
                cleanup_task = self._request_cleanup_locked(loaded)
            self._condition.notify_all()
        self._schedule(cleanup_task)

    def _request_cleanup_locked(
        self,
        target: RuntimeKey,
    ) -> Callable[[], None] | None:
        self._phase = RuntimePhase.UNLOADING
        self._cleanup_state = CleanupState.PENDING
        self._cleanup_detail = "runtime references queued for release"
        if self._cleanup_running:
            return None
        self._cleanup_running = True
        self._cleanup_target = target
        return lambda: self._run_cleanup(target)

    def _run_cleanup(self, target: RuntimeKey) -> None:
        with self._condition:
            while self._generation_running:
                self._cleanup_detail = "waiting for the current generation"
                self._condition.wait()
            self._cleanup_state = CleanupState.RUNNING
            self._cleanup_detail = "releasing runtime and device caches"

        errors: list[str] = []
        released: tuple[object, ...] = ()
        adapter = self._registry.get(target.engine_id)
        try:
            released = adapter.unload_runtime(target.model_id)
        except Exception as error:
            errors.append(f"adapter unload failed: {error}")

        for runtime in released:
            try:
                _close_runtime(runtime)
            except Exception as error:
                errors.append(f"runtime cleanup failed: {error}")

        callbacks = tuple(self._cleanup_callbacks)
        for callback in callbacks:
            try:
                callback(target, released)
            except Exception as error:
                errors.append(f"cleanup callback failed: {error}")

        released_count = len(released)
        if released:
            # Loop targets retain their final value in Python. Remove that last
            # strong reference before collecting accelerator-backed sessions.
            del runtime
        released = ()
        try:
            collected = self._garbage_collector()
        except Exception as error:
            collected = 0
            errors.append(f"garbage collection failed: {error}")

        cache_messages, cache_errors = self._clear_device_caches()
        errors.extend(cache_errors)
        detail_parts = [
            f"released {released_count} session(s)",
            f"garbage-collected {collected} object(s)",
            *cache_messages,
        ]

        with self._condition:
            if self._active == target:
                self._active = None
            self._cleanup_running = False
            self._cleanup_target = None
            if errors:
                self._cleanup_state = CleanupState.FAILED
                self._cleanup_detail = "; ".join((*detail_parts, *errors))
                self._phase = RuntimePhase.ERROR
            else:
                self._cleanup_state = CleanupState.COMPLETED
                self._cleanup_detail = "; ".join(detail_parts)
                self._phase = (
                    RuntimePhase.READY
                    if self._selected is not None and self._active == self._selected
                    else RuntimePhase.UNLOADED
                )
            self._condition.notify_all()
        _LOGGER.info(
            "Unloaded engine=%s model=%s cleanup=%s",
            target.engine_id,
            target.model_id,
            self._cleanup_detail,
        )

    def _clear_device_caches(self) -> tuple[list[str], list[str]]:
        messages: list[str] = []
        errors: list[str] = []
        try:
            torch = self._torch_loader()
        except ModuleNotFoundError:
            return ["PyTorch cache cleanup not required"], []
        except Exception as error:
            return [], [f"PyTorch cache cleanup unavailable: {error}"]

        for backend_name in ("xpu", "cuda"):
            backend = getattr(torch, backend_name, None)
            empty_cache = getattr(backend, "empty_cache", None)
            if not callable(empty_cache):
                messages.append(f"{backend_name} cache unavailable")
                continue
            try:
                empty_cache()
            except Exception as error:
                errors.append(f"{backend_name} cache cleanup failed: {error}")
            else:
                messages.append(f"{backend_name} cache cleared")
        return messages, errors

    def _schedule(self, task: Callable[[], None] | None) -> None:
        if task is None:
            return
        try:
            self._cleanup_scheduler(task)
        except Exception as error:
            with self._condition:
                self._cleanup_running = False
                self._cleanup_target = None
                self._cleanup_state = CleanupState.FAILED
                self._cleanup_detail = f"cleanup scheduling failed: {error}"
                self._phase = RuntimePhase.ERROR
                self._condition.notify_all()

    def _snapshot_locked(self) -> RuntimeManagerSnapshot:
        return RuntimeManagerSnapshot(
            selected=self._selected,
            active=self._active,
            phase=self._phase,
            cleanup_state=self._cleanup_state,
            cleanup_detail=self._cleanup_detail,
        )


def _schedule_daemon(task: Callable[[], None]) -> None:
    Thread(target=task, name="tts-runtime-cleanup", daemon=True).start()


def _load_torch() -> Any:
    import torch

    return torch


def _close_runtime(runtime: object) -> None:
    close = getattr(runtime, "close", None)
    if callable(close):
        close()
