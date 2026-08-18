"""Abstract interface implemented by every TTS engine adapter."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from tts_workbench.domain.engines import (
    EngineDescriptor,
    InputFieldSpec,
    ModelDescriptor,
    SettingSpec,
    VoiceDescriptor,
)
from tts_workbench.domain.requests import SynthesisRequest
from tts_workbench.domain.results import SynthesisResult


class EngineRuntimeError(RuntimeError):
    """Base class for expected engine loading and generation failures."""


class EngineDependencyError(EngineRuntimeError):
    """Raised when an optional engine dependency is not installed."""


class EngineOutputError(EngineRuntimeError):
    """Raised when an engine produces missing or invalid audio."""


class InferenceNotImplementedError(EngineRuntimeError):
    """Raised when an adapter does not yet implement generation."""


class EngineAdapter(ABC):
    """Lightweight engine metadata and generation boundary.

    Implementations must not import Gradio or load model weights while exposing
    descriptors. Real adapters may lazily create and cache runtime sessions later.
    """

    @property
    @abstractmethod
    def descriptor(self) -> EngineDescriptor:
        """Return stable engine metadata and capabilities."""

    @abstractmethod
    def list_models(self) -> tuple[ModelDescriptor, ...]:
        """Return models exposed by this adapter."""

    @abstractmethod
    def list_voices(self, model_id: str) -> tuple[VoiceDescriptor, ...]:
        """Return preset voices for a model, or an empty tuple."""

    @abstractmethod
    def input_schema(self, model_id: str) -> tuple[InputFieldSpec, ...]:
        """Return required and optional engine-specific inputs."""

    @abstractmethod
    def setting_schema(self, model_id: str) -> tuple[SettingSpec, ...]:
        """Return generation settings supported by a model."""

    @abstractmethod
    def generate(self, request: SynthesisRequest) -> SynthesisResult:
        """Generate audio for a normalized request."""

    def require_model(self, model_id: str) -> ModelDescriptor:
        """Resolve a model ID or raise a consistent error."""
        for model in self.list_models():
            if model.id == model_id:
                return model
        raise KeyError(f"unknown model {model_id!r} for engine {self.descriptor.id!r}")

    def runtime_status(self) -> tuple[Any, ...]:
        """Return loaded runtime descriptors without creating a session."""
        return ()

    def unload_runtime(self, model_id: str | None = None) -> tuple[object, ...]:
        """Release cached sessions and return references for generic cleanup."""
        if model_id is not None:
            self.require_model(model_id)
        return ()

    def set_runtime_loaded_callback(
        self,
        callback: Callable[[Any], None] | None,
    ) -> None:
        """Register the manager hook called after a session enters the cache."""
        self._runtime_loaded_callback = callback

    def _notify_runtime_loaded(self, runtime: Any) -> None:
        callback = getattr(self, "_runtime_loaded_callback", None)
        if callback is not None:
            callback(runtime)
