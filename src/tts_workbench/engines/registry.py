"""Deterministic registry for engine adapter instances."""

from collections.abc import Iterator

from tts_workbench.domain.engines import EngineDescriptor
from tts_workbench.engines.base import EngineAdapter


class EngineRegistryError(RuntimeError):
    """Base class for registry failures."""


class DuplicateEngineError(EngineRegistryError):
    """Raised when two adapters use the same stable engine ID."""


class EngineNotFoundError(EngineRegistryError):
    """Raised when an engine ID is not registered."""


class EngineRegistry:
    """Stores adapters in registration order and resolves them by stable ID."""

    def __init__(self) -> None:
        self._adapters: dict[str, EngineAdapter] = {}

    def register(self, adapter: EngineAdapter) -> None:
        engine_id = adapter.descriptor.id
        if engine_id in self._adapters:
            raise DuplicateEngineError(f"engine {engine_id!r} is already registered")
        models = adapter.list_models()
        model_ids = [model.id for model in models]
        if not models:
            raise EngineRegistryError(f"engine {engine_id!r} does not expose any models")
        if len(model_ids) != len(set(model_ids)):
            raise EngineRegistryError(f"engine {engine_id!r} exposes duplicate model IDs")
        if adapter.descriptor.default_model_id not in model_ids:
            raise EngineRegistryError(
                f"default model {adapter.descriptor.default_model_id!r} is not exposed "
                f"by engine {engine_id!r}"
            )
        for model in models:
            input_keys = [field.key for field in adapter.input_schema(model.id)]
            setting_keys = [setting.key for setting in adapter.setting_schema(model.id)]
            voice_ids = [voice.id for voice in adapter.list_voices(model.id)]
            if len(input_keys) != len(set(input_keys)):
                raise EngineRegistryError(
                    f"engine {engine_id!r} model {model.id!r} has duplicate input keys"
                )
            if len(setting_keys) != len(set(setting_keys)):
                raise EngineRegistryError(
                    f"engine {engine_id!r} model {model.id!r} has duplicate setting keys"
                )
            if set(input_keys) & set(setting_keys):
                raise EngineRegistryError(
                    f"engine {engine_id!r} model {model.id!r} reuses a key across inputs "
                    "and settings"
                )
            if len(voice_ids) != len(set(voice_ids)):
                raise EngineRegistryError(
                    f"engine {engine_id!r} model {model.id!r} has duplicate voice IDs"
                )
        self._adapters[engine_id] = adapter

    def get(self, engine_id: str) -> EngineAdapter:
        try:
            return self._adapters[engine_id]
        except KeyError as error:
            raise EngineNotFoundError(f"engine {engine_id!r} is not registered") from error

    def descriptors(self) -> tuple[EngineDescriptor, ...]:
        return tuple(adapter.descriptor for adapter in self._adapters.values())

    def ids(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    def __iter__(self) -> Iterator[EngineAdapter]:
        return iter(self._adapters.values())

    def __len__(self) -> int:
        return len(self._adapters)
