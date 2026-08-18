"""Declarative engine, model, voice, input, and setting schemas."""

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any

from tts_workbench.domain.capabilities import Capability
from tts_workbench.domain.devices import DeviceBackend, Precision


_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _validate_id(value: str, label: str) -> None:
    if not _STABLE_ID.fullmatch(value):
        raise ValueError(f"{label} must be a stable lowercase identifier: {value!r}")


def _validate_model_id(value: str, label: str) -> None:
    if not _MODEL_ID.fullmatch(value) or "//" in value or value.endswith("/"):
        raise ValueError(f"{label} must be a valid model or repository identifier: {value!r}")


class FieldKind(StrEnum):
    DROPDOWN = "dropdown"
    TEXT = "text"
    MULTILINE_TEXT = "multiline_text"
    AUDIO = "audio"
    NUMBER = "number"
    SLIDER = "slider"
    CHECKBOX = "checkbox"


class ChoiceSource(StrEnum):
    STATIC = "static"
    VOICES = "voices"


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    id: str
    name: str
    description: str = ""
    supported_backends: tuple[DeviceBackend, ...] = (DeviceBackend.CPU,)
    supported_precisions: tuple[Precision, ...] = (Precision.FP32,)

    def __post_init__(self) -> None:
        _validate_model_id(self.id, "model id")
        if not self.name.strip():
            raise ValueError("model name cannot be empty")
        if not self.supported_backends:
            raise ValueError("model must support at least one device backend")
        if not self.supported_precisions:
            raise ValueError("model must support at least one precision")


@dataclass(frozen=True, slots=True)
class VoiceDescriptor:
    id: str
    name: str
    language: str | None = None
    description: str = ""
    accent: str | None = None
    gender: str | None = None
    category: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.id, "voice id")
        if not self.name.strip():
            raise ValueError("voice name cannot be empty")
        for label, value in (
            ("voice language", self.language),
            ("voice accent", self.accent),
            ("voice gender", self.gender),
            ("voice category", self.category),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{label} cannot be blank")


@dataclass(frozen=True, slots=True)
class InputFieldSpec:
    key: str
    label: str
    kind: FieldKind
    required: bool = False
    help_text: str = ""
    default: Any = None
    choice_source: ChoiceSource | None = None
    choices: tuple[tuple[str, str], ...] = ()
    file_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_id(self.key, "input key")
        if not self.label.strip():
            raise ValueError("input label cannot be empty")
        if self.kind is FieldKind.DROPDOWN and self.choice_source is None:
            raise ValueError("dropdown inputs require a choice source")
        if self.choice_source is ChoiceSource.STATIC and not self.choices:
            raise ValueError("static dropdown inputs require choices")
        if self.kind is not FieldKind.DROPDOWN and self.choice_source is not None:
            raise ValueError("only dropdown inputs can have a choice source")


@dataclass(frozen=True, slots=True)
class SettingSpec:
    key: str
    label: str
    kind: FieldKind
    default: Any
    help_text: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    advanced: bool = False

    def __post_init__(self) -> None:
        _validate_id(self.key, "setting key")
        if not self.label.strip():
            raise ValueError("setting label cannot be empty")
        if self.kind not in {
            FieldKind.DROPDOWN,
            FieldKind.NUMBER,
            FieldKind.SLIDER,
            FieldKind.CHECKBOX,
        }:
            raise ValueError(f"unsupported setting field kind: {self.kind.value}")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("setting minimum cannot exceed maximum")
        if self.kind is FieldKind.DROPDOWN and not self.choices:
            raise ValueError("dropdown settings require choices")


@dataclass(frozen=True, slots=True)
class EngineDescriptor:
    id: str
    name: str
    description: str
    capabilities: frozenset[Capability]
    default_model_id: str
    adapter_version: str = "1"
    available: bool = True
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.id, "engine id")
        _validate_model_id(self.default_model_id, "default model id")
        if not self.name.strip():
            raise ValueError("engine name cannot be empty")
        if self.available and self.unavailable_reason:
            raise ValueError("available engines cannot have an unavailable reason")
        if not self.available and not self.unavailable_reason:
            raise ValueError("unavailable engines must provide a reason")
