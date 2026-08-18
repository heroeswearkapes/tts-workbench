"""Normalized synthesis request passed from services to engine adapters."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from tts_workbench.domain.devices import DeviceSpec


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    engine_id: str
    model_id: str
    text: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    device: DeviceSpec = field(default_factory=DeviceSpec)

    def __post_init__(self) -> None:
        if not self.engine_id:
            raise ValueError("engine id cannot be empty")
        if not self.model_id:
            raise ValueError("model id cannot be empty")
        if not self.text.strip():
            raise ValueError("text cannot be empty")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))
