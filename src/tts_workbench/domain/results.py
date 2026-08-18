"""Engine-independent synthesis result contract."""

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    engine_id: str
    model_id: str
    sample_rate: int
    audio_path: Path | None = None
    duration_seconds: float | None = None
    voice_id: str | None = None
    generation_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("duration cannot be negative")
        if self.generation_seconds is not None and self.generation_seconds < 0:
            raise ValueError("generation time cannot be negative")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
