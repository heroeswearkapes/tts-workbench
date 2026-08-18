"""Engine-independent domain contracts."""

from tts_workbench.domain.capabilities import Capability
from tts_workbench.domain.devices import DeviceBackend, DeviceSpec, Precision
from tts_workbench.domain.engines import (
    ChoiceSource,
    EngineDescriptor,
    FieldKind,
    InputFieldSpec,
    ModelDescriptor,
    SettingSpec,
    VoiceDescriptor,
)
from tts_workbench.domain.requests import SynthesisRequest
from tts_workbench.domain.results import SynthesisResult

__all__ = [
    "Capability",
    "ChoiceSource",
    "DeviceBackend",
    "DeviceSpec",
    "EngineDescriptor",
    "FieldKind",
    "InputFieldSpec",
    "ModelDescriptor",
    "Precision",
    "SettingSpec",
    "SynthesisRequest",
    "SynthesisResult",
    "VoiceDescriptor",
]
