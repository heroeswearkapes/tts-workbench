"""F5-TTS adapter backed by lazy CPU or XPU runtime sessions."""

from collections.abc import Callable
import logging
from pathlib import Path
from threading import Lock

from tts_workbench.domain import (
    Capability,
    DeviceBackend,
    DeviceSpec,
    EngineDescriptor,
    FieldKind,
    InputFieldSpec,
    ModelDescriptor,
    Precision,
    SettingSpec,
    SynthesisRequest,
    SynthesisResult,
    VoiceDescriptor,
)
from tts_workbench.engines.base import EngineAdapter
from tts_workbench.engines.f5_runtime import F5RuntimeConfig, F5Session
from tts_workbench.device_diagnostics import LoadedRuntime


_LOGGER = logging.getLogger(__name__)


class F5Adapter(EngineAdapter):
    """Declares F5 inputs and owns lazy sessions keyed by inference device."""

    _MODEL = ModelDescriptor(
        id="f5-tts-v1-base",
        name="F5-TTS v1 Base",
        description="Reference-audio voice cloning model.",
        supported_backends=(DeviceBackend.CPU, DeviceBackend.XPU),
        supported_precisions=(Precision.FP32,),
    )
    _INPUTS = (
        InputFieldSpec(
            key="reference_audio",
            label="Reference audio",
            kind=FieldKind.AUDIO,
            required=True,
            help_text="Upload a clean voice sample to clone.",
            file_types=(".wav", ".flac", ".mp3", ".m4a", ".ogg"),
        ),
        InputFieldSpec(
            key="reference_text",
            label="Reference transcript",
            kind=FieldKind.MULTILINE_TEXT,
            required=False,
            help_text=(
                "Enter the exact words spoken in the reference audio. Leave blank to "
                "transcribe on CPU."
            ),
        ),
    )
    _SETTINGS = (
        SettingSpec(
            key="speed",
            label="Speed",
            kind=FieldKind.SLIDER,
            default=1.0,
            minimum=0.5,
            maximum=2.0,
            step=0.05,
            help_text="Speech speed multiplier.",
        ),
    )
    _DESCRIPTOR = EngineDescriptor(
        id="f5",
        name="F5-TTS",
        description="Voice cloning from reference audio and its transcript.",
        capabilities=frozenset(
            {
                Capability.REFERENCE_AUDIO,
                Capability.REFERENCE_TEXT,
                Capability.VOICE_CLONING,
                Capability.SPEED_CONTROL,
            }
        ),
        default_model_id=_MODEL.id,
    )

    def __init__(
        self,
        runtime_config: F5RuntimeConfig | None = None,
        session_factory: Callable[[F5RuntimeConfig], F5Session] | None = None,
        device_resolver: Callable[[DeviceSpec], str] | None = None,
    ) -> None:
        self._runtime_config = runtime_config or F5RuntimeConfig.from_environment()
        self._session_factory = session_factory or F5Session.load
        self._device_resolver = device_resolver or resolve_f5_device
        self._sessions: dict[str, F5Session] = {}
        self._session_lock = Lock()

    @property
    def descriptor(self) -> EngineDescriptor:
        return self._DESCRIPTOR

    def list_models(self) -> tuple[ModelDescriptor, ...]:
        return (self._MODEL,)

    def list_voices(self, model_id: str) -> tuple[VoiceDescriptor, ...]:
        self.require_model(model_id)
        return ()

    def input_schema(self, model_id: str) -> tuple[InputFieldSpec, ...]:
        self.require_model(model_id)
        return self._INPUTS

    def setting_schema(self, model_id: str) -> tuple[SettingSpec, ...]:
        self.require_model(model_id)
        return self._SETTINGS

    def create_session(self, device: str) -> F5Session:
        """Create and cache one F5 session per resolved inference device."""
        if device not in self._sessions:
            with self._session_lock:
                if device not in self._sessions:
                    config = self._runtime_config.for_device(device)
                    session = self._session_factory(config)
                    if not isinstance(session, F5Session):
                        raise TypeError("F5 session factory returned an invalid session")
                    self._sessions[device] = session
                    _LOGGER.info(
                        "Loaded engine=%s model=%s device=%s",
                        self.descriptor.id,
                        self._MODEL.id,
                        device,
                    )
                    self._notify_runtime_loaded(
                        LoadedRuntime(
                            self.descriptor.id,
                            self._MODEL.id,
                            device,
                            precision="fp32",
                        )
                    )
        return self._sessions[device]

    def unload_runtime(self, model_id: str | None = None) -> tuple[object, ...]:
        if model_id is not None:
            self.require_model(model_id)
        with self._session_lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        return sessions

    def runtime_status(self) -> tuple[LoadedRuntime, ...]:
        with self._session_lock:
            devices = tuple(self._sessions)
        return tuple(
            LoadedRuntime(
                self.descriptor.id,
                self._MODEL.id,
                device,
                precision="fp32",
            )
            for device in devices
        )

    def generate(self, request: SynthesisRequest) -> SynthesisResult:
        if request.engine_id != self.descriptor.id:
            raise ValueError(
                f"request engine {request.engine_id!r} does not match {self.descriptor.id!r}"
            )
        self.require_model(request.model_id)
        if request.device.backend not in {
            DeviceBackend.AUTO,
            DeviceBackend.CPU,
            DeviceBackend.XPU,
        }:
            raise ValueError("F5-TTS supports CPU and XPU inference only")
        if request.device.precision not in {Precision.AUTO, Precision.FP32}:
            raise ValueError("F5-TTS currently supports FP32 precision only")

        unknown_inputs = set(request.inputs) - {field.key for field in self._INPUTS}
        unknown_settings = set(request.settings) - {
            setting.key for setting in self._SETTINGS
        }
        if unknown_inputs:
            raise ValueError(f"unsupported F5 inputs: {sorted(unknown_inputs)}")
        if unknown_settings:
            raise ValueError(f"unsupported F5 settings: {sorted(unknown_settings)}")

        reference_audio_value = request.inputs.get("reference_audio")
        if not isinstance(reference_audio_value, (str, Path)) or not str(
            reference_audio_value
        ).strip():
            raise ValueError("F5 reference audio is required")
        reference_audio = Path(reference_audio_value).expanduser().resolve()
        if not reference_audio.is_file():
            raise ValueError(f"F5 reference audio does not exist: {reference_audio}")

        reference_text_value = request.inputs.get("reference_text", "")
        if reference_text_value is None:
            reference_text = ""
        elif isinstance(reference_text_value, str):
            reference_text = reference_text_value.strip()
        else:
            raise ValueError("F5 reference transcript must be text")

        speed = request.settings.get("speed", self._SETTINGS[0].default)
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise ValueError("F5 speed must be numeric")
        minimum = self._SETTINGS[0].minimum
        maximum = self._SETTINGS[0].maximum
        if minimum is not None and speed < minimum:
            raise ValueError(f"F5 speed must be at least {minimum}")
        if maximum is not None and speed > maximum:
            raise ValueError(f"F5 speed must be at most {maximum}")

        device = self._device_resolver(request.device)
        return self.create_session(device).generate(
            text=request.text,
            engine_id=self.descriptor.id,
            model_id=request.model_id,
            reference_audio=reference_audio,
            reference_text=reference_text,
            speed=float(speed),
        )


def resolve_f5_device(device: DeviceSpec) -> str:
    """Resolve auto/CPU/XPU without importing torch during adapter discovery."""
    if device.backend is DeviceBackend.CPU:
        return "cpu"
    if device.backend is DeviceBackend.XPU:
        return f"xpu:{device.index}" if device.index is not None else "xpu"
    if device.backend is not DeviceBackend.AUTO:
        raise ValueError(f"unsupported F5 device backend: {device.backend.value}")
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    xpu = getattr(torch, "xpu", None)
    return "xpu" if xpu is not None and xpu.is_available() else "cpu"
