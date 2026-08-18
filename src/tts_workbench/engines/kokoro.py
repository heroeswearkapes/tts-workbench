"""Kokoro adapter backed by a lazy-loaded CPU runtime session."""

from collections.abc import Callable
import logging
from threading import Lock

from tts_workbench.domain import (
    Capability,
    ChoiceSource,
    DeviceBackend,
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
from tts_workbench.engines.kokoro_runtime import KokoroRuntimeConfig, KokoroSession
from tts_workbench.engines.kokoro_voices import KOKORO_VOICES
from tts_workbench.device_diagnostics import LoadedRuntime


_LOGGER = logging.getLogger(__name__)


class KokoroAdapter(EngineAdapter):
    """Declares Kokoro metadata and owns one lazily loaded runtime session."""

    _MODEL = ModelDescriptor(
        id="kokoro-82m",
        name="Kokoro 82M",
        description="Fast preset-voice narration model.",
    )
    _VOICES = KOKORO_VOICES
    _INPUTS = (
        InputFieldSpec(
            key="voice",
            label="Voice",
            kind=FieldKind.DROPDOWN,
            required=True,
            help_text="Choose one of Kokoro's preset voices.",
            default="af_heart",
            choice_source=ChoiceSource.VOICES,
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
            help_text="Playback speed multiplier.",
        ),
    )
    _DESCRIPTOR = EngineDescriptor(
        id="kokoro",
        name="Kokoro",
        description="Fast narration using built-in preset voices.",
        capabilities=frozenset(
            {
                Capability.PRESET_VOICES,
                Capability.SPEED_CONTROL,
                Capability.FAST_NARRATION,
            }
        ),
        default_model_id=_MODEL.id,
    )

    def __init__(
        self,
        runtime_config: KokoroRuntimeConfig | None = None,
        session_factory: Callable[[KokoroRuntimeConfig], KokoroSession] | None = None,
    ) -> None:
        self._runtime_config = runtime_config or KokoroRuntimeConfig.from_environment()
        self._session_factory = session_factory or KokoroSession.load
        self._session: KokoroSession | None = None
        self._session_lock = Lock()

    @property
    def descriptor(self) -> EngineDescriptor:
        return self._DESCRIPTOR

    def list_models(self) -> tuple[ModelDescriptor, ...]:
        return (self._MODEL,)

    def list_voices(self, model_id: str) -> tuple[VoiceDescriptor, ...]:
        self.require_model(model_id)
        return self._VOICES

    def input_schema(self, model_id: str) -> tuple[InputFieldSpec, ...]:
        self.require_model(model_id)
        return self._INPUTS

    def setting_schema(self, model_id: str) -> tuple[SettingSpec, ...]:
        self.require_model(model_id)
        return self._SETTINGS

    def create_session(self) -> KokoroSession:
        """Create and cache the Kokoro session without involving the UI."""
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    session = self._session_factory(self._runtime_config)
                    if not isinstance(session, KokoroSession):
                        raise TypeError("Kokoro session factory returned an invalid session")
                    self._session = session
                    _LOGGER.info(
                        "Loaded engine=%s model=%s device=cpu",
                        self.descriptor.id,
                        self._MODEL.id,
                    )
                    self._notify_runtime_loaded(
                        LoadedRuntime(
                            self.descriptor.id,
                            self._MODEL.id,
                            "cpu",
                            precision="fp32",
                        )
                    )
        return self._session

    def unload_runtime(self, model_id: str | None = None) -> tuple[object, ...]:
        if model_id is not None:
            self.require_model(model_id)
        with self._session_lock:
            session = self._session
            self._session = None
        return (session,) if session is not None else ()

    def runtime_status(self) -> tuple[LoadedRuntime, ...]:
        with self._session_lock:
            loaded = self._session is not None
        if not loaded:
            return ()
        return (
            LoadedRuntime(
                self.descriptor.id,
                self._MODEL.id,
                "cpu",
                precision="fp32",
            ),
        )

    def generate(self, request: SynthesisRequest) -> SynthesisResult:
        if request.engine_id != self.descriptor.id:
            raise ValueError(
                f"request engine {request.engine_id!r} does not match {self.descriptor.id!r}"
            )
        self.require_model(request.model_id)
        if request.device.backend not in {DeviceBackend.AUTO, DeviceBackend.CPU}:
            raise ValueError("Kokoro Phase 2 supports CPU inference only")
        if request.device.precision not in {Precision.AUTO, Precision.FP32}:
            raise ValueError("Kokoro CPU inference supports FP32 precision only")
        unknown_inputs = set(request.inputs) - {field.key for field in self._INPUTS}
        unknown_settings = set(request.settings) - {
            setting.key for setting in self._SETTINGS
        }
        if unknown_inputs:
            raise ValueError(f"unsupported Kokoro inputs: {sorted(unknown_inputs)}")
        if unknown_settings:
            raise ValueError(f"unsupported Kokoro settings: {sorted(unknown_settings)}")

        voice_id = request.inputs.get("voice")
        available_voice_ids = {voice.id for voice in self._VOICES}
        if not isinstance(voice_id, str) or voice_id not in available_voice_ids:
            raise ValueError(f"unknown Kokoro voice: {voice_id!r}")

        speed = request.settings.get("speed", self._SETTINGS[0].default)
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise ValueError("Kokoro speed must be numeric")
        minimum = self._SETTINGS[0].minimum
        maximum = self._SETTINGS[0].maximum
        if minimum is not None and speed < minimum:
            raise ValueError(f"Kokoro speed must be at least {minimum}")
        if maximum is not None and speed > maximum:
            raise ValueError(f"Kokoro speed must be at most {maximum}")

        return self.create_session().generate(
            text=request.text,
            engine_id=self.descriptor.id,
            model_id=request.model_id,
            voice_id=voice_id,
            speed=float(speed),
        )
