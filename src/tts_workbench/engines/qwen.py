"""Qwen3-TTS adapter backed by lazy CPU or XPU runtime sessions."""

from collections.abc import Callable
import logging
from threading import Lock

from tts_workbench.domain import (
    Capability,
    ChoiceSource,
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
from tts_workbench.device_diagnostics import (
    LoadedRuntime,
    probe_device_availability,
    select_device,
)
from tts_workbench.engines.base import EngineAdapter, InferenceNotImplementedError
from tts_workbench.engines.qwen_runtime import (
    QWEN_BASE_0_6B,
    QWEN_BASE_1_7B,
    QWEN_CUSTOM_VOICE_0_6B,
    QWEN_CUSTOM_VOICE_1_7B,
    QWEN_MODEL_ID,
    QWEN_VOICE_DESIGN_1_7B,
    QwenRuntimeConfig,
    QwenSession,
)


_LOGGER = logging.getLogger(__name__)


_LANGUAGE_CHOICES = (
    ("Auto detect", "auto"),
    ("Chinese", "chinese"),
    ("English", "english"),
    ("Japanese", "japanese"),
    ("Korean", "korean"),
    ("German", "german"),
    ("French", "french"),
    ("Russian", "russian"),
    ("Portuguese", "portuguese"),
    ("Spanish", "spanish"),
    ("Italian", "italian"),
)

_LANGUAGE_INPUT = InputFieldSpec(
    key="language",
    label="Language",
    kind=FieldKind.DROPDOWN,
    required=True,
    default="auto",
    choice_source=ChoiceSource.STATIC,
    choices=_LANGUAGE_CHOICES,
    help_text="Select a synthesis language or let Qwen detect it from the text.",
)

_VOICE_INPUT = InputFieldSpec(
    key="voice",
    label="Voice",
    kind=FieldKind.DROPDOWN,
    required=True,
    default="vivian",
    choice_source=ChoiceSource.VOICES,
    help_text="Choose one of Qwen3-TTS's preset speakers.",
)

_STYLE_INPUT = InputFieldSpec(
    key="style_instruction",
    label="Style instruction",
    kind=FieldKind.MULTILINE_TEXT,
    required=False,
    help_text=(
        "Recorded and forwarded to Qwen. The initial 0.6B checkpoint currently "
        "ignores instruction control upstream."
    ),
)

_VOICE_DESIGN_INPUT = InputFieldSpec(
    key="voice_description",
    label="Voice design description",
    kind=FieldKind.MULTILINE_TEXT,
    required=True,
    help_text="Describe the desired speaker identity, timbre, emotion, and delivery.",
)

_REFERENCE_AUDIO_INPUT = InputFieldSpec(
    key="reference_audio",
    label="Reference audio",
    kind=FieldKind.AUDIO,
    required=True,
    help_text="Upload a short reference recording for voice cloning.",
    file_types=(".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"),
)

_REFERENCE_TEXT_INPUT = InputFieldSpec(
    key="reference_text",
    label="Reference transcript",
    kind=FieldKind.MULTILINE_TEXT,
    required=False,
    help_text=(
        "Enter the words spoken in the reference audio. Required for the default "
        "high-fidelity cloning mode."
    ),
)

_SAMPLING_SETTINGS = (
    SettingSpec(
        key="temperature",
        label="Temperature",
        kind=FieldKind.SLIDER,
        default=0.9,
        minimum=0.1,
        maximum=2.0,
        step=0.05,
        advanced=True,
        help_text="Sampling temperature for speech token generation.",
    ),
    SettingSpec(
        key="top_p",
        label="Top P",
        kind=FieldKind.SLIDER,
        default=0.8,
        minimum=0.1,
        maximum=1.0,
        step=0.05,
        advanced=True,
        help_text="Nucleus sampling probability threshold.",
    ),
    SettingSpec(
        key="top_k",
        label="Top K",
        kind=FieldKind.NUMBER,
        default=50,
        minimum=1,
        maximum=1000,
        step=1,
        advanced=True,
        help_text="Maximum candidate speech tokens considered per step.",
    ),
    SettingSpec(
        key="repetition_penalty",
        label="Repetition penalty",
        kind=FieldKind.SLIDER,
        default=1.05,
        minimum=0.5,
        maximum=2.0,
        step=0.05,
        advanced=True,
        help_text="Penalty used to reduce repeated speech tokens.",
    ),
)

_SPEAKER_EMBEDDING_ONLY = SettingSpec(
    key="speaker_embedding_only",
    label="Speaker embedding only",
    kind=FieldKind.CHECKBOX,
    default=False,
    help_text=(
        "Clone without a transcript using only the speaker embedding. This may "
        "reduce cloning quality."
    ),
)


class QwenAdapter(EngineAdapter):
    """Declare Qwen controls and cache sessions by model/device/precision."""

    _CUSTOM_0_6B = ModelDescriptor(
        id=QWEN_CUSTOM_VOICE_0_6B,
        name="Qwen3-TTS 12Hz 0.6B CustomVoice",
        description="Atlas-validated compact multilingual preset-voice runtime.",
        supported_backends=(DeviceBackend.CPU, DeviceBackend.XPU),
        supported_precisions=(Precision.BF16, Precision.FP32),
    )
    _BASE_0_6B = ModelDescriptor(
        id=QWEN_BASE_0_6B,
        name="Qwen3-TTS 12Hz 0.6B Base",
        description="Compact voice-cloning model; inference is not integrated yet.",
        supported_backends=(DeviceBackend.CPU, DeviceBackend.XPU),
        supported_precisions=(Precision.BF16, Precision.FP32),
    )
    _CUSTOM_1_7B = ModelDescriptor(
        id=QWEN_CUSTOM_VOICE_1_7B,
        name="Qwen3-TTS 12Hz 1.7B CustomVoice",
        description="Larger preset/style model; inference is not integrated yet.",
        supported_backends=(DeviceBackend.CPU, DeviceBackend.XPU),
        supported_precisions=(Precision.BF16, Precision.FP32),
    )
    _BASE_1_7B = ModelDescriptor(
        id=QWEN_BASE_1_7B,
        name="Qwen3-TTS 12Hz 1.7B Base",
        description="Larger voice-cloning model; inference is not integrated yet.",
        supported_backends=(DeviceBackend.CPU, DeviceBackend.XPU),
        supported_precisions=(Precision.BF16, Precision.FP32),
    )
    _VOICE_DESIGN_1_7B = ModelDescriptor(
        id=QWEN_VOICE_DESIGN_1_7B,
        name="Qwen3-TTS 12Hz 1.7B VoiceDesign",
        description="Free-form voice design model; inference is not integrated yet.",
        supported_backends=(DeviceBackend.CPU, DeviceBackend.XPU),
        supported_precisions=(Precision.BF16, Precision.FP32),
    )
    _MODELS = (
        _CUSTOM_0_6B,
        _BASE_0_6B,
        _CUSTOM_1_7B,
        _BASE_1_7B,
        _VOICE_DESIGN_1_7B,
    )

    _VOICES = (
        VoiceDescriptor(
            id="vivian",
            name="Vivian",
            language="zh-CN",
            gender="female",
            category="Chinese preset",
            description="Bright, slightly edgy young voice.",
        ),
        VoiceDescriptor(
            id="serena",
            name="Serena",
            language="zh-CN",
            gender="female",
            category="Chinese preset",
            description="Warm, gentle young voice.",
        ),
        VoiceDescriptor(
            id="uncle_fu",
            name="Uncle Fu",
            language="zh-CN",
            gender="male",
            category="Chinese preset",
            description="Seasoned voice with a low, mellow timbre.",
        ),
        VoiceDescriptor(
            id="dylan",
            name="Dylan",
            language="zh-CN",
            accent="Beijing",
            gender="male",
            category="Chinese dialect preset",
            description="Youthful voice with a clear, natural timbre.",
        ),
        VoiceDescriptor(
            id="eric",
            name="Eric",
            language="zh-CN",
            accent="Sichuan",
            gender="male",
            category="Chinese dialect preset",
            description="Lively voice with a slightly husky brightness.",
        ),
        VoiceDescriptor(
            id="ryan",
            name="Ryan",
            language="en",
            gender="male",
            category="English preset",
            description="Dynamic voice with strong rhythmic drive.",
        ),
        VoiceDescriptor(
            id="aiden",
            name="Aiden",
            language="en-US",
            accent="American",
            gender="male",
            category="English preset",
            description="Sunny voice with a clear midrange.",
        ),
        VoiceDescriptor(
            id="ono_anna",
            name="Ono Anna",
            language="ja-JP",
            gender="female",
            category="Japanese preset",
            description="Playful voice with a light, nimble timbre.",
        ),
        VoiceDescriptor(
            id="sohee",
            name="Sohee",
            language="ko-KR",
            gender="female",
            category="Korean preset",
            description="Warm voice with rich emotion.",
        ),
    )

    _CUSTOM_INPUTS = (_VOICE_INPUT, _LANGUAGE_INPUT, _STYLE_INPUT)
    _BASE_INPUTS = (
        _LANGUAGE_INPUT,
        _REFERENCE_AUDIO_INPUT,
        _REFERENCE_TEXT_INPUT,
    )
    _VOICE_DESIGN_INPUTS = (_LANGUAGE_INPUT, _VOICE_DESIGN_INPUT)

    _DESCRIPTOR = EngineDescriptor(
        id="qwen",
        name="Qwen3-TTS",
        description=(
            "Qwen3-TTS model family with an Atlas-validated 0.6B CustomVoice runtime."
        ),
        capabilities=frozenset(
            {
                Capability.PRESET_VOICES,
                Capability.REFERENCE_AUDIO,
                Capability.REFERENCE_TEXT,
                Capability.VOICE_CLONING,
                Capability.VOICE_DESIGN,
                Capability.STYLE_CONTROL,
                Capability.LANGUAGE_SELECTION,
                Capability.MULTILINGUAL,
                Capability.STREAMING,
            }
        ),
        default_model_id=_CUSTOM_0_6B.id,
    )

    def __init__(
        self,
        runtime_config: QwenRuntimeConfig | None = None,
        session_factory: Callable[[QwenRuntimeConfig], QwenSession] | None = None,
        device_resolver: Callable[[DeviceSpec], str] | None = None,
    ) -> None:
        self._runtime_config = runtime_config or QwenRuntimeConfig.from_environment()
        self._session_factory = session_factory or QwenSession.load
        self._device_resolver = device_resolver or resolve_qwen_device
        self._sessions: dict[tuple[str, str, str], QwenSession] = {}
        self._session_lock = Lock()

    @property
    def descriptor(self) -> EngineDescriptor:
        return self._DESCRIPTOR

    def list_models(self) -> tuple[ModelDescriptor, ...]:
        return self._MODELS

    def list_voices(self, model_id: str) -> tuple[VoiceDescriptor, ...]:
        self.require_model(model_id)
        if model_id in {QWEN_CUSTOM_VOICE_0_6B, QWEN_CUSTOM_VOICE_1_7B}:
            return self._VOICES
        return ()

    def input_schema(self, model_id: str) -> tuple[InputFieldSpec, ...]:
        self.require_model(model_id)
        if model_id in {QWEN_CUSTOM_VOICE_0_6B, QWEN_CUSTOM_VOICE_1_7B}:
            return self._CUSTOM_INPUTS
        if model_id == QWEN_VOICE_DESIGN_1_7B:
            return self._VOICE_DESIGN_INPUTS
        return self._BASE_INPUTS

    def setting_schema(self, model_id: str) -> tuple[SettingSpec, ...]:
        self.require_model(model_id)
        if model_id in {QWEN_BASE_0_6B, QWEN_BASE_1_7B}:
            return (_SPEAKER_EMBEDDING_ONLY, *_SAMPLING_SETTINGS)
        return _SAMPLING_SETTINGS

    def create_session(
        self,
        model_id: str,
        device: str,
        precision: str,
    ) -> QwenSession:
        """Create and cache one session per model, device, and precision."""
        self.require_model(model_id)
        if model_id != QWEN_MODEL_ID:
            raise InferenceNotImplementedError(
                f"Qwen runtime is currently available only for {QWEN_MODEL_ID}"
            )
        key = (model_id, device, precision)
        if key not in self._sessions:
            with self._session_lock:
                if key not in self._sessions:
                    config = self._runtime_config.for_session(
                        model_id=model_id,
                        device=device,
                        precision=precision,
                    )
                    session = self._session_factory(config)
                    if not isinstance(session, QwenSession):
                        raise TypeError("Qwen session factory returned an invalid session")
                    self._sessions[key] = session
                    _LOGGER.info(
                        "Loaded engine=%s model=%s device=%s precision=%s",
                        self.descriptor.id,
                        model_id,
                        device,
                        precision,
                    )
                    self._notify_runtime_loaded(
                        LoadedRuntime(
                            engine_id=self.descriptor.id,
                            model_id=model_id,
                            device=device,
                            precision=precision,
                        )
                    )
        return self._sessions[key]

    def unload_runtime(self, model_id: str | None = None) -> tuple[object, ...]:
        if model_id is not None:
            self.require_model(model_id)
        with self._session_lock:
            keys = tuple(
                key
                for key in self._sessions
                if model_id is None or key[0] == model_id
            )
            sessions = tuple(self._sessions.pop(key) for key in keys)
        return sessions

    def runtime_status(self) -> tuple[LoadedRuntime, ...]:
        with self._session_lock:
            session_keys = tuple(self._sessions)
        return tuple(
            LoadedRuntime(
                engine_id=self.descriptor.id,
                model_id=model_id,
                device=device,
                precision=precision,
            )
            for model_id, device, precision in session_keys
        )

    def generate(self, request: SynthesisRequest) -> SynthesisResult:
        if request.engine_id != self.descriptor.id:
            raise ValueError(
                f"request engine {request.engine_id!r} does not match "
                f"{self.descriptor.id!r}"
            )
        self.require_model(request.model_id)
        if request.model_id != QWEN_MODEL_ID:
            raise InferenceNotImplementedError(
                f"Qwen runtime is currently available only for {QWEN_MODEL_ID}"
            )
        if request.device.backend not in {
            DeviceBackend.AUTO,
            DeviceBackend.CPU,
            DeviceBackend.XPU,
        }:
            raise ValueError("Qwen3-TTS supports CPU and XPU inference only")
        if request.device.precision not in {
            Precision.AUTO,
            Precision.FP32,
            Precision.BF16,
        }:
            raise ValueError("Qwen3-TTS supports FP32 and BF16 precision only")

        model_inputs = self.input_schema(request.model_id)
        model_settings = self.setting_schema(request.model_id)
        unknown_inputs = set(request.inputs) - {field.key for field in model_inputs}
        unknown_settings = set(request.settings) - {
            setting.key for setting in model_settings
        }
        if unknown_inputs:
            raise ValueError(f"unsupported Qwen inputs: {sorted(unknown_inputs)}")
        if unknown_settings:
            raise ValueError(f"unsupported Qwen settings: {sorted(unknown_settings)}")

        voice_id = request.inputs.get("voice", _VOICE_INPUT.default)
        voice_ids = {voice.id for voice in self._VOICES}
        if not isinstance(voice_id, str) or voice_id not in voice_ids:
            raise ValueError(f"unknown Qwen voice: {voice_id!r}")

        language = request.inputs.get("language", _LANGUAGE_INPUT.default)
        language_ids = {choice_id for _label, choice_id in _LANGUAGE_CHOICES}
        if not isinstance(language, str) or language not in language_ids:
            raise ValueError(f"unknown Qwen language: {language!r}")

        style_value = request.inputs.get("style_instruction", "")
        if style_value is None:
            style_instruction = ""
        elif isinstance(style_value, str):
            style_instruction = style_value.strip()
        else:
            raise ValueError("Qwen style instruction must be text")

        generation_settings = {
            setting.key: _validated_setting_value(
                setting,
                request.settings.get(setting.key, setting.default),
            )
            for setting in _SAMPLING_SETTINGS
        }
        device = self._device_resolver(request.device)
        precision = resolve_qwen_precision(request.device, device)
        return self.create_session(request.model_id, device, precision).generate(
            text=request.text,
            engine_id=self.descriptor.id,
            model_id=request.model_id,
            voice_id=voice_id,
            language=language,
            style_instruction=style_instruction,
            generation_settings=generation_settings,
        )


def resolve_qwen_device(device: DeviceSpec) -> str:
    """Resolve auto, CPU, or XPU through the shared hardware diagnostics."""
    if device.backend not in {
        DeviceBackend.AUTO,
        DeviceBackend.CPU,
        DeviceBackend.XPU,
    }:
        raise ValueError(f"unsupported Qwen device backend: {device.backend.value}")
    availability = probe_device_availability()
    return select_device(
        availability,
        requested_device=device.backend.value,
        xpu_index=device.index or 0,
    )


def resolve_qwen_precision(device: DeviceSpec, resolved_device: str) -> str:
    if device.precision is Precision.AUTO:
        return "bf16" if resolved_device.startswith("xpu") else "fp32"
    if device.precision in {Precision.FP32, Precision.BF16}:
        return device.precision.value
    raise ValueError(f"unsupported Qwen precision: {device.precision.value}")


def _validated_setting_value(setting: SettingSpec, value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Qwen {setting.label} must be numeric")
    if setting.key == "top_k" and not isinstance(value, int):
        raise ValueError("Qwen Top K must be an integer")
    if setting.minimum is not None and value < setting.minimum:
        raise ValueError(f"Qwen {setting.label} must be at least {setting.minimum}")
    if setting.maximum is not None and value > setting.maximum:
        raise ValueError(f"Qwen {setting.label} must be at most {setting.maximum}")
    return value
