"""Lazy Qwen3-TTS model loading, generation, and output persistence."""

from array import array
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4
import wave

from tts_workbench.domain.results import SynthesisResult
from tts_workbench.engines.base import (
    EngineDependencyError,
    EngineOutputError,
    EngineRuntimeError,
)


QWEN_CUSTOM_VOICE_0_6B = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
QWEN_BASE_0_6B = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN_CUSTOM_VOICE_1_7B = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_BASE_1_7B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN_VOICE_DESIGN_1_7B = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN_MODEL_IDS = (
    QWEN_CUSTOM_VOICE_0_6B,
    QWEN_BASE_0_6B,
    QWEN_CUSTOM_VOICE_1_7B,
    QWEN_BASE_1_7B,
    QWEN_VOICE_DESIGN_1_7B,
)
QWEN_MODEL_ID = QWEN_CUSTOM_VOICE_0_6B
QWEN_REPOSITORY = QWEN_CUSTOM_VOICE_0_6B
QWEN_REQUIRED_DEPENDENCIES = {
    "qwen-tts": "0.1.1",
    "transformers": "4.57.3",
    "accelerate": "1.12.0",
}


@dataclass(frozen=True, slots=True)
class QwenDependencyStatus:
    """Installed Qwen package versions without importing the runtime itself."""

    available: bool
    versions: tuple[tuple[str, str | None], ...]
    problems: tuple[str, ...] = ()

    def version(self, distribution: str) -> str | None:
        return dict(self.versions).get(distribution)


def detect_qwen_dependencies() -> QwenDependencyStatus:
    """Report required package presence and pins without importing Qwen or Torch."""
    versions: list[tuple[str, str | None]] = []
    problems: list[str] = []
    for distribution, expected_version in QWEN_REQUIRED_DEPENDENCIES.items():
        try:
            installed_version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            installed_version = None
            problems.append(f"{distribution} is not installed")
        else:
            if installed_version != expected_version:
                problems.append(
                    f"{distribution} must be {expected_version}, found {installed_version}"
                )
        versions.append((distribution, installed_version))

    for distribution in ("torch", "torchaudio"):
        try:
            installed_version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            installed_version = None
            problems.append(f"{distribution} is not installed")
        versions.append((distribution, installed_version))
    return QwenDependencyStatus(
        available=not problems,
        versions=tuple(versions),
        problems=tuple(problems),
    )


def qwen_runtime_available() -> bool:
    """Return whether the required pinned packages are installed."""
    return detect_qwen_dependencies().available


@dataclass(frozen=True, slots=True)
class QwenRuntimeConfig:
    """Filesystem, checkpoint, device, and precision for one Qwen session."""

    model_dir: Path
    output_dir: Path
    model_id: str = QWEN_MODEL_ID
    repository_id: str = QWEN_REPOSITORY
    device: str = "cpu"
    precision: str = "fp32"
    attention_implementation: str = "sdpa"

    def __post_init__(self) -> None:
        if self.model_id != QWEN_MODEL_ID:
            raise ValueError(f"unsupported Qwen runtime model: {self.model_id!r}")
        if self.device != "cpu" and not self.device.startswith("xpu"):
            raise ValueError("Qwen runtime device must be CPU or XPU")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("Qwen runtime precision must be fp32 or bf16")
        if not self.attention_implementation:
            raise ValueError("Qwen attention implementation cannot be empty")
        object.__setattr__(self, "model_dir", self.model_dir.resolve())
        object.__setattr__(self, "output_dir", self.output_dir.resolve())

    @classmethod
    def from_root(cls, root: Path) -> "QwenRuntimeConfig":
        root = root.resolve()
        return cls(
            model_dir=root / "models" / "qwen" / QWEN_MODEL_ID.rsplit("/", 1)[-1],
            output_dir=root / "output",
        )

    @classmethod
    def from_environment(cls) -> "QwenRuntimeConfig":
        configured_root = os.environ.get("TTS_WORKBENCH_ROOT")
        return cls.from_root(Path(configured_root) if configured_root else Path.cwd())

    def for_session(
        self,
        *,
        model_id: str,
        device: str,
        precision: str,
    ) -> "QwenRuntimeConfig":
        return replace(
            self,
            model_id=model_id,
            device=device,
            precision=precision,
        )


@dataclass(frozen=True, slots=True)
class QwenGeneratedAudio:
    samples: Any
    sample_rate: int


class QwenBackend(Protocol):
    device: str
    precision: str

    def generate(
        self,
        *,
        text: str,
        voice_id: str,
        language: str,
        style_instruction: str,
        generation_settings: Mapping[str, int | float],
    ) -> QwenGeneratedAudio: ...


class QwenSession:
    """A loaded Qwen model with serialized generation and normalized outputs."""

    def __init__(self, config: QwenRuntimeConfig, backend: QwenBackend) -> None:
        if backend is None or not callable(getattr(backend, "generate", None)):
            raise ValueError("Qwen backend must provide generate()")
        if backend.device != config.device:
            raise ValueError("Qwen backend device does not match its runtime configuration")
        if backend.precision != config.precision:
            raise ValueError("Qwen backend precision does not match its runtime configuration")
        self.config = config
        self._backend = backend
        self._generation_lock = Lock()

    @classmethod
    def load(cls, config: QwenRuntimeConfig) -> "QwenSession":
        """Lazily import Qwen dependencies and load the configured checkpoint."""
        config.model_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        backend = OfficialQwenBackend.load(config)
        return cls(config, backend)

    def generate(
        self,
        *,
        text: str,
        engine_id: str,
        model_id: str,
        voice_id: str,
        language: str,
        style_instruction: str,
        generation_settings: Mapping[str, int | float],
    ) -> SynthesisResult:
        """Generate one utterance and persist its WAV and normalized metadata."""
        started = perf_counter()
        with self._generation_lock:
            try:
                generated = self._backend.generate(
                    text=text,
                    voice_id=voice_id,
                    language=language,
                    style_instruction=style_instruction,
                    generation_settings=generation_settings,
                )
            except EngineRuntimeError:
                raise
            except Exception as error:
                raise EngineRuntimeError(f"Qwen3-TTS generation failed: {error}") from error

            samples = list(_flatten_samples(generated.samples))
            if not samples:
                raise EngineOutputError("Qwen3-TTS produced no audio samples")
            if generated.sample_rate <= 0:
                raise EngineOutputError("Qwen3-TTS produced an invalid sample rate")

            job_id = uuid4().hex
            job_dir = self.config.output_dir / datetime.now(UTC).date().isoformat() / job_id
            job_dir.mkdir(parents=True, exist_ok=False)
            audio_path = job_dir / "audio.wav"
            _write_pcm16_wav(audio_path, samples, generated.sample_rate)
            _validate_wav_output(
                audio_path,
                expected_sample_rate=generated.sample_rate,
                expected_frames=len(samples),
            )

            duration_seconds = len(samples) / generated.sample_rate
            generation_seconds = perf_counter() - started
            metadata_path = job_dir / "metadata.json"
            metadata = {
                "engine_id": engine_id,
                "model_id": model_id,
                "repository_id": self.config.repository_id,
                "voice_id": voice_id,
                "language": language,
                "style_instruction": style_instruction,
                "style_instruction_supported": False,
                "generation_settings": dict(generation_settings),
                "sample_rate": generated.sample_rate,
                "duration_seconds": duration_seconds,
                "generation_seconds": generation_seconds,
                "device": self.config.device,
                "precision": self.config.precision,
            }
            _write_json(metadata_path, metadata)

        return SynthesisResult(
            engine_id=engine_id,
            model_id=model_id,
            voice_id=voice_id,
            audio_path=audio_path,
            sample_rate=generated.sample_rate,
            duration_seconds=duration_seconds,
            generation_seconds=generation_seconds,
            metadata={
                "repository_id": self.config.repository_id,
                "language": language,
                "style_instruction": style_instruction,
                "style_instruction_supported": False,
                "generation_settings": dict(generation_settings),
                "device": self.config.device,
                "precision": self.config.precision,
                "metadata_path": str(metadata_path),
            },
        )


class OfficialQwenBackend:
    """Thin lazy wrapper around the published qwen-tts Python API."""

    def __init__(self, model: Any, device: str, precision: str) -> None:
        self._model = model
        self.device = device
        self.precision = precision

    @classmethod
    def load(cls, config: QwenRuntimeConfig) -> "OfficialQwenBackend":
        torch, model_class, snapshot_download = _load_qwen_dependencies()
        if config.device.startswith("xpu"):
            xpu = getattr(torch, "xpu", None)
            if xpu is None or not xpu.is_available():
                raise EngineRuntimeError("Qwen3-TTS XPU was requested but is unavailable")

        model_path = _resolve_model_path(config, snapshot_download)
        dtype = torch.bfloat16 if config.precision == "bf16" else torch.float32
        try:
            model = model_class.from_pretrained(
                str(model_path),
                device_map=config.device,
                dtype=dtype,
                attn_implementation=config.attention_implementation,
            )
            underlying_model = getattr(model, "model", None)
            if underlying_model is not None and callable(
                getattr(underlying_model, "eval", None)
            ):
                underlying_model.eval()
        except Exception as error:
            raise EngineRuntimeError(
                f"failed to load Qwen3-TTS on {config.device} "
                f"with {config.precision}: {error}"
            ) from error
        return cls(model, config.device, config.precision)

    def generate(
        self,
        *,
        text: str,
        voice_id: str,
        language: str,
        style_instruction: str,
        generation_settings: Mapping[str, int | float],
    ) -> QwenGeneratedAudio:
        try:
            waveforms, sample_rate = self._model.generate_custom_voice(
                text=text,
                speaker=voice_id,
                language=language,
                instruct=style_instruction or None,
                non_streaming_mode=True,
                do_sample=True,
                **dict(generation_settings),
            )
        except Exception as error:
            raise EngineRuntimeError(f"Qwen3-TTS generation failed: {error}") from error
        if not waveforms:
            raise EngineOutputError("Qwen3-TTS returned no waveforms")
        return QwenGeneratedAudio(waveforms[0], int(sample_rate))


def _import_qwen_dependencies() -> tuple[Any, Any, Callable[..., str]]:
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    return torch, Qwen3TTSModel, snapshot_download


def _load_qwen_dependencies() -> tuple[Any, Any, Callable[..., str]]:
    try:
        return _import_qwen_dependencies()
    except Exception as error:
        raise EngineDependencyError(
            "Qwen3-TTS dependencies are unavailable. Install the project with "
            "the 'qwen' extra and a matching Torch/Torchaudio build."
        ) from error


def _resolve_model_path(
    config: QwenRuntimeConfig,
    snapshot_download: Callable[..., str],
) -> Path:
    required_files = (
        config.model_dir / "config.json",
        config.model_dir / "model.safetensors",
        config.model_dir / "speech_tokenizer" / "model.safetensors",
    )
    if all(path.is_file() and path.stat().st_size > 0 for path in required_files):
        return config.model_dir
    try:
        downloaded = Path(
            snapshot_download(
                repo_id=config.repository_id,
                local_dir=str(config.model_dir),
            )
        )
    except Exception as error:
        raise EngineRuntimeError(
            f"failed to obtain Qwen3-TTS model {config.repository_id!r}: {error}"
        ) from error
    if not all(path.is_file() and path.stat().st_size > 0 for path in required_files):
        raise EngineRuntimeError(
            f"downloaded Qwen3-TTS model is incomplete: {downloaded}"
        )
    return config.model_dir


def _flatten_samples(value: Any) -> Iterator[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_samples(item)
        return
    try:
        sample = float(value)
    except (TypeError, ValueError) as error:
        raise EngineOutputError(
            f"Qwen3-TTS returned a non-numeric audio sample: {value!r}"
        ) from error
    if not math.isfinite(sample):
        raise EngineOutputError("Qwen3-TTS returned a non-finite audio sample")
    yield sample


def _write_pcm16_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    if not samples:
        raise EngineOutputError("cannot write an empty Qwen3-TTS audio output")
    pcm = array(
        "h",
        (round(max(-1.0, min(1.0, sample)) * 32767) for sample in samples),
    )
    if sys.byteorder != "little":
        pcm.byteswap()
    temporary_path = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        with wave.open(str(temporary_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        temporary_path.replace(path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise EngineOutputError(f"failed to write Qwen3-TTS WAV output: {error}") from error


def _validate_wav_output(
    path: Path,
    *,
    expected_sample_rate: int,
    expected_frames: int,
) -> None:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise EngineOutputError(f"Qwen3-TTS output is missing or empty: {path}")
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise EngineOutputError("Qwen3-TTS output must be mono 16-bit PCM")
            if wav_file.getframerate() != expected_sample_rate:
                raise EngineOutputError("Qwen3-TTS output sample rate is incorrect")
            if wav_file.getnframes() != expected_frames:
                raise EngineOutputError("Qwen3-TTS output frame count is incorrect")
    except EngineOutputError:
        raise
    except (OSError, EOFError, wave.Error) as error:
        raise EngineOutputError(
            f"Qwen3-TTS output is not a valid WAV file: {error}"
        ) from error


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write("\n")
        temporary_path.replace(path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise EngineOutputError(f"failed to write Qwen3-TTS metadata: {error}") from error
