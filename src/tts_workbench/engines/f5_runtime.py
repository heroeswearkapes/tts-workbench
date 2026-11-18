"""F5-TTS CPU/XPU loading with normalized SoundFile reference decoding."""

from array import array
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from threading import Lock
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4
import wave

from tts_workbench.audio import (
    AudioNormalizationError,
    AudioNormalizerDependencyError,
    normalize_audio_with_ffmpeg,
)
from tts_workbench.domain.results import SynthesisResult
from tts_workbench.engines.base import (
    EngineDependencyError,
    EngineOutputError,
    EngineRuntimeError,
)


F5_UPSTREAM_MODEL = "F5TTS_v1_Base"
F5_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
F5_SAMPLE_RATE = 24_000
WHISPER_SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class F5RuntimeConfig:
    """Filesystem, model, and device configuration for one F5 session."""

    model_dir: Path
    output_dir: Path
    device: str = "cpu"
    upstream_model: str = F5_UPSTREAM_MODEL
    whisper_model: str = F5_WHISPER_MODEL
    maximum_reference_seconds: float = 12.0

    def __post_init__(self) -> None:
        if self.device != "cpu" and not self.device.startswith("xpu"):
            raise ValueError("F5 runtime device must be CPU or XPU")
        if self.maximum_reference_seconds <= 0:
            raise ValueError("maximum reference duration must be positive")
        object.__setattr__(self, "model_dir", self.model_dir.resolve())
        object.__setattr__(self, "output_dir", self.output_dir.resolve())

    @classmethod
    def from_root(cls, root: Path) -> "F5RuntimeConfig":
        root = root.resolve()
        return cls(
            model_dir=root / "models" / "f5" / "f5-tts-v1-base",
            output_dir=root / "output",
        )

    @classmethod
    def from_environment(cls) -> "F5RuntimeConfig":
        configured_root = os.environ.get("TTS_WORKBENCH_ROOT")
        return cls.from_root(Path(configured_root) if configured_root else Path.cwd())

    def for_device(self, device: str) -> "F5RuntimeConfig":
        return replace(self, device=device)


@dataclass(frozen=True, slots=True)
class DecodedReferenceAudio:
    """Reference audio decoded by soundfile before any model library sees it."""

    samples: Any
    sample_rate: int
    frames: int
    channels: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


@dataclass(frozen=True, slots=True)
class F5GeneratedAudio:
    samples: Any
    sample_rate: int
    seed: int | None = None


class F5Backend(Protocol):
    device: str

    def generate(
        self,
        reference_audio: DecodedReferenceAudio,
        reference_text: str,
        generation_text: str,
        speed: float,
    ) -> F5GeneratedAudio: ...


class ReferenceTranscriber(Protocol):
    device: str

    def transcribe(self, reference_audio: Path, maximum_seconds: float) -> str: ...


class F5Session:
    """A loaded F5 model plus optional CPU-only reference transcription."""

    def __init__(
        self,
        config: F5RuntimeConfig,
        backend: F5Backend,
        *,
        audio_decoder: Callable[[Path, float], DecodedReferenceAudio] | None = None,
        audio_normalizer: Callable[[Path, int], AbstractContextManager[Path]] | None = None,
        transcriber: ReferenceTranscriber | None = None,
    ) -> None:
        if backend is None or not callable(getattr(backend, "generate", None)):
            raise ValueError("F5 backend must provide generate()")
        if backend.device != config.device:
            raise ValueError("F5 backend device does not match its runtime configuration")
        self.config = config
        self._backend = backend
        self._audio_decoder = audio_decoder or decode_reference_audio_with_soundfile
        self._audio_normalizer = audio_normalizer or normalize_audio_with_ffmpeg
        self._transcriber = transcriber
        self._generation_lock = Lock()

    @classmethod
    def load(cls, config: F5RuntimeConfig) -> "F5Session":
        """Load F5 on the requested device without initializing Whisper."""
        config.model_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        backend = OfficialF5Backend.load(config)
        return cls(config=config, backend=backend)

    def generate(
        self,
        *,
        text: str,
        engine_id: str,
        model_id: str,
        reference_audio: Path,
        reference_text: str,
        speed: float,
    ) -> SynthesisResult:
        """Decode, optionally transcribe, infer, persist, and normalize a result."""
        started = perf_counter()
        with self._generation_lock:
            try:
                with self._audio_normalizer(
                    reference_audio,
                    F5_SAMPLE_RATE,
                ) as normalized_audio:
                    decoded = self._audio_decoder(
                        normalized_audio,
                        self.config.maximum_reference_seconds,
                    )
                    if decoded.sample_rate != F5_SAMPLE_RATE:
                        raise EngineRuntimeError(
                            f"F5 reference audio must decode at {F5_SAMPLE_RATE} Hz"
                        )
                    transcript = reference_text.strip()
                    transcript_source = "provided"
                    if not transcript:
                        transcriber = self._get_transcriber()
                        if transcriber.device != "cpu":
                            raise EngineRuntimeError(
                                "F5 reference transcription must run on CPU"
                            )
                        transcript = transcriber.transcribe(
                            normalized_audio,
                            self.config.maximum_reference_seconds,
                        ).strip()
                        transcript_source = "transcribed"
                        if not transcript:
                            raise EngineRuntimeError(
                                "F5 reference transcription produced no text"
                            )

                    transcript = _normalize_reference_text(transcript)
                    try:
                        generated = self._backend.generate(
                            reference_audio=decoded,
                            reference_text=transcript,
                            generation_text=text,
                            speed=speed,
                        )
                    except EngineRuntimeError:
                        raise
                    except Exception as error:
                        raise EngineRuntimeError(f"F5 generation failed: {error}") from error
            except AudioNormalizerDependencyError as error:
                raise EngineDependencyError(str(error)) from error
            except AudioNormalizationError as error:
                raise EngineRuntimeError(
                    f"failed to normalize F5 reference audio: {error}"
                ) from error

            samples = list(_flatten_samples(generated.samples))
            if not samples:
                raise EngineOutputError("F5 produced no audio samples")
            if generated.sample_rate <= 0:
                raise EngineOutputError("F5 produced an invalid sample rate")

            job_id = uuid4().hex
            job_dir = self.config.output_dir / datetime.now().date().isoformat() / job_id
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
                "voice_id": None,
                "speed": speed,
                "sample_rate": generated.sample_rate,
                "duration_seconds": duration_seconds,
                "generation_seconds": generation_seconds,
                "device": self.config.device,
                "reference_audio_name": reference_audio.name,
                "reference_audio_sha256": _sha256(reference_audio),
                "reference_duration_seconds": decoded.duration_seconds,
                "reference_text": transcript,
                "reference_text_source": transcript_source,
                "seed": generated.seed,
            }
            _write_json(metadata_path, metadata)

        return SynthesisResult(
            engine_id=engine_id,
            model_id=model_id,
            voice_id=None,
            audio_path=audio_path,
            sample_rate=generated.sample_rate,
            duration_seconds=duration_seconds,
            generation_seconds=generation_seconds,
            metadata={
                "speed": speed,
                "device": self.config.device,
                "reference_audio_name": reference_audio.name,
                "reference_text_source": transcript_source,
                "seed": generated.seed,
                "metadata_path": str(metadata_path),
            },
        )

    def _get_transcriber(self) -> ReferenceTranscriber:
        if self._transcriber is None:
            self._transcriber = CPUWhisperTranscriber(self.config.whisper_model)
        return self._transcriber


class OfficialF5Backend:
    """Thin wrapper over F5's loaded model and decoded-audio inference path."""

    def __init__(self, api: Any, infer_batch_process: Callable[..., Any], device: str):
        self._api = api
        self._infer_batch_process = infer_batch_process
        self.device = device

    @classmethod
    def load(cls, config: F5RuntimeConfig) -> "OfficialF5Backend":
        try:
            import torch
            from f5_tts.api import F5TTS
            from f5_tts.infer.utils_infer import infer_batch_process
        except ModuleNotFoundError as error:
            raise EngineDependencyError(
                "F5-TTS is not installed. Install the project with the 'f5' extra."
            ) from error

        if config.device.startswith("xpu"):
            xpu = getattr(torch, "xpu", None)
            if xpu is None or not xpu.is_available():
                raise EngineRuntimeError("F5-TTS XPU was requested but is unavailable")

        try:
            api = F5TTS(
                model=config.upstream_model,
                device=config.device,
                hf_cache_dir=str(config.model_dir),
            )
        except Exception as error:
            raise EngineRuntimeError(
                f"failed to load F5-TTS on {config.device}: {error}"
            ) from error
        return cls(api=api, infer_batch_process=infer_batch_process, device=config.device)

    def generate(
        self,
        reference_audio: DecodedReferenceAudio,
        reference_text: str,
        generation_text: str,
        speed: float,
    ) -> F5GeneratedAudio:
        try:
            import torch
            from f5_tts.infer.utils_infer import chunk_text
            from f5_tts.model.utils import seed_everything
        except ModuleNotFoundError as error:
            raise EngineDependencyError("F5-TTS runtime dependencies are missing") from error

        audio_array = reference_audio.samples
        tensor = torch.from_numpy(audio_array.T.copy())
        available_seconds = max(1.0, 22.0 - reference_audio.duration_seconds)
        maximum_characters = max(
            1,
            int(
                len(reference_text.encode("utf-8"))
                / reference_audio.duration_seconds
                * available_seconds
                * speed
            ),
        )
        text_batches = chunk_text(generation_text, max_chars=maximum_characters)
        if not text_batches:
            raise EngineOutputError("F5 could not create a generation text batch")

        seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        result = next(
            self._infer_batch_process(
                (tensor, reference_audio.sample_rate),
                reference_text,
                text_batches,
                self._api.ema_model,
                self._api.vocoder,
                self._api.mel_spec_type,
                speed=speed,
                device=self.device,
            )
        )
        waveform, sample_rate, _spectrogram = result
        return F5GeneratedAudio(waveform, int(sample_rate), seed)


class CPUWhisperTranscriber:
    """Lazy explicit Whisper inference that always runs on CPU."""

    device = "cpu"

    def __init__(
        self,
        model_id: str,
        *,
        component_loader: Callable[[str], tuple[Any, Any]] | None = None,
        runtime_loader: Callable[[], tuple[Any, Any]] | None = None,
        audio_normalizer: Callable[[Path, int], AbstractContextManager[Path]] | None = None,
        audio_decoder: Callable[[Path, float], DecodedReferenceAudio] | None = None,
    ) -> None:
        self._model_id = model_id
        self._component_loader = component_loader or _load_whisper_components
        self._runtime_loader = runtime_loader or _load_transcription_runtime
        self._audio_normalizer = audio_normalizer or normalize_audio_with_ffmpeg
        self._audio_decoder = audio_decoder or decode_reference_audio_with_soundfile
        self._processor: Any = None
        self._model: Any = None

    def transcribe(self, reference_audio: Path, maximum_seconds: float) -> str:
        """Normalize to Whisper audio, extract features, and generate text on CPU."""
        try:
            np, torch = self._runtime_loader()
        except ModuleNotFoundError as error:
            raise EngineDependencyError(
                "Transformers, NumPy, and Torch are required for F5 transcription"
            ) from error
        processor, model = self._get_components(torch)

        try:
            with self._audio_normalizer(
                reference_audio,
                WHISPER_SAMPLE_RATE,
            ) as whisper_audio_path:
                decoded = self._audio_decoder(whisper_audio_path, maximum_seconds)
                if decoded.sample_rate != WHISPER_SAMPLE_RATE:
                    raise EngineRuntimeError(
                        f"Whisper reference audio must decode at {WHISPER_SAMPLE_RATE} Hz"
                    )
                mono = np.mean(decoded.samples, axis=1, dtype=np.float32)
                processed = processor(
                    mono,
                    sampling_rate=WHISPER_SAMPLE_RATE,
                    return_tensors="pt",
                )
                input_features = processed.input_features.to(
                    device=self.device,
                    dtype=torch.float32,
                )
                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_features=input_features,
                    )
                transcriptions = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
        except (
            AudioNormalizationError,
            EngineDependencyError,
            EngineRuntimeError,
        ):
            raise
        except Exception as error:
            raise EngineRuntimeError(f"Whisper CPU transcription failed: {error}") from error
        return str(transcriptions[0]).strip() if transcriptions else ""

    def _get_components(self, torch: Any) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            try:
                processor, model = self._component_loader(self._model_id)
                model = model.to(device=self.device, dtype=torch.float32)
                model.eval()
            except ModuleNotFoundError as error:
                raise EngineDependencyError(
                    "Transformers is required for F5 reference transcription"
                ) from error
            except Exception as error:
                raise EngineRuntimeError(
                    f"failed to load Whisper on CPU: {error}"
                ) from error
            self._processor = processor
            self._model = model
        return self._processor, self._model


def _load_whisper_components(model_id: str) -> tuple[Any, Any]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    return processor, model


def _load_transcription_runtime() -> tuple[Any, Any]:
    import numpy as np
    import torch

    return np, torch


def decode_reference_audio_with_soundfile(
    path: Path,
    maximum_seconds: float,
) -> DecodedReferenceAudio:
    """Decode with soundfile only; no torchaudio or TorchCodec path is used."""
    try:
        import numpy as np
        import soundfile as sf
    except ModuleNotFoundError as error:
        raise EngineDependencyError(
            "soundfile and NumPy are required to decode F5 reference audio"
        ) from error
    try:
        samples, sample_rate = sf.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )
    except Exception as error:
        raise EngineRuntimeError(f"failed to decode F5 reference audio: {error}") from error
    if sample_rate <= 0 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise EngineRuntimeError("F5 reference audio is empty or invalid")
    if not np.isfinite(samples).all():
        raise EngineRuntimeError("F5 reference audio contains non-finite samples")
    maximum_frames = max(1, int(sample_rate * maximum_seconds))
    samples = samples[:maximum_frames]
    return DecodedReferenceAudio(
        samples=samples,
        sample_rate=int(sample_rate),
        frames=int(samples.shape[0]),
        channels=int(samples.shape[1]),
    )


def _normalize_reference_text(text: str) -> str:
    text = text.strip()
    if text.endswith("。"):
        return text
    if text.endswith("."):
        return f"{text} "
    return f"{text}. "


def _flatten_samples(samples: Any) -> Iterator[float]:
    value = samples
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
        raise EngineOutputError(f"F5 returned a non-numeric audio sample: {value!r}") from error
    if not math.isfinite(sample):
        raise EngineOutputError("F5 returned a non-finite audio sample")
    yield sample


def _write_pcm16_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    pcm = array(
        "h",
        (round(max(-1.0, min(1.0, sample)) * 32767) for sample in samples),
    )
    if sys.byteorder != "little":
        pcm.byteswap()
    temporary_path = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        with wave.open(str(temporary_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm.tobytes())
        temporary_path.replace(path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise EngineOutputError(f"failed to write F5 WAV output: {error}") from error


def _validate_wav_output(
    path: Path,
    *,
    expected_sample_rate: int,
    expected_frames: int,
) -> None:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise EngineOutputError(f"F5 output file is missing or empty: {path}")
        with wave.open(str(path), "rb") as output:
            if output.getnchannels() != 1 or output.getsampwidth() != 2:
                raise EngineOutputError("F5 output must be mono 16-bit PCM")
            if output.getframerate() != expected_sample_rate:
                raise EngineOutputError("F5 output sample rate does not match the result")
            if output.getnframes() != expected_frames:
                raise EngineOutputError("F5 output frame count does not match generated audio")
    except EngineOutputError:
        raise
    except (OSError, EOFError, wave.Error) as error:
        raise EngineOutputError(f"F5 output is not a valid WAV file: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write("\n")
        temporary_path.replace(path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise EngineOutputError(f"failed to write F5 metadata: {error}") from error
