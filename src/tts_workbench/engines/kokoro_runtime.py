"""CPU-only Kokoro model loading, generation, and WAV persistence."""

from array import array
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4
import wave

from tts_workbench.domain.results import SynthesisResult
from tts_workbench.engines.base import (
    EngineDependencyError,
    EngineOutputError,
    EngineRuntimeError,
)


KOKORO_REPOSITORY = "hexgrad/Kokoro-82M"
KOKORO_MODEL_FILENAME = "kokoro-v1_0.pth"
KOKORO_CONFIG_FILENAME = "config.json"
KOKORO_SAMPLE_RATE = 24_000


@dataclass(frozen=True, slots=True)
class KokoroRuntimeConfig:
    """Filesystem and upstream model settings for one Kokoro runtime."""

    model_dir: Path
    voice_dir: Path
    output_dir: Path
    repository_id: str = KOKORO_REPOSITORY
    sample_rate: int = KOKORO_SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("Kokoro sample rate must be positive")
        object.__setattr__(self, "model_dir", self.model_dir.resolve())
        object.__setattr__(self, "voice_dir", self.voice_dir.resolve())
        object.__setattr__(self, "output_dir", self.output_dir.resolve())

    @classmethod
    def from_root(cls, root: Path) -> "KokoroRuntimeConfig":
        root = root.resolve()
        return cls(
            model_dir=root / "models" / "kokoro" / "kokoro-82m",
            voice_dir=root / "voices" / "kokoro",
            output_dir=root / "output",
        )

    @classmethod
    def from_environment(cls) -> "KokoroRuntimeConfig":
        configured_root = os.environ.get("TTS_WORKBENCH_ROOT")
        return cls.from_root(Path(configured_root) if configured_root else Path.cwd())


class KokoroSession:
    """A loaded Kokoro pipeline with serialized CPU generation."""

    def __init__(
        self,
        config: KokoroRuntimeConfig,
        pipeline: Any,
        *,
        voice_resolver: Callable[[str], Path] | None = None,
        pipeline_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if pipeline is None or not callable(pipeline):
            raise ValueError("Kokoro pipeline must be callable")
        self.config = config
        self._pipeline = pipeline
        self._pipelines = {"a": pipeline}
        self._pipeline_factory = pipeline_factory
        self._voice_resolver = voice_resolver or self._resolve_voice
        self._generation_lock = Lock()

    @classmethod
    def load(cls, config: KokoroRuntimeConfig) -> "KokoroSession":
        """Download missing CPU assets into project storage and load KPipeline."""
        try:
            from huggingface_hub import hf_hub_download
            from kokoro import KModel, KPipeline
        except ModuleNotFoundError as error:
            raise EngineDependencyError(
                "Kokoro is not installed. Install the project with the 'kokoro' extra."
            ) from error

        config.model_dir.mkdir(parents=True, exist_ok=True)
        config.voice_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            config_path = _download_model_file(
                config,
                KOKORO_CONFIG_FILENAME,
                hf_hub_download,
            )
            model_path = _download_model_file(
                config,
                KOKORO_MODEL_FILENAME,
                hf_hub_download,
            )
            model = KModel(
                repo_id=config.repository_id,
                config=str(config_path),
                model=str(model_path),
            ).to("cpu").eval()
            def create_pipeline(language_code: str) -> Any:
                return KPipeline(
                    lang_code=language_code,
                    repo_id=config.repository_id,
                    model=model,
                    device="cpu",
                )

            pipeline = create_pipeline("a")
        except EngineRuntimeError:
            raise
        except Exception as error:
            raise EngineRuntimeError(f"failed to load Kokoro on CPU: {error}") from error
        return cls(
            config=config,
            pipeline=pipeline,
            pipeline_factory=create_pipeline,
        )

    def generate(
        self,
        *,
        text: str,
        engine_id: str,
        model_id: str,
        voice_id: str,
        speed: float,
    ) -> SynthesisResult:
        """Generate all Kokoro chunks, persist one PCM WAV, and return metadata."""
        started = perf_counter()
        with self._generation_lock:
            voice_path = self._voice_resolver(voice_id)
            pipeline = self._pipeline_for_voice(voice_id)
            try:
                outputs = pipeline(
                    text,
                    voice=str(voice_path),
                    speed=speed,
                    split_pattern=r"\n+",
                )
                samples, chunk_count = _collect_audio(outputs)
            except EngineRuntimeError:
                raise
            except Exception as error:
                raise EngineRuntimeError(f"Kokoro generation failed: {error}") from error

            job_id = uuid4().hex
            job_dir = self.config.output_dir / datetime.now().date().isoformat() / job_id
            job_dir.mkdir(parents=True, exist_ok=False)
            audio_path = job_dir / "audio.wav"
            write_pcm16_wav(audio_path, samples, self.config.sample_rate)
            validate_wav_output(
                audio_path,
                expected_sample_rate=self.config.sample_rate,
                expected_frames=len(samples),
            )

            duration_seconds = len(samples) / self.config.sample_rate
            generation_seconds = perf_counter() - started
            metadata_path = job_dir / "metadata.json"
            metadata = {
                "engine_id": engine_id,
                "model_id": model_id,
                "voice_id": voice_id,
                "speed": speed,
                "sample_rate": self.config.sample_rate,
                "duration_seconds": duration_seconds,
                "generation_seconds": generation_seconds,
                "chunk_count": chunk_count,
            }
            _write_json(metadata_path, metadata)

        return SynthesisResult(
            engine_id=engine_id,
            model_id=model_id,
            voice_id=voice_id,
            audio_path=audio_path,
            sample_rate=self.config.sample_rate,
            duration_seconds=duration_seconds,
            generation_seconds=generation_seconds,
            metadata={
                "speed": speed,
                "chunk_count": chunk_count,
                "metadata_path": str(metadata_path),
            },
        )

    def _pipeline_for_voice(self, voice_id: str) -> Any:
        language_code = _kokoro_language_code(voice_id)
        existing = self._pipelines.get(language_code)
        if existing is not None:
            return existing
        if self._pipeline_factory is None:
            # Injected test/custom pipelines can handle all declared voices unless
            # they opt into language-specific construction.
            return self._pipeline
        try:
            pipeline = self._pipeline_factory(language_code)
        except Exception as error:
            raise EngineRuntimeError(
                f"failed to initialize Kokoro language pipeline {language_code!r}: "
                f"{error}"
            ) from error
        if pipeline is None or not callable(pipeline):
            raise EngineRuntimeError(
                "Kokoro language pipeline factory returned an invalid pipeline"
            )
        self._pipelines[language_code] = pipeline
        return pipeline

    def _resolve_voice(self, voice_id: str) -> Path:
        voice_path = self.config.voice_dir / f"{voice_id}.pt"
        if voice_path.is_file() and voice_path.stat().st_size > 0:
            return voice_path
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as error:
            raise EngineDependencyError(
                "huggingface_hub is required to download Kokoro voice assets"
            ) from error

        self.config.voice_dir.mkdir(parents=True, exist_ok=True)
        try:
            cached_path = Path(
                hf_hub_download(
                    repo_id=self.config.repository_id,
                    filename=f"voices/{voice_id}.pt",
                )
            )
            temporary_path = voice_path.with_suffix(f".{uuid4().hex}.tmp")
            shutil.copyfile(cached_path, temporary_path)
            temporary_path.replace(voice_path)
        except Exception as error:
            raise EngineRuntimeError(
                f"failed to obtain Kokoro voice {voice_id!r}: {error}"
            ) from error
        return voice_path


def _download_model_file(
    config: KokoroRuntimeConfig,
    filename: str,
    downloader: Callable[..., str],
) -> Path:
    path = config.model_dir / filename
    if path.is_file() and path.stat().st_size > 0:
        return path
    try:
        downloaded = Path(
            downloader(
                repo_id=config.repository_id,
                filename=filename,
                local_dir=str(config.model_dir),
            )
        )
    except Exception as error:
        raise EngineRuntimeError(f"failed to obtain Kokoro asset {filename!r}: {error}") from error
    if not downloaded.is_file() or downloaded.stat().st_size == 0:
        raise EngineRuntimeError(f"downloaded Kokoro asset is invalid: {downloaded}")
    return downloaded


def _kokoro_language_code(voice_id: str) -> str:
    """Derive the Kokoro G2P language code from its stable voice identifier."""
    language_code = voice_id[:1]
    if language_code not in {"a", "b", "e", "f", "h", "i", "j", "p", "z"}:
        raise EngineRuntimeError(
            f"cannot determine Kokoro language pipeline for voice {voice_id!r}"
        )
    return language_code


def _collect_audio(outputs: Iterable[Any]) -> tuple[list[float], int]:
    samples: list[float] = []
    chunk_count = 0
    for output in outputs:
        audio = getattr(output, "audio", None)
        if audio is None and isinstance(output, (tuple, list)) and len(output) >= 3:
            audio = output[2]
        if audio is None:
            continue
        chunk = list(_flatten_audio(audio))
        if not chunk:
            continue
        samples.extend(chunk)
        chunk_count += 1
    if not samples:
        raise EngineOutputError("Kokoro produced no audio samples")
    return samples, chunk_count


def _flatten_audio(audio: Any) -> Iterator[float]:
    value = audio
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_audio(item)
        return
    try:
        sample = float(value)
    except (TypeError, ValueError) as error:
        raise EngineOutputError(
            f"Kokoro returned a non-numeric audio sample: {value!r}"
        ) from error
    if not math.isfinite(sample):
        raise EngineOutputError("Kokoro returned a non-finite audio sample")
    yield sample


def write_pcm16_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    """Write finite float samples as an atomic mono 16-bit PCM WAV file."""
    if not samples:
        raise EngineOutputError("cannot write an empty audio output")
    if sample_rate <= 0:
        raise EngineOutputError("cannot write audio with an invalid sample rate")
    pcm = array(
        "h",
        (
            round(max(-1.0, min(1.0, sample)) * 32767)
            for sample in samples
            if math.isfinite(sample)
        ),
    )
    if len(pcm) != len(samples):
        raise EngineOutputError("cannot write non-finite audio samples")
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
        raise EngineOutputError(f"failed to write Kokoro WAV output: {error}") from error


def validate_wav_output(
    path: Path,
    *,
    expected_sample_rate: int,
    expected_frames: int,
) -> None:
    """Validate the persisted audio contract before returning it to callers."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise EngineOutputError(f"Kokoro output file is missing or empty: {path}")
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise EngineOutputError("Kokoro output must be mono")
            if wav_file.getsampwidth() != 2:
                raise EngineOutputError("Kokoro output must use 16-bit PCM")
            if wav_file.getframerate() != expected_sample_rate:
                raise EngineOutputError("Kokoro output sample rate does not match the result")
            if wav_file.getnframes() != expected_frames:
                raise EngineOutputError("Kokoro output frame count does not match generated audio")
    except EngineOutputError:
        raise
    except (OSError, EOFError, wave.Error) as error:
        raise EngineOutputError(f"Kokoro output is not a valid WAV file: {error}") from error


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write("\n")
        temporary_path.replace(path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise EngineOutputError(f"failed to write Kokoro metadata: {error}") from error
