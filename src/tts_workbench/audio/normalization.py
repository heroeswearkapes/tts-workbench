"""Normalize user-provided audio into a predictable inference format."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import wave


class AudioNormalizationError(RuntimeError):
    """Raised when audio cannot be converted to the normalized contract."""


class AudioNormalizerDependencyError(AudioNormalizationError):
    """Raised when the configured decoder executable is unavailable."""


@contextmanager
def normalize_audio_with_ffmpeg(
    source: Path,
    target_sample_rate: int,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> Iterator[Path]:
    """Yield a mono PCM16 WAV, transcoding to a temporary file when necessary."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise AudioNormalizationError(f"reference audio does not exist: {source}")
    if target_sample_rate <= 0:
        raise ValueError("target audio sample rate must be positive")

    if _is_target_pcm_wav(source, target_sample_rate):
        yield source
        return

    with TemporaryDirectory(prefix="tts-workbench-audio-") as temporary_directory:
        normalized = Path(temporary_directory) / "normalized.wav"
        command = [
            ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map_metadata",
            "-1",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(target_sample_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(normalized),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except FileNotFoundError as error:
            raise AudioNormalizerDependencyError(
                "ffmpeg is required to normalize reference audio"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise AudioNormalizationError("reference audio normalization timed out") from error

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
            raise AudioNormalizationError(
                f"failed to normalize reference audio with ffmpeg: {detail}"
            )

        _validate_target_pcm_wav(normalized, target_sample_rate)
        yield normalized


def _is_target_pcm_wav(path: Path, target_sample_rate: int) -> bool:
    try:
        _validate_target_pcm_wav(path, target_sample_rate)
    except AudioNormalizationError:
        return False
    return True


def _validate_target_pcm_wav(path: Path, target_sample_rate: int) -> None:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise AudioNormalizationError("normalized reference audio is missing or empty")
        with wave.open(str(path), "rb") as audio:
            if audio.getcomptype() != "NONE":
                raise AudioNormalizationError("normalized reference audio must use PCM encoding")
            if audio.getnchannels() != 1:
                raise AudioNormalizationError("normalized reference audio must be mono")
            if audio.getsampwidth() != 2:
                raise AudioNormalizationError("normalized reference audio must use 16-bit PCM")
            if audio.getframerate() != target_sample_rate:
                raise AudioNormalizationError(
                    "normalized reference audio sample rate does not match the target"
                )
            if audio.getnframes() <= 0:
                raise AudioNormalizationError("normalized reference audio contains no frames")
    except AudioNormalizationError:
        raise
    except (OSError, EOFError, wave.Error) as error:
        raise AudioNormalizationError(
            f"normalized reference audio is not a valid WAV file: {error}"
        ) from error
