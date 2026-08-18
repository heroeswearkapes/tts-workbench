from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

from tts_workbench.audio.normalization import (
    AudioNormalizationError,
    normalize_audio_with_ffmpeg,
)


def write_pcm_wav(path: Path, *, sample_rate: int = 24_000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * channels * 8)


class AudioNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    @patch("tts_workbench.audio.normalization.subprocess.run")
    def test_matching_pcm_wav_is_passed_through(self, run) -> None:
        source = self.root / "reference.wav"
        write_pcm_wav(source)

        with normalize_audio_with_ffmpeg(source, 24_000) as normalized:
            self.assertEqual(normalized, source.resolve())

        run.assert_not_called()

    @patch("tts_workbench.audio.normalization.subprocess.run")
    def test_unsupported_format_reports_ffmpeg_failure(self, run) -> None:
        source = self.root / "reference.unsupported"
        source.write_bytes(b"not audio")
        run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stdout="",
            stderr="Invalid data found when processing input",
        )

        with self.assertRaisesRegex(
            AudioNormalizationError,
            "failed to normalize reference audio.*Invalid data",
        ):
            with normalize_audio_with_ffmpeg(source, 24_000):
                pass

    @patch("tts_workbench.audio.normalization.subprocess.run")
    def test_normalized_output_is_validated(self, run) -> None:
        source = self.root / "reference.m4a"
        source.write_bytes(b"fake compressed audio")

        def write_invalid_output(command, **kwargs):
            write_pcm_wav(Path(command[-1]), sample_rate=44_100)
            return subprocess.CompletedProcess(command, 0, "", "")

        run.side_effect = write_invalid_output
        with self.assertRaisesRegex(AudioNormalizationError, "sample rate"):
            with normalize_audio_with_ffmpeg(source, 24_000):
                pass


if __name__ == "__main__":
    unittest.main()
