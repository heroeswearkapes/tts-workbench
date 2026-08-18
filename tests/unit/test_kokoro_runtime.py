import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave

from tts_workbench.domain import DeviceBackend, DeviceSpec, SynthesisRequest
from tts_workbench.engines.base import EngineOutputError
from tts_workbench.engines.kokoro import KokoroAdapter
from tts_workbench.engines.kokoro_runtime import (
    KokoroRuntimeConfig,
    KokoroSession,
    validate_wav_output,
)


class FakePipeline:
    def __init__(self, chunks=None) -> None:
        self.chunks = chunks if chunks is not None else [[0.0, 0.25], [-0.25, 0.0]]
        self.calls = []

    def __call__(self, text, *, voice, speed, split_pattern):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "split_pattern": split_pattern,
            }
        )
        for chunk in self.chunks:
            yield "text", "phonemes", chunk


class KokoroRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = KokoroRuntimeConfig.from_root(self.root)
        self.config.voice_dir.mkdir(parents=True)
        (self.config.voice_dir / "af_heart.pt").write_bytes(b"fake voice")
        (self.config.voice_dir / "af_bella.pt").write_bytes(b"fake voice")
        (self.config.voice_dir / "bf_emma.pt").write_bytes(b"fake voice")

    def test_adapter_creates_and_caches_valid_session(self) -> None:
        pipeline = FakePipeline()
        created = []

        def create_session(config):
            created.append(config)
            return KokoroSession(config, pipeline)

        adapter = KokoroAdapter(self.config, session_factory=create_session)
        first = adapter.create_session()
        second = adapter.create_session()

        self.assertIsInstance(first, KokoroSession)
        self.assertIs(first, second)
        self.assertEqual(created, [self.config])
        self.assertEqual(adapter.runtime_status()[0].engine_id, "kokoro")
        self.assertEqual(adapter.runtime_status()[0].model_id, "kokoro-82m")
        self.assertEqual(adapter.runtime_status()[0].device, "cpu")

        released = adapter.unload_runtime("kokoro-82m")
        self.assertEqual(adapter.runtime_status(), ())
        third = adapter.create_session()
        self.assertEqual(released, (first,))
        self.assertEqual(adapter.runtime_status()[0].engine_id, "kokoro")
        self.assertIsNot(third, first)
        self.assertEqual(created, [self.config, self.config])

    def test_fake_generation_writes_normalized_result_and_metadata(self) -> None:
        pipeline = FakePipeline()
        adapter = KokoroAdapter(
            self.config,
            session_factory=lambda config: KokoroSession(config, pipeline),
        )
        result = adapter.generate(
            SynthesisRequest(
                engine_id="kokoro",
                model_id="kokoro-82m",
                text="Hello from Kokoro",
                inputs={"voice": "af_heart"},
                settings={"speed": 1.25},
            )
        )

        self.assertEqual(result.engine_id, "kokoro")
        self.assertEqual(result.model_id, "kokoro-82m")
        self.assertEqual(result.voice_id, "af_heart")
        self.assertEqual(result.sample_rate, 24_000)
        self.assertAlmostEqual(result.duration_seconds, 4 / 24_000)
        self.assertIsNotNone(result.generation_seconds)
        self.assertGreaterEqual(result.generation_seconds, 0)
        self.assertTrue(result.audio_path.is_relative_to(self.config.output_dir))
        self.assertTrue(result.audio_path.is_file())
        self.assertEqual(pipeline.calls[0]["speed"], 1.25)
        self.assertEqual(
            pipeline.calls[0]["voice"],
            str(self.config.voice_dir / "af_heart.pt"),
        )

        with wave.open(str(result.audio_path), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 24_000)
            self.assertEqual(wav_file.getnframes(), 4)
            self.assertEqual(wav_file.getnchannels(), 1)

        metadata_path = Path(result.metadata["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["voice_id"], "af_heart")
        self.assertEqual(metadata["chunk_count"], 2)

    def test_non_default_voice_reaches_kokoro_pipeline(self) -> None:
        american_pipeline = FakePipeline()
        british_pipeline = FakePipeline()
        initialized_languages = []

        def create_pipeline(language_code):
            initialized_languages.append(language_code)
            return {"b": british_pipeline}[language_code]

        adapter = KokoroAdapter(
            self.config,
            session_factory=lambda config: KokoroSession(
                config,
                american_pipeline,
                pipeline_factory=create_pipeline,
            ),
        )

        result = adapter.generate(
            SynthesisRequest(
                engine_id="kokoro",
                model_id="kokoro-82m",
                text="Use Emma",
                inputs={"voice": "bf_emma"},
                settings={"speed": 1.0},
            )
        )

        self.assertEqual(result.voice_id, "bf_emma")
        self.assertEqual(initialized_languages, ["b"])
        self.assertEqual(american_pipeline.calls, [])
        self.assertEqual(
            british_pipeline.calls[0]["voice"],
            str(self.config.voice_dir / "bf_emma.pt"),
        )
        metadata = json.loads(
            Path(result.metadata["metadata_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["voice_id"], "bf_emma")

    def test_empty_generation_output_is_rejected(self) -> None:
        adapter = KokoroAdapter(
            self.config,
            session_factory=lambda config: KokoroSession(
                config,
                FakePipeline(chunks=[]),
            ),
        )
        request = SynthesisRequest(
            engine_id="kokoro",
            model_id="kokoro-82m",
            text="No output",
            inputs={"voice": "af_heart"},
            settings={"speed": 1.0},
        )
        with self.assertRaisesRegex(EngineOutputError, "no audio samples"):
            adapter.generate(request)

    def test_non_finite_generation_output_is_rejected(self) -> None:
        adapter = KokoroAdapter(
            self.config,
            session_factory=lambda config: KokoroSession(
                config,
                FakePipeline(chunks=[[float("nan")]]),
            ),
        )
        request = SynthesisRequest(
            engine_id="kokoro",
            model_id="kokoro-82m",
            text="Invalid output",
            inputs={"voice": "af_heart"},
            settings={"speed": 1.0},
        )
        with self.assertRaisesRegex(EngineOutputError, "non-finite"):
            adapter.generate(request)

    def test_xpu_request_is_rejected_before_session_loading(self) -> None:
        adapter = KokoroAdapter(
            self.config,
            session_factory=lambda config: self.fail("session should not load"),
        )
        request = SynthesisRequest(
            engine_id="kokoro",
            model_id="kokoro-82m",
            text="Unsupported device",
            inputs={"voice": "af_heart"},
            settings={"speed": 1.0},
            device=DeviceSpec(backend=DeviceBackend.XPU, index=0),
        )
        with self.assertRaisesRegex(ValueError, "CPU inference only"):
            adapter.generate(request)

    def test_corrupt_wav_output_is_rejected(self) -> None:
        output = self.root / "corrupt.wav"
        output.write_bytes(b"not a wav")

        with self.assertRaisesRegex(EngineOutputError, "valid WAV"):
            validate_wav_output(
                output,
                expected_sample_rate=24_000,
                expected_frames=10,
            )


if __name__ == "__main__":
    unittest.main()
