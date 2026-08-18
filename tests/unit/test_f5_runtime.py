import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave
from types import SimpleNamespace

from tts_workbench.domain import Capability, DeviceBackend, DeviceSpec, SynthesisRequest
from tts_workbench.engines.base import EngineRuntimeError
from tts_workbench.engines.f5 import F5Adapter
from tts_workbench.engines.f5_runtime import (
    CPUWhisperTranscriber,
    DecodedReferenceAudio,
    F5GeneratedAudio,
    F5RuntimeConfig,
    F5Session,
    WHISPER_SAMPLE_RATE,
)


class FakeBackend:
    def __init__(self, device="cpu") -> None:
        self.device = device
        self.calls = []

    def generate(self, reference_audio, reference_text, generation_text, speed):
        self.calls.append(
            {
                "reference_audio": reference_audio,
                "reference_text": reference_text,
                "generation_text": generation_text,
                "speed": speed,
            }
        )
        return F5GeneratedAudio(
            samples=[0.0, 0.25, -0.25, 0.0],
            sample_rate=24_000,
            seed=1234,
        )


class FakeTranscriber:
    device = "cpu"

    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, reference_audio, maximum_seconds):
        self.calls.append((reference_audio, maximum_seconds))
        return "Transcribed reference"


class F5RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = F5RuntimeConfig.from_root(self.root)
        self.reference_audio = self.root / "reference.wav"
        with wave.open(str(self.reference_audio), "wb") as reference:
            reference.setnchannels(1)
            reference.setsampwidth(2)
            reference.setframerate(24_000)
            reference.writeframes(b"\x00\x00" * 4)
        self.backend = FakeBackend()
        self.decoded = DecodedReferenceAudio(
            samples=[[0.0], [0.1], [-0.1], [0.0]],
            sample_rate=24_000,
            frames=4,
            channels=1,
        )

    def make_adapter(self, *, transcriber=None, device="cpu"):
        backend = self.backend if device == "cpu" else FakeBackend(device)

        def create_session(config):
            return F5Session(
                config,
                backend,
                audio_decoder=lambda path, maximum: self.decoded,
                transcriber=transcriber,
            )

        return F5Adapter(
            self.config,
            session_factory=create_session,
            device_resolver=lambda request: device,
        ), backend

    def make_request(self, **changes):
        values = {
            "engine_id": "f5",
            "model_id": "f5-tts-v1-base",
            "text": "Generated speech",
            "inputs": {
                "reference_audio": str(self.reference_audio),
                "reference_text": "Reference words",
            },
            "settings": {"speed": 1.1},
        }
        values.update(changes)
        return SynthesisRequest(**values)

    def test_adapter_validation_and_capabilities(self) -> None:
        adapter, _ = self.make_adapter()
        self.assertIn(Capability.VOICE_CLONING, adapter.descriptor.capabilities)
        self.assertIn(Capability.SPEED_CONTROL, adapter.descriptor.capabilities)
        self.assertEqual(adapter.list_voices("f5-tts-v1-base"), ())
        with self.assertRaisesRegex(ValueError, "speed must be at most"):
            adapter.generate(self.make_request(settings={"speed": 3.0}))
        with self.assertRaisesRegex(ValueError, "CPU and XPU"):
            adapter.generate(
                self.make_request(
                    device=DeviceSpec(backend=DeviceBackend.CUDA, index=0)
                )
            )

    def test_adapter_unload_releases_cached_sessions(self) -> None:
        adapter, _ = self.make_adapter()
        session = adapter.create_session("cpu")

        released = adapter.unload_runtime("f5-tts-v1-base")

        self.assertEqual(released, (session,))
        self.assertEqual(adapter.runtime_status(), ())

    def test_missing_reference_audio_is_rejected_before_loading(self) -> None:
        adapter, _ = self.make_adapter()
        request = self.make_request(
            inputs={"reference_audio": None, "reference_text": "Reference words"}
        )
        with self.assertRaisesRegex(ValueError, "reference audio is required"):
            adapter.generate(request)

    @patch("tts_workbench.audio.normalization.subprocess.run")
    def test_unsupported_reference_format_has_clear_runtime_error(self, run) -> None:
        unsupported = self.root / "reference.unsupported"
        unsupported.write_bytes(b"not audio")
        run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stdout="",
            stderr="Invalid data found when processing input",
        )
        session = F5Session(self.config, self.backend)

        with self.assertRaisesRegex(
            EngineRuntimeError,
            "failed to normalize F5 reference audio.*Invalid data",
        ):
            session.generate(
                text="Generated speech",
                engine_id="f5",
                model_id="f5-tts-v1-base",
                reference_audio=unsupported,
                reference_text="Reference words",
                speed=1.0,
            )

    def test_missing_reference_text_uses_cpu_transcription(self) -> None:
        transcriber = FakeTranscriber()
        adapter, backend = self.make_adapter(transcriber=transcriber, device="xpu:0")
        request = self.make_request(
            inputs={"reference_audio": str(self.reference_audio), "reference_text": ""},
            device=DeviceSpec(backend=DeviceBackend.XPU, index=0),
        )
        result = adapter.generate(request)

        self.assertEqual(len(transcriber.calls), 1)
        self.assertEqual(transcriber.device, "cpu")
        self.assertEqual(backend.device, "xpu:0")
        self.assertEqual(backend.calls[0]["reference_text"], "Transcribed reference. ")
        self.assertEqual(result.metadata["reference_text_source"], "transcribed")
        self.assertEqual(adapter.runtime_status()[0].engine_id, "f5")
        self.assertEqual(adapter.runtime_status()[0].model_id, "f5-tts-v1-base")
        self.assertEqual(adapter.runtime_status()[0].device, "xpu:0")

    def test_provided_reference_text_skips_whisper(self) -> None:
        transcriber = FakeTranscriber()
        adapter, backend = self.make_adapter(transcriber=transcriber, device="xpu:0")

        adapter.generate(
            self.make_request(device=DeviceSpec(backend=DeviceBackend.XPU, index=0))
        )

        self.assertEqual(transcriber.calls, [])
        self.assertEqual(backend.device, "xpu:0")
        self.assertEqual(backend.calls[0]["reference_text"], "Reference words. ")

    def test_explicit_whisper_processor_and_model_stay_on_cpu(self) -> None:
        class FakeFeatures:
            def __init__(self) -> None:
                self.to_calls = []

            def to(self, *, device, dtype):
                self.to_calls.append((device, dtype))
                return self

        class FakeProcessor:
            def __init__(self, features) -> None:
                self.features = features
                self.calls = []
                self.decode_calls = []

            def __call__(self, samples, **kwargs):
                self.calls.append((samples, kwargs))
                return SimpleNamespace(input_features=self.features)

            def batch_decode(self, token_ids, **kwargs):
                self.decode_calls.append((token_ids, kwargs))
                return ["Explicit Whisper transcript"]

        class FakeModel:
            def __init__(self) -> None:
                self.to_calls = []
                self.eval_calls = 0
                self.generate_calls = []

            def to(self, *, device, dtype):
                self.to_calls.append((device, dtype))
                return self

            def eval(self):
                self.eval_calls += 1
                return self

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                return [[1, 2, 3]]

        class FakeNumpy:
            float32 = "float32"

            @staticmethod
            def mean(samples, *, axis, dtype):
                return [0.0, 0.1, -0.1]

        class FakeTorch:
            float32 = "torch.float32"

            @staticmethod
            def inference_mode():
                return nullcontext()

        features = FakeFeatures()
        processor = FakeProcessor(features)
        model = FakeModel()
        whisper_decoded = DecodedReferenceAudio(
            samples=[[0.0], [0.1], [-0.1]],
            sample_rate=WHISPER_SAMPLE_RATE,
            frames=3,
            channels=1,
        )
        transcriber = CPUWhisperTranscriber(
            "fake-whisper",
            component_loader=lambda model_id: (processor, model),
            runtime_loader=lambda: (FakeNumpy, FakeTorch),
            audio_normalizer=lambda path, rate: nullcontext(path),
            audio_decoder=lambda path, maximum: whisper_decoded,
        )

        transcript = transcriber.transcribe(self.reference_audio, 12.0)

        self.assertEqual(transcriber.device, "cpu")
        self.assertEqual(transcript, "Explicit Whisper transcript")
        self.assertEqual(model.to_calls, [("cpu", FakeTorch.float32)])
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(features.to_calls, [("cpu", FakeTorch.float32)])
        self.assertEqual(processor.calls[0][1]["sampling_rate"], WHISPER_SAMPLE_RATE)
        self.assertIs(model.generate_calls[0]["input_features"], features)

    def test_fake_generation_returns_normalized_result(self) -> None:
        adapter, backend = self.make_adapter()
        result = adapter.generate(self.make_request())

        self.assertEqual(result.engine_id, "f5")
        self.assertEqual(result.model_id, "f5-tts-v1-base")
        self.assertIsNone(result.voice_id)
        self.assertEqual(result.sample_rate, 24_000)
        self.assertAlmostEqual(result.duration_seconds, 4 / 24_000)
        self.assertGreaterEqual(result.generation_seconds, 0)
        self.assertTrue(result.audio_path.is_relative_to(self.config.output_dir))
        self.assertTrue(result.audio_path.is_file())
        self.assertEqual(backend.calls[0]["speed"], 1.1)
        with wave.open(str(result.audio_path), "rb") as output:
            self.assertEqual(output.getframerate(), 24_000)
            self.assertEqual(output.getnframes(), 4)

    def test_output_metadata_is_correct(self) -> None:
        adapter, _ = self.make_adapter(device="xpu:0")
        result = adapter.generate(
            self.make_request(device=DeviceSpec(backend=DeviceBackend.XPU, index=0))
        )
        metadata_path = Path(result.metadata["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["engine_id"], "f5")
        self.assertEqual(metadata["model_id"], "f5-tts-v1-base")
        self.assertEqual(metadata["device"], "xpu:0")
        self.assertEqual(metadata["speed"], 1.1)
        self.assertEqual(metadata["reference_audio_name"], "reference.wav")
        self.assertEqual(metadata["reference_text_source"], "provided")
        self.assertEqual(metadata["reference_text"], "Reference words. ")
        self.assertEqual(metadata["seed"], 1234)
        self.assertEqual(
            metadata["reference_audio_sha256"],
            hashlib.sha256(self.reference_audio.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
