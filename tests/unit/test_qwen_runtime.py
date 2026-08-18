import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

from tts_workbench.device_diagnostics import (
    DeviceAvailability,
    XPUDevice,
)
from tts_workbench.domain import (
    DeviceBackend,
    DeviceSpec,
    Precision,
    SynthesisRequest,
)
from tts_workbench.engines.base import EngineDependencyError
from tts_workbench.engines.qwen import QwenAdapter, resolve_qwen_device
from tts_workbench.engines.qwen_runtime import (
    QWEN_MODEL_ID,
    QWEN_MODEL_IDS,
    QWEN_REPOSITORY,
    OfficialQwenBackend,
    QwenGeneratedAudio,
    QwenRuntimeConfig,
    QwenSession,
    detect_qwen_dependencies,
    qwen_runtime_available,
)
from tts_workbench.runtime_status import RuntimePhase, status_for_selection


class FakeQwenBackend:
    def __init__(self, device: str = "cpu", precision: str = "fp32") -> None:
        self.device = device
        self.precision = precision
        self.calls = []

    def generate(
        self,
        *,
        text,
        voice_id,
        language,
        style_instruction,
        generation_settings,
    ):
        self.calls.append(
            {
                "text": text,
                "voice_id": voice_id,
                "language": language,
                "style_instruction": style_instruction,
                "generation_settings": dict(generation_settings),
            }
        )
        return QwenGeneratedAudio(
            samples=[0.0, 0.25, -0.25, 0.0],
            sample_rate=24_000,
        )


class QwenRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = QwenRuntimeConfig.from_root(self.root)

    def make_adapter(self, *, device: str = "cpu"):
        backends = []
        created_configs = []

        def create_session(config):
            created_configs.append(config)
            backend = FakeQwenBackend(config.device, config.precision)
            backends.append(backend)
            return QwenSession(config, backend)

        adapter = QwenAdapter(
            self.config,
            session_factory=create_session,
            device_resolver=lambda request: device,
        )
        return adapter, backends, created_configs

    def make_request(self, **changes) -> SynthesisRequest:
        values = {
            "engine_id": "qwen",
            "model_id": QWEN_MODEL_ID,
            "text": "Hello from Qwen",
            "inputs": {
                "voice": "aiden",
                "language": "english",
                "style_instruction": "Speak warmly",
            },
            "settings": {
                "temperature": 0.7,
                "top_p": 0.85,
                "top_k": 40,
                "repetition_penalty": 1.1,
            },
        }
        values.update(changes)
        return SynthesisRequest(**values)

    def test_adapter_creates_and_caches_runtime_session(self) -> None:
        adapter, _backends, configs = self.make_adapter()

        first = adapter.create_session(QWEN_MODEL_ID, "cpu", "fp32")
        second = adapter.create_session(QWEN_MODEL_ID, "cpu", "fp32")

        self.assertIsInstance(first, QwenSession)
        self.assertIs(first, second)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].repository_id, QWEN_REPOSITORY)
        self.assertEqual(
            configs[0].model_dir,
            self.root / "models" / "qwen" / QWEN_MODEL_ID.rsplit("/", 1)[-1],
        )

        released = adapter.unload_runtime(QWEN_MODEL_ID)
        self.assertEqual(released, (first,))
        self.assertEqual(adapter.runtime_status(), ())

    def test_official_model_ids_are_repository_ids(self) -> None:
        self.assertEqual(
            QWEN_MODEL_IDS,
            (
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            ),
        )

    def test_dependency_detection_and_runtime_availability(self) -> None:
        versions = {
            "qwen-tts": "0.1.1",
            "transformers": "4.57.3",
            "accelerate": "1.12.0",
            "torch": "2.11.0+xpu",
            "torchaudio": "2.11.0+xpu",
        }
        with patch(
            "tts_workbench.engines.qwen_runtime.importlib_metadata.version",
            side_effect=lambda distribution: versions[distribution],
        ):
            status = detect_qwen_dependencies()
            self.assertTrue(status.available)
            self.assertEqual(status.version("torch"), "2.11.0+xpu")
            self.assertTrue(qwen_runtime_available())

        def missing_qwen(distribution):
            if distribution == "qwen-tts":
                raise importlib_metadata.PackageNotFoundError(distribution)
            return versions[distribution]

        with patch(
            "tts_workbench.engines.qwen_runtime.importlib_metadata.version",
            side_effect=missing_qwen,
        ):
            status = detect_qwen_dependencies()
            self.assertFalse(status.available)
            self.assertIn("qwen-tts is not installed", status.problems)
            self.assertFalse(qwen_runtime_available())

    def test_missing_dependencies_fail_only_when_qwen_is_requested(self) -> None:
        adapter = QwenAdapter(
            self.config,
            device_resolver=lambda request: "cpu",
        )
        self.assertTrue(adapter.descriptor.available)

        with patch(
            "tts_workbench.engines.qwen_runtime._import_qwen_dependencies",
            side_effect=ModuleNotFoundError("No module named 'qwen_tts'"),
        ):
            with self.assertRaisesRegex(
                EngineDependencyError,
                "Install the project with the 'qwen' extra",
            ):
                adapter.generate(self.make_request())

    def test_fake_generation_persists_normalized_audio_and_metadata(self) -> None:
        adapter, backends, _configs = self.make_adapter()

        result = adapter.generate(self.make_request())

        self.assertEqual(result.engine_id, "qwen")
        self.assertEqual(result.model_id, QWEN_MODEL_ID)
        self.assertEqual(result.voice_id, "aiden")
        self.assertEqual(result.sample_rate, 24_000)
        self.assertAlmostEqual(result.duration_seconds, 4 / 24_000)
        self.assertGreaterEqual(result.generation_seconds, 0)
        self.assertTrue(result.audio_path.is_relative_to(self.config.output_dir))
        self.assertTrue(result.audio_path.is_file())
        with wave.open(str(result.audio_path), "rb") as output:
            self.assertEqual(output.getnchannels(), 1)
            self.assertEqual(output.getsampwidth(), 2)
            self.assertEqual(output.getframerate(), 24_000)
            self.assertEqual(output.getnframes(), 4)

        call = backends[0].calls[0]
        self.assertEqual(call["voice_id"], "aiden")
        self.assertEqual(call["language"], "english")
        self.assertEqual(call["style_instruction"], "Speak warmly")
        self.assertEqual(call["generation_settings"]["top_k"], 40)

        metadata_path = Path(result.metadata["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["repository_id"], QWEN_REPOSITORY)
        self.assertEqual(metadata["voice_id"], "aiden")
        self.assertEqual(metadata["language"], "english")
        self.assertEqual(metadata["style_instruction"], "Speak warmly")
        self.assertFalse(metadata["style_instruction_supported"])
        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["precision"], "fp32")

    def test_official_backend_forwards_custom_voice_controls(self) -> None:
        class FakeOfficialModel:
            def __init__(self) -> None:
                self.calls = []

            def generate_custom_voice(self, **kwargs):
                self.calls.append(kwargs)
                return [[0.0, 0.1]], 24_000

        model = FakeOfficialModel()
        backend = OfficialQwenBackend(model, "cpu", "fp32")

        generated = backend.generate(
            text="Styled speech",
            voice_id="serena",
            language="chinese",
            style_instruction="Speak softly",
            generation_settings={
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 30,
                "repetition_penalty": 1.05,
            },
        )

        self.assertEqual(generated.sample_rate, 24_000)
        self.assertEqual(model.calls[0]["speaker"], "serena")
        self.assertEqual(model.calls[0]["language"], "chinese")
        self.assertEqual(model.calls[0]["instruct"], "Speak softly")
        self.assertTrue(model.calls[0]["non_streaming_mode"])
        self.assertEqual(model.calls[0]["top_k"], 30)

    def test_runtime_status_reports_mocked_xpu_session(self) -> None:
        adapter, _backends, _configs = self.make_adapter(device="xpu:0")
        adapter.generate(
            self.make_request(
                device=DeviceSpec(
                    backend=DeviceBackend.XPU,
                    index=0,
                    precision=Precision.BF16,
                )
            )
        )

        loaded = adapter.runtime_status()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].engine_id, "qwen")
        self.assertEqual(loaded[0].model_id, QWEN_MODEL_ID)
        self.assertEqual(loaded[0].device, "xpu:0")
        self.assertEqual(loaded[0].precision, "bf16")

        model = adapter.require_model(QWEN_MODEL_ID)
        status = status_for_selection(
            adapter.descriptor,
            model,
            loaded_runtimes=loaded,
            availability=DeviceAvailability(
                cpu_available=True,
                cpu_name="Test CPU",
                logical_cpu_count=8,
                torch_available=True,
                torch_version="2.test+xpu",
                xpu_available=True,
                xpu_devices=(XPUDevice(0, "Intel Arc Pro B70"),),
            ),
        )
        self.assertEqual(status.phase, RuntimePhase.READY)
        self.assertEqual(status.device_backend, "xpu")
        self.assertEqual(status.device_name, "Intel Arc Pro B70")
        self.assertEqual(status.precision, "bf16")

    def test_auto_device_uses_shared_diagnostics_cpu_fallback(self) -> None:
        cpu_only = DeviceAvailability(
            cpu_available=True,
            cpu_name="Test CPU",
            logical_cpu_count=8,
            torch_available=True,
            torch_version="2.test+xpu",
            xpu_available=False,
            xpu_devices=(),
        )
        with patch(
            "tts_workbench.engines.qwen.probe_device_availability",
            return_value=cpu_only,
        ):
            self.assertEqual(resolve_qwen_device(DeviceSpec()), "cpu")
            with self.assertRaisesRegex(ValueError, "no usable XPU"):
                resolve_qwen_device(
                    DeviceSpec(backend=DeviceBackend.XPU, index=0)
                )


if __name__ == "__main__":
    unittest.main()
