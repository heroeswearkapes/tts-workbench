import unittest

from tts_workbench.domain import (
    Capability,
    DeviceBackend,
    DeviceSpec,
    InputFieldSpec,
    ModelDescriptor,
    Precision,
    SettingSpec,
    SynthesisRequest,
    VoiceDescriptor,
)
from tts_workbench.engines.qwen import QwenAdapter
from tts_workbench.engines.qwen_runtime import QWEN_MODEL_ID
from tts_workbench.engines.qwen_runtime import QWEN_MODEL_IDS


class QwenAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = QwenAdapter()

    def test_descriptors_are_valid_and_model_ids_are_unique(self) -> None:
        descriptor = self.adapter.descriptor
        models = self.adapter.list_models()

        self.assertEqual(descriptor.id, "qwen")
        self.assertEqual(descriptor.name, "Qwen3-TTS")
        self.assertTrue(descriptor.available)
        self.assertIsNone(descriptor.unavailable_reason)
        self.assertTrue(all(isinstance(model, ModelDescriptor) for model in models))
        self.assertEqual(len(models), 5)
        self.assertEqual(len({model.id for model in models}), len(models))
        self.assertEqual(tuple(model.id for model in models), QWEN_MODEL_IDS)
        self.assertIn(descriptor.default_model_id, {model.id for model in models})
        self.assertEqual(descriptor.default_model_id, QWEN_MODEL_ID)
        for model in models:
            self.assertEqual(
                model.supported_backends,
                (DeviceBackend.CPU, DeviceBackend.XPU),
            )
            self.assertEqual(
                model.supported_precisions,
                (Precision.BF16, Precision.FP32),
            )

    def test_capability_reporting_matches_initial_custom_voice_runtime(self) -> None:
        self.assertEqual(
            self.adapter.descriptor.capabilities,
            frozenset(
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
        )

    def test_custom_voice_model_exposes_nine_described_voices(self) -> None:
        voices = self.adapter.list_voices(QWEN_MODEL_ID)

        self.assertEqual(len(voices), 9)
        self.assertTrue(all(isinstance(voice, VoiceDescriptor) for voice in voices))
        self.assertEqual(
            {voice.id for voice in voices},
            {
                "vivian",
                "serena",
                "uncle_fu",
                "dylan",
                "eric",
                "ryan",
                "aiden",
                "ono_anna",
                "sohee",
            },
        )
        self.assertTrue(
            all(voice.language and voice.gender and voice.category for voice in voices)
        )

    def test_model_exposes_typed_input_and_setting_schemas(self) -> None:
        inputs = self.adapter.input_schema(QWEN_MODEL_ID)
        settings = self.adapter.setting_schema(QWEN_MODEL_ID)

        self.assertTrue(all(isinstance(field, InputFieldSpec) for field in inputs))
        self.assertTrue(all(isinstance(setting, SettingSpec) for setting in settings))
        self.assertEqual(
            [field.key for field in inputs],
            ["voice", "language", "style_instruction"],
        )
        self.assertEqual(
            [setting.key for setting in settings],
            ["temperature", "top_p", "top_k", "repetition_penalty"],
        )

    def test_capability_validation_rejects_unsupported_values_before_loading(self) -> None:
        request = SynthesisRequest(
            engine_id="qwen",
            model_id=QWEN_MODEL_ID,
            text="Invalid request",
            inputs={
                "voice": "not-a-speaker",
                "language": "english",
                "style_instruction": "",
            },
        )
        with self.assertRaisesRegex(ValueError, "unknown Qwen voice"):
            self.adapter.generate(request)

        cuda_request = SynthesisRequest(
            engine_id="qwen",
            model_id=QWEN_MODEL_ID,
            text="Invalid device",
            inputs={"voice": "vivian", "language": "english"},
            device=DeviceSpec(backend=DeviceBackend.CUDA, index=0),
        )
        with self.assertRaisesRegex(ValueError, "CPU and XPU"):
            self.adapter.generate(cuda_request)


if __name__ == "__main__":
    unittest.main()
