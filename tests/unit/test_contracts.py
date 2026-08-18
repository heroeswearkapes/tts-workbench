from types import MappingProxyType
import unittest

from tts_workbench.domain import (
    ChoiceSource,
    DeviceBackend,
    DeviceSpec,
    FieldKind,
    InputFieldSpec,
    ModelDescriptor,
    SynthesisRequest,
    VoiceDescriptor,
)


class ContractTests(unittest.TestCase):
    def test_model_descriptor_accepts_repository_identifier(self) -> None:
        model = ModelDescriptor(
            id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            name="Qwen3-TTS",
        )
        self.assertEqual(model.id, "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

        with self.assertRaisesRegex(ValueError, "model or repository identifier"):
            ModelDescriptor(id="Qwen//invalid", name="Invalid")

    def test_voice_descriptor_supports_catalog_metadata(self) -> None:
        voice = VoiceDescriptor(
            id="bf_emma",
            name="Emma",
            language="en-GB",
            accent="British",
            gender="female",
            category="British English",
        )

        self.assertEqual(voice.language, "en-GB")
        self.assertEqual(voice.accent, "British")
        self.assertEqual(voice.gender, "female")
        self.assertEqual(voice.category, "British English")

    def test_device_rejects_cpu_index(self) -> None:
        with self.assertRaises(ValueError):
            DeviceSpec(backend=DeviceBackend.CPU, index=0)

    def test_dropdown_requires_choice_source(self) -> None:
        with self.assertRaises(ValueError):
            InputFieldSpec(key="voice", label="Voice", kind=FieldKind.DROPDOWN)

    def test_static_dropdown_requires_choices(self) -> None:
        with self.assertRaises(ValueError):
            InputFieldSpec(
                key="voice",
                label="Voice",
                kind=FieldKind.DROPDOWN,
                choice_source=ChoiceSource.STATIC,
            )

    def test_request_requires_non_blank_text_and_freezes_mappings(self) -> None:
        with self.assertRaises(ValueError):
            SynthesisRequest(engine_id="kokoro", model_id="kokoro-82m", text="  ")

        request = SynthesisRequest(
            engine_id="kokoro",
            model_id="kokoro-82m",
            text="Hello",
            inputs={"voice": "af_heart"},
        )
        self.assertIsInstance(request.inputs, MappingProxyType)
        with self.assertRaises(TypeError):
            request.inputs["voice"] = "af_bella"  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
