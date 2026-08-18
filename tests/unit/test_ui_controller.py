import unittest
from unittest.mock import patch

from tts_workbench.bootstrap import create_default_registry
from tts_workbench.domain import SynthesisResult
from tts_workbench.ui.controller import (
    build_request,
    generate_from_form,
    get_engine_form,
)


class UiControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = create_default_registry()

    def test_kokoro_form_contains_voice_and_speed_only(self) -> None:
        form = get_engine_form(self.registry, "kokoro")
        self.assertEqual([field.key for field in form.inputs], ["voice"])
        self.assertEqual([setting.key for setting in form.settings], ["speed"])

    def test_f5_form_contains_reference_fields_only(self) -> None:
        form = get_engine_form(self.registry, "f5")
        self.assertEqual(
            [field.key for field in form.inputs],
            ["reference_audio", "reference_text"],
        )
        self.assertEqual([setting.key for setting in form.settings], ["speed"])

    def test_engine_switch_ignores_previous_engines_model_id(self) -> None:
        form = get_engine_form(self.registry, "f5", "kokoro-82m")
        self.assertEqual(form.model.id, "f5-tts-v1-base")

    def test_request_contains_selected_non_default_voice_id(self) -> None:
        form = get_engine_form(self.registry, "kokoro")
        request = build_request(form, "Hello world", ["af_bella"], [1.25])
        self.assertEqual(request.inputs["voice"], "af_bella")
        self.assertEqual(request.settings["speed"], 1.25)

    def test_controller_passes_selected_voice_to_adapter(self) -> None:
        form = get_engine_form(self.registry, "kokoro")
        adapter = self.registry.get("kokoro")
        expected = SynthesisResult(
            engine_id="kokoro",
            model_id="kokoro-82m",
            sample_rate=24_000,
        )

        with patch.object(adapter, "generate", return_value=expected) as generate:
            result = generate_from_form(
                self.registry,
                form,
                "Hello world",
                ["af_bella"],
                [1.0],
            )

        self.assertIs(result, expected)
        request = generate.call_args.args[0]
        self.assertEqual(request.inputs["voice"], "af_bella")

    def test_required_engine_input_is_validated(self) -> None:
        form = get_engine_form(self.registry, "f5")
        with self.assertRaisesRegex(ValueError, "Reference audio"):
            build_request(form, "Hello world", [None, "Transcript"], [1.0])


if __name__ == "__main__":
    unittest.main()
