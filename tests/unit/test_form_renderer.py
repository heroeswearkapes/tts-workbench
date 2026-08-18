import unittest

from tts_workbench.bootstrap import create_default_registry
from tts_workbench.ui.controller import get_engine_form
from tts_workbench.ui.form_renderer import render_input, render_setting


class FakeGradio:
    """Records component types without requiring Gradio in the unit environment."""

    @staticmethod
    def Dropdown(**kwargs):  # noqa: N802 - mirrors the Gradio API
        return "dropdown", kwargs

    @staticmethod
    def Textbox(**kwargs):  # noqa: N802 - mirrors the Gradio API
        return "textbox", kwargs

    @staticmethod
    def Audio(**kwargs):  # noqa: N802 - mirrors the Gradio API
        return "audio", kwargs

    @staticmethod
    def Number(**kwargs):  # noqa: N802 - mirrors the Gradio API
        return "number", kwargs

    @staticmethod
    def Slider(**kwargs):  # noqa: N802 - mirrors the Gradio API
        return "slider", kwargs

    @staticmethod
    def Checkbox(**kwargs):  # noqa: N802 - mirrors the Gradio API
        return "checkbox", kwargs


class FormRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = create_default_registry()

    def test_kokoro_schema_renders_voice_dropdown_and_speed_slider(self) -> None:
        form = get_engine_form(self.registry, "kokoro")
        components = [render_input(FakeGradio, form, field) for field in form.inputs]
        components += [
            render_setting(FakeGradio, form, setting) for setting in form.settings
        ]

        self.assertEqual([kind for kind, _ in components], ["dropdown", "slider"])
        self.assertEqual(
            components[0][1]["choices"],
            [(voice.name, voice.id) for voice in form.voices],
        )
        self.assertEqual(components[0][1]["preserved_by_key"], "value")

    def test_f5_schema_renders_audio_and_transcript_without_voice(self) -> None:
        form = get_engine_form(self.registry, "f5")
        components = [render_input(FakeGradio, form, field) for field in form.inputs]

        components += [
            render_setting(FakeGradio, form, setting) for setting in form.settings
        ]
        self.assertEqual(
            [kind for kind, _ in components],
            ["audio", "textbox", "slider"],
        )
        self.assertNotIn("info", components[0][1])
        self.assertEqual(form.voices, ())
        self.assertEqual([setting.key for setting in form.settings], ["speed"])

    def test_qwen_model_schemas_generate_generic_fields(self) -> None:
        expected = {
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": (
                ["voice", "language", "style_instruction"],
                ["dropdown", "dropdown", "textbox"],
            ),
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base": (
                ["language", "reference_audio", "reference_text"],
                ["dropdown", "audio", "textbox"],
            ),
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": (
                ["voice", "language", "style_instruction"],
                ["dropdown", "dropdown", "textbox"],
            ),
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base": (
                ["language", "reference_audio", "reference_text"],
                ["dropdown", "audio", "textbox"],
            ),
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": (
                ["language", "voice_description"],
                ["dropdown", "textbox"],
            ),
        }
        for model_id, (field_keys, component_kinds) in expected.items():
            with self.subTest(model=model_id):
                form = get_engine_form(self.registry, "qwen", model_id)
                components = [
                    render_input(FakeGradio, form, field) for field in form.inputs
                ]
                self.assertEqual([field.key for field in form.inputs], field_keys)
                self.assertEqual(
                    [kind for kind, _ in components],
                    component_kinds,
                )


if __name__ == "__main__":
    unittest.main()
