import unittest

from tts_workbench.bootstrap import create_default_registry
from tts_workbench.domain import Capability, SynthesisRequest


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = create_default_registry()

    def test_kokoro_exposes_preset_voices_and_speed(self) -> None:
        adapter = self.registry.get("kokoro")
        model_id = adapter.descriptor.default_model_id

        self.assertIn(Capability.PRESET_VOICES, adapter.descriptor.capabilities)
        self.assertEqual([field.key for field in adapter.input_schema(model_id)], ["voice"])
        self.assertEqual([setting.key for setting in adapter.setting_schema(model_id)], ["speed"])
        self.assertGreater(len(adapter.list_voices(model_id)), 0)

    def test_kokoro_exposes_complete_voice_catalog(self) -> None:
        adapter = self.registry.get("kokoro")
        voices = adapter.list_voices(adapter.descriptor.default_model_id)

        self.assertEqual(
            {voice.id for voice in voices},
            {
                "af_alloy",
                "af_aoede",
                "af_bella",
                "af_heart",
                "af_jessica",
                "af_kore",
                "af_nicole",
                "af_nova",
                "af_river",
                "af_sarah",
                "af_sky",
                "am_adam",
                "am_echo",
                "am_eric",
                "am_fenrir",
                "am_liam",
                "am_michael",
                "am_onyx",
                "am_puck",
                "am_santa",
                "bf_alice",
                "bf_emma",
                "bf_isabella",
                "bf_lily",
                "bm_daniel",
                "bm_fable",
                "bm_george",
                "bm_lewis",
                "ef_dora",
                "em_alex",
                "em_santa",
                "ff_siwis",
                "hf_alpha",
                "hf_beta",
                "hm_omega",
                "hm_psi",
                "if_sara",
                "im_nicola",
                "jf_alpha",
                "jf_gongitsune",
                "jf_nezumi",
                "jf_tebukuro",
                "jm_kumo",
                "pf_dora",
                "pm_alex",
                "pm_santa",
                "zf_xiaobei",
                "zf_xiaoni",
                "zf_xiaoxiao",
                "zf_xiaoyi",
                "zm_yunjian",
                "zm_yunxi",
                "zm_yunxia",
                "zm_yunyang",
            },
        )
        self.assertEqual(len(voices), 54)
        self.assertTrue(
            all(voice.language and voice.gender and voice.category for voice in voices)
        )

    def test_kokoro_voice_ids_are_unique(self) -> None:
        adapter = self.registry.get("kokoro")
        voices = adapter.list_voices(adapter.descriptor.default_model_id)
        voice_ids = [voice.id for voice in voices]

        self.assertEqual(len(voice_ids), len(set(voice_ids)))

    def test_f5_exposes_reference_fields_without_preset_voices(self) -> None:
        adapter = self.registry.get("f5")
        model_id = adapter.descriptor.default_model_id

        self.assertIn(Capability.VOICE_CLONING, adapter.descriptor.capabilities)
        self.assertEqual(
            [field.key for field in adapter.input_schema(model_id)],
            ["reference_audio", "reference_text"],
        )
        self.assertEqual(
            [setting.key for setting in adapter.setting_schema(model_id)],
            ["speed"],
        )
        self.assertEqual(adapter.list_voices(model_id), ())


if __name__ == "__main__":
    unittest.main()
