import unittest

from tts_workbench.bootstrap import create_default_registry
from tts_workbench.engines.f5 import F5Adapter
from tts_workbench.engines.registry import DuplicateEngineError, EngineNotFoundError


class RegistryTests(unittest.TestCase):
    def test_default_registry_contains_engines_in_registration_order(self) -> None:
        registry = create_default_registry()
        self.assertEqual(registry.ids(), ("kokoro", "f5", "qwen"))
        self.assertEqual(len(registry), 3)

    def test_qwen_is_registered_with_its_default_model(self) -> None:
        registry = create_default_registry()
        adapter = registry.get("qwen")

        self.assertEqual(adapter.descriptor.name, "Qwen3-TTS")
        self.assertEqual(
            adapter.descriptor.default_model_id,
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        )

    def test_duplicate_engine_is_rejected(self) -> None:
        registry = create_default_registry()
        with self.assertRaises(DuplicateEngineError):
            registry.register(F5Adapter())

    def test_unknown_engine_uses_domain_error(self) -> None:
        registry = create_default_registry()
        with self.assertRaises(EngineNotFoundError):
            registry.get("missing")


if __name__ == "__main__":
    unittest.main()
