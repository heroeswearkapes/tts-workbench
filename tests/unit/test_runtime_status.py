from pathlib import Path
import unittest

from tts_workbench.bootstrap import create_default_registry
from tts_workbench.device_diagnostics import (
    DeviceAvailability,
    LoadedRuntime,
    XPUDevice,
)
from tts_workbench.domain import SynthesisResult
from tts_workbench.runtime_status import (
    RuntimePhase,
    calculate_real_time_factor,
    format_runtime_status,
    status_after_error,
    status_after_generation,
    status_after_runtime_load,
    status_before_generation,
    status_for_selection,
    status_with_runtime_lifecycle,
)


def availability(*, xpu: bool = False) -> DeviceAvailability:
    return DeviceAvailability(
        cpu_available=True,
        cpu_name="Test CPU",
        logical_cpu_count=8,
        torch_available=True,
        torch_version="2.test",
        xpu_available=xpu,
        xpu_devices=(XPUDevice(0, "Intel Arc Pro B70"),) if xpu else (),
    )


class RuntimeStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = create_default_registry()

    def test_cpu_runtime_status_reports_device_and_precision(self) -> None:
        adapter = self.registry.get("kokoro")
        model = adapter.require_model(adapter.descriptor.default_model_id)
        status = status_for_selection(
            adapter.descriptor,
            model,
            loaded_runtimes=(
                LoadedRuntime(
                    "kokoro",
                    model.id,
                    "cpu",
                    precision="fp32",
                ),
            ),
            availability=availability(),
        )

        self.assertEqual(status.phase, RuntimePhase.READY)
        self.assertTrue(status.runtime_loaded)
        self.assertEqual(status.device_backend, "cpu")
        self.assertEqual(status.device_name, "Test CPU")
        self.assertEqual(status.precision, "fp32")

    def test_mocked_xpu_runtime_status_reports_selected_device(self) -> None:
        adapter = self.registry.get("f5")
        model = adapter.require_model(adapter.descriptor.default_model_id)
        status = status_for_selection(
            adapter.descriptor,
            model,
            loaded_runtimes=(
                LoadedRuntime(
                    "f5",
                    model.id,
                    "xpu:0",
                    precision="fp32",
                ),
            ),
            availability=availability(xpu=True),
        )

        self.assertEqual(status.device_backend, "xpu")
        self.assertEqual(status.device_name, "Intel Arc Pro B70")
        self.assertEqual(status.precision, "fp32")
        self.assertTrue(status.runtime_loaded)

    def test_unloaded_qwen_runtime_is_available_without_eager_loading(self) -> None:
        adapter = self.registry.get("qwen")
        model = adapter.require_model(adapter.descriptor.default_model_id)
        status = status_for_selection(
            adapter.descriptor,
            model,
            availability=availability(),
        )

        self.assertEqual(status.phase, RuntimePhase.UNLOADED)
        self.assertFalse(status.runtime_loaded)
        self.assertIsNone(status.device_backend)
        self.assertEqual(status.warnings, ())

    def test_generation_lifecycle_records_timing_and_audio_duration(self) -> None:
        adapter = self.registry.get("kokoro")
        model = adapter.require_model(adapter.descriptor.default_model_id)
        status = status_for_selection(
            adapter.descriptor,
            model,
            availability=availability(),
        )
        status = status_before_generation(status)
        self.assertEqual(status.phase, RuntimePhase.LOADING)
        status = status_after_runtime_load(
            status,
            (LoadedRuntime("kokoro", model.id, "cpu", precision="fp32"),),
            availability=availability(),
        )
        self.assertEqual(status.phase, RuntimePhase.GENERATING)
        status = status_after_generation(
            status,
            SynthesisResult(
                engine_id="kokoro",
                model_id=model.id,
                sample_rate=24_000,
                audio_path=Path("audio.wav"),
                generation_seconds=2.0,
                duration_seconds=5.0,
            ),
        )

        self.assertEqual(status.phase, RuntimePhase.READY)
        self.assertEqual(status.last_generation_seconds, 2.0)
        self.assertEqual(status.audio_duration_seconds, 5.0)
        self.assertAlmostEqual(status.real_time_factor, 0.4)
        rendered = format_runtime_status(status)
        self.assertIn("2.000 s", rendered)
        self.assertIn("5.000 s", rendered)
        self.assertIn("0.400×", rendered)

    def test_real_time_factor_handles_missing_and_zero_audio(self) -> None:
        self.assertAlmostEqual(calculate_real_time_factor(1.5, 3.0), 0.5)
        self.assertIsNone(calculate_real_time_factor(None, 3.0))
        self.assertIsNone(calculate_real_time_factor(1.5, None))
        self.assertIsNone(calculate_real_time_factor(1.5, 0.0))

    def test_errors_are_reported_without_discarding_loaded_state(self) -> None:
        adapter = self.registry.get("f5")
        model = adapter.require_model(adapter.descriptor.default_model_id)
        status = status_for_selection(
            adapter.descriptor,
            model,
            loaded_runtimes=(LoadedRuntime("f5", model.id, "cuda:0", "Test GPU"),),
            availability=availability(),
        )
        status = status_after_error(status, "generation failed")

        self.assertTrue(status.runtime_loaded)
        self.assertEqual(status.device_backend, "cuda")
        self.assertEqual(status.device_name, "Test GPU")
        self.assertEqual(status.phase, RuntimePhase.ERROR)
        self.assertIn("generation failed", format_runtime_status(status))

    def test_unloading_status_reports_active_runtime_and_device_cleanup(self) -> None:
        adapter = self.registry.get("f5")
        model = adapter.require_model(adapter.descriptor.default_model_id)
        status = status_for_selection(
            adapter.descriptor,
            model,
            availability=availability(xpu=True),
        )
        status = status_with_runtime_lifecycle(
            status,
            phase=RuntimePhase.UNLOADING,
            active_runtime_engine_id="kokoro",
            active_runtime_model_id="kokoro-82m",
            device_cleanup_status="running: releasing runtime and device caches",
        )

        self.assertEqual(status.phase, RuntimePhase.UNLOADING)
        self.assertFalse(status.runtime_loaded)
        rendered = format_runtime_status(status)
        self.assertIn("Active runtime | `kokoro` / `kokoro-82m`", rendered)
        self.assertIn("State | unloading", rendered)
        self.assertIn("Device cleanup | running", rendered)


if __name__ == "__main__":
    unittest.main()
