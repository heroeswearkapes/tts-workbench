from contextlib import redirect_stdout
from io import StringIO
import json
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tts_workbench.device_diagnostics import (
    DeviceAvailability,
    DeviceReport,
    LoadedRuntime,
    XPUDevice,
    build_device_report,
    collect_loaded_runtimes,
    format_device_report,
    log_startup_report,
    main,
    probe_device_availability,
    select_device,
)


class MockXPU:
    def __init__(self, available=True, names=("Intel Arc Pro B70",)) -> None:
        self.available = available
        self.names = names

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.names) if self.available else 0

    def get_device_name(self, index):
        return self.names[index]

    def get_device_properties(self, index):
        return SimpleNamespace(total_memory=(index + 1) * 16 * 1024**3)


class MockTorch:
    __version__ = "2.test+xpu"

    def __init__(self, xpu):
        self.xpu = xpu


class DeviceDiagnosticTests(unittest.TestCase):
    def test_mocked_xpu_is_reported_and_auto_selected(self) -> None:
        availability = probe_device_availability(
            torch_module=MockTorch(MockXPU(names=("Intel Arc Pro B70", "Second XPU")))
        )
        self.assertTrue(availability.cpu_available)
        self.assertTrue(availability.xpu_available)
        self.assertEqual(len(availability.xpu_devices), 2)
        self.assertEqual(availability.xpu_devices[0].name, "Intel Arc Pro B70")
        self.assertEqual(select_device(availability, "auto", 0), "xpu:0")
        self.assertEqual(select_device(availability, "xpu", 1), "xpu:1")

    def test_mocked_unavailable_xpu_falls_back_to_cpu(self) -> None:
        availability = probe_device_availability(
            torch_module=MockTorch(MockXPU(available=False))
        )
        self.assertFalse(availability.xpu_available)
        self.assertEqual(select_device(availability, "auto"), "cpu")
        with self.assertRaisesRegex(ValueError, "no usable XPU"):
            select_device(availability, "xpu")

    def test_missing_torch_still_reports_cpu(self) -> None:
        availability = probe_device_availability(torch_module=None)
        self.assertTrue(availability.cpu_available)
        self.assertFalse(availability.torch_available)
        self.assertFalse(availability.xpu_available)
        self.assertIn("PyTorch is not installed", availability.warnings[0])

    def test_loaded_engine_and_model_are_collected(self) -> None:
        expected = LoadedRuntime("f5", "f5-tts-v1-base", "xpu:0")

        class Adapter:
            def runtime_status(self):
                return (expected,)

        self.assertEqual(collect_loaded_runtimes([Adapter()]), (expected,))
        report = build_device_report(
            registry=[Adapter()],
            torch_module=MockTorch(MockXPU()),
        )
        payload = report.as_dict()
        self.assertEqual(payload["loaded_engine"], "f5")
        self.assertEqual(payload["loaded_model"], "f5-tts-v1-base")

    def test_human_report_and_startup_logging_are_clear(self) -> None:
        availability = DeviceAvailability(
            cpu_available=True,
            cpu_name="Mock CPU",
            logical_cpu_count=8,
            torch_available=True,
            torch_version="2.test+xpu",
            xpu_available=True,
            xpu_devices=(XPUDevice(0, "Intel Arc Pro B70", 16 * 1024**3),),
        )
        report = DeviceReport(
            availability=availability,
            requested_device="auto",
            selected_device="xpu:0",
            selection_error=None,
            loaded_runtimes=(LoadedRuntime("f5", "f5-tts-v1-base", "xpu:0"),),
        )
        text = format_device_report(report)
        self.assertIn("CPU available: yes", text)
        self.assertIn("XPU available: yes", text)
        self.assertIn("Selected device: xpu:0", text)
        self.assertIn("Loaded engine: f5", text)
        self.assertIn("Loaded model: f5-tts-v1-base", text)

        logger = logging.getLogger("tests.device_startup")
        with self.assertLogs(logger, level="INFO") as captured:
            log_startup_report(report, logger=logger)
        logs = "\n".join(captured.output)
        self.assertIn("Selected device=xpu:0", logs)
        self.assertIn("Loaded engine=f5 model=f5-tts-v1-base device=xpu:0", logs)

    def test_json_command_reports_mocked_devices(self) -> None:
        report = build_device_report(torch_module=MockTorch(MockXPU()))
        output = StringIO()
        with patch(
            "tts_workbench.device_diagnostics.build_device_report",
            return_value=report,
        ), redirect_stdout(output):
            exit_code = main(["--json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["cpu"]["available"])
        self.assertEqual(payload["selected_device"], "xpu:0")


if __name__ == "__main__":
    unittest.main()
