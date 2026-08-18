import os
import unittest
from unittest.mock import MagicMock, patch

from tts_workbench.__main__ import _gradio_allowed_paths, _server_port, main


class ApplicationMainTests(unittest.TestCase):
    def test_allowed_paths_include_runtime_artifacts_but_exclude_models(self) -> None:
        with patch.dict(os.environ, {"TTS_WORKBENCH_ROOT": "/"}, clear=True):
            self.assertEqual(_gradio_allowed_paths(), ["/output", "/voices", "/input"])
            self.assertNotIn("/models", _gradio_allowed_paths())

    def test_server_port_uses_container_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_server_port(), 7860)

    def test_server_port_rejects_invalid_values(self) -> None:
        for value in ("invalid", "0", "65536"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"GRADIO_SERVER_PORT": value},
            ):
                with self.assertRaisesRegex(ValueError, "GRADIO_SERVER_PORT"):
                    _server_port()

    @patch("tts_workbench.__main__.log_startup_report")
    @patch("tts_workbench.__main__.build_device_report")
    @patch("tts_workbench.__main__.build_app")
    @patch("tts_workbench.__main__.create_default_registry")
    def test_main_binds_to_configured_gradio_address(
        self,
        create_registry: MagicMock,
        build_app: MagicMock,
        build_report: MagicMock,
        log_report: MagicMock,
    ) -> None:
        registry = create_registry.return_value
        app = build_app.return_value
        queued_app = app.queue.return_value
        with patch.dict(
            os.environ,
            {
                "GRADIO_SERVER_NAME": "0.0.0.0",
                "GRADIO_SERVER_PORT": "7860",
                "TTS_WORKBENCH_ROOT": "/",
            },
        ):
            main()

        build_report.assert_called_once_with(registry=registry)
        log_report.assert_called_once_with(build_report.return_value)
        build_app.assert_called_once_with(registry)
        app.queue.assert_called_once_with(default_concurrency_limit=1)
        queued_app.launch.assert_called_once_with(
            server_name="0.0.0.0",
            server_port=7860,
            allowed_paths=["/output", "/voices", "/input"],
        )


if __name__ == "__main__":
    unittest.main()
