"""Command-line entry point for the Gradio application."""

import logging
import os
from pathlib import Path

from tts_workbench.bootstrap import create_default_registry
from tts_workbench.device_diagnostics import build_device_report, log_startup_report
from tts_workbench.ui.app import build_app


def _server_port() -> int:
    value = os.environ.get("GRADIO_SERVER_PORT", "7860")
    try:
        port = int(value)
    except ValueError as error:
        raise ValueError("GRADIO_SERVER_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("GRADIO_SERVER_PORT must be between 1 and 65535")
    return port


def _gradio_allowed_paths() -> list[str]:
    """Return only the storage paths Gradio may serve to browser clients."""
    root = Path(os.environ.get("TTS_WORKBENCH_ROOT", ".")).expanduser().resolve()
    return [str(root / directory) for directory in ("output", "voices", "input")]


def main() -> None:
    """Build and launch the local workbench."""
    logging.basicConfig(
        level=os.environ.get("TTS_WORKBENCH_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    registry = create_default_registry()
    report = build_device_report(registry=registry)
    log_startup_report(report)
    app = build_app(registry)
    app.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=_server_port(),
        allowed_paths=_gradio_allowed_paths(),
    )


if __name__ == "__main__":
    main()
