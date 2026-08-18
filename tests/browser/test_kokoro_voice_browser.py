"""Exercise the Kokoro voice selection through a real Gradio browser client.

Run this test against an already-running test deployment by setting
``TTS_WORKBENCH_BROWSER_URL`` and mounting its output directory at the path in
``TTS_WORKBENCH_BROWSER_OUTPUT_DIR``.
"""

import json
import os
from pathlib import Path
import time
import unittest

try:
    from playwright.sync_api import expect, sync_playwright
except ModuleNotFoundError:
    expect = None
    sync_playwright = None


BROWSER_URL = os.environ.get("TTS_WORKBENCH_BROWSER_URL")


@unittest.skipUnless(
    BROWSER_URL and sync_playwright is not None,
    "requires Playwright and TTS_WORKBENCH_BROWSER_URL",
)
class KokoroVoiceBrowserTests(unittest.TestCase):
    def test_selecting_bella_reaches_persisted_metadata(self) -> None:
        output_dir = Path(
            os.environ.get("TTS_WORKBENCH_BROWSER_OUTPUT_DIR", "/output")
        )
        started = time.time()

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(BROWSER_URL, wait_until="domcontentloaded", timeout=30_000)

            voice = page.get_by_label("Voice")
            expect(voice).to_be_visible(timeout=30_000)
            voice.click()
            page.get_by_role("option", name="Bella", exact=True).click()
            expect(voice).to_have_value("Bella")

            page.get_by_label("Text").fill("Browser lifecycle voice regression test.")
            page.get_by_role("button", name="Generate", exact=True).click()
            metadata_files = []
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                metadata_files = [
                    path
                    for path in output_dir.glob("*/*/metadata.json")
                    if path.stat().st_mtime >= started
                ]
                if metadata_files:
                    break
                page.wait_for_timeout(250)
            browser.close()

        self.assertEqual(len(metadata_files), 1)
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        self.assertEqual(metadata["voice_id"], "af_bella")


if __name__ == "__main__":
    unittest.main()
