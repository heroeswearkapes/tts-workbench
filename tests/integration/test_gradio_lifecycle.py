import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

try:
    from gradio.state_holder import SessionState
except ModuleNotFoundError:
    SessionState = None

from tts_workbench.bootstrap import create_default_registry
from tts_workbench.domain import SynthesisRequest
from tts_workbench.engines.kokoro import KokoroAdapter
from tts_workbench.engines.kokoro_runtime import KokoroRuntimeConfig, KokoroSession
from tts_workbench.engines.registry import EngineRegistry
from tts_workbench.runtime_manager import RuntimeManager
from tts_workbench.ui.app import build_app


class FakePipeline:
    def __call__(self, text, *, voice, speed, split_pattern):
        yield "text", "phonemes", [0.0, 0.1, -0.1, 0.0]


@unittest.skipIf(SessionState is None, "Gradio is not installed")
class GradioLifecycleTests(unittest.TestCase):
    def test_ui_engine_switching_uses_metadata_without_loading_models(self) -> None:
        registry = create_default_registry()
        session_factories = []
        for adapter in registry:
            session_factory = MagicMock(
                side_effect=AssertionError("selection must not create a session")
            )
            adapter._session_factory = session_factory  # type: ignore[attr-defined]
            session_factories.append(session_factory)
        runtime_manager = RuntimeManager(
            registry,
            cleanup_scheduler=lambda task: task(),
        )

        async def exercise_selection_lifecycle() -> None:
            app = build_app(registry, runtime_manager)
            state = SessionState(app)
            engine_listener = self._function_with_key(
                app,
                "engine-selection-listener",
            )
            model_listener = self._function_with_key(
                app,
                "model-selection-listener",
            )
            self.assertFalse(engine_listener.queue)
            self.assertFalse(model_listener.queue)
            for engine_id in ("kokoro", "f5", "qwen"):
                adapter = registry.get(engine_id)
                model_id = adapter.descriptor.default_model_id
                await self._process(app, engine_listener, [engine_id], state)
                await self._process(
                    app,
                    model_listener,
                    [engine_id, model_id],
                    state,
                )

        asyncio.run(exercise_selection_lifecycle())

        for session_factory in session_factories:
            session_factory.assert_not_called()
        self.assertIsNone(runtime_manager.snapshot().active)
        self.assertEqual(runtime_manager.loaded_runtimes(), ())

    def test_single_bottom_status_panel_is_wired_to_every_form(self) -> None:
        registry = create_default_registry()
        app = build_app(registry)
        config = app.get_config_file()
        runtime_panels = [
            component
            for component in config["components"]
            if component["props"].get("key") == "runtime-status-panel"
        ]
        self.assertEqual(len(runtime_panels), 1)
        runtime_panel_id = runtime_panels[0]["id"]
        runtime_accordions = [
            component
            for component in config["components"]
            if component["props"].get("key") == "runtime-status-accordion"
        ]
        self.assertEqual(len(runtime_accordions), 1)
        self.assertFalse(runtime_accordions[0]["props"]["open"])
        self.assertEqual(
            config["layout"]["children"][-1],
            {
                "id": runtime_accordions[0]["id"],
                "children": [{"id": runtime_panel_id}],
            },
        )

        expected_form_count = sum(
            len(registry.get(descriptor.id).list_models())
            for descriptor in registry.descriptors()
        )
        begin_listeners = [
            function
            for function in app.fns.values()
            if (function.key or "").endswith(":begin-generation")
        ]
        generation_listeners = [
            function
            for function in app.fns.values()
            if (function.key or "").endswith(":generate-listener")
        ]
        self.assertEqual(len(begin_listeners), expected_form_count)
        self.assertEqual(len(generation_listeners), expected_form_count)
        for listener in [*begin_listeners, *generation_listeners]:
            self.assertEqual(
                [component._id for component in listener.outputs].count(
                    runtime_panel_id
                ),
                1,
            )

    def test_persistent_voice_component_reaches_request_and_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = KokoroRuntimeConfig.from_root(root)
            config.voice_dir.mkdir(parents=True)
            (config.voice_dir / "af_heart.pt").write_bytes(b"heart")
            (config.voice_dir / "af_bella.pt").write_bytes(b"bella")
            adapter = KokoroAdapter(
                config,
                session_factory=lambda runtime_config: KokoroSession(
                    runtime_config,
                    FakePipeline(),
                ),
            )
            captured_requests: list[SynthesisRequest] = []
            adapter_generate = adapter.generate

            def capture(request: SynthesisRequest):
                captured_requests.append(request)
                return adapter_generate(request)

            adapter.generate = capture  # type: ignore[method-assign]
            registry = EngineRegistry()
            registry.register(adapter)

            async def exercise_gradio_lifecycle() -> None:
                app = build_app(registry)
                state = SessionState(app)
                voice_components = [
                    component
                    for component in app.get_config_file()["components"]
                    if component["type"] == "dropdown"
                    and component["props"].get("label") == "Voice"
                ]
                self.assertEqual(len(voice_components), 1)
                self.assertTrue(voice_components[0]["props"]["interactive"])
                self.assertFalse(
                    any(function.api_name == "apply" for function in app.fns.values())
                )
                runtime_panels = [
                    component
                    for component in app.get_config_file()["components"]
                    if component["props"].get("key") == "runtime-status-panel"
                ]
                self.assertEqual(len(runtime_panels), 1)
                runtime_panel_id = runtime_panels[0]["id"]
                runtime_accordions = [
                    component
                    for component in app.get_config_file()["components"]
                    if component["props"].get("key")
                    == "runtime-status-accordion"
                ]
                self.assertEqual(len(runtime_accordions), 1)
                self.assertFalse(runtime_accordions[0]["props"]["open"])
                bottom_layout = app.get_config_file()["layout"]["children"][-1]
                self.assertEqual(bottom_layout["id"], runtime_accordions[0]["id"])
                self.assertEqual(
                    bottom_layout["children"],
                    [{"id": runtime_panel_id}],
                )

                begin_listener = self._function_with_key(
                    app,
                    "kokoro:kokoro-82m:begin-generation",
                )
                self.assertIn(
                    runtime_panel_id,
                    [component._id for component in begin_listener.outputs],
                )
                await self._process(app, begin_listener, [None], state)
                generate_listener = self._function_with_key(
                    app,
                    "kokoro:kokoro-82m:generate-listener",
                )
                self.assertIn(
                    runtime_panel_id,
                    [component._id for component in generate_listener.outputs],
                )
                generation_result = await self._process(
                    app,
                    generate_listener,
                    ["Hello from Bella", "af_bella", 1.0, None],
                    state,
                )
                self.assertIn("Runtime loaded | yes", generation_result["data"][3])
                self.assertIn("Real-time factor", generation_result["data"][3])

            asyncio.run(exercise_gradio_lifecycle())

            self.assertEqual(captured_requests[0].inputs["voice"], "af_bella")
            result_directory = next(config.output_dir.glob("*/*"))
            metadata = json.loads(
                (result_directory / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["voice_id"], "af_bella")

    @staticmethod
    async def _process(app, function, inputs, state):
        return await app.process_api(
            function,
            inputs,
            state=state,
            session_hash="gradio-lifecycle-test",
        )

    @staticmethod
    def _function_with_key(app, key):
        return next(
            function
            for function in app.fns.values()
            if function.key == key
        )


if __name__ == "__main__":
    unittest.main()
