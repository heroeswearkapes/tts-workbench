from threading import Event, Thread
import unittest

from tts_workbench.device_diagnostics import LoadedRuntime
from tts_workbench.domain import (
    EngineDescriptor,
    ModelDescriptor,
    SynthesisRequest,
    SynthesisResult,
)
from tts_workbench.engines.base import EngineAdapter
from tts_workbench.engines.registry import EngineRegistry
from tts_workbench.runtime_manager import CleanupState, RuntimeKey, RuntimeManager
from tts_workbench.runtime_status import RuntimePhase


class FakeSession:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class ManagedAdapter(EngineAdapter):
    def __init__(
        self,
        engine_id: str,
        model_id: str,
        *,
        device: str = "cpu",
        generation_started: Event | None = None,
        generation_release: Event | None = None,
    ) -> None:
        self._model = ModelDescriptor(id=model_id, name=model_id)
        self._descriptor = EngineDescriptor(
            id=engine_id,
            name=engine_id.title(),
            description=f"Fake {engine_id} adapter",
            capabilities=frozenset(),
            default_model_id=model_id,
        )
        self.device = device
        self.generation_started = generation_started
        self.generation_release = generation_release
        self.load_calls = 0
        self.generate_calls = 0
        self.unload_calls = 0
        self.session: FakeSession | None = None

    @property
    def descriptor(self) -> EngineDescriptor:
        return self._descriptor

    def list_models(self) -> tuple[ModelDescriptor, ...]:
        return (self._model,)

    def list_voices(self, model_id):
        self.require_model(model_id)
        return ()

    def input_schema(self, model_id):
        self.require_model(model_id)
        return ()

    def setting_schema(self, model_id):
        self.require_model(model_id)
        return ()

    def generate(self, request: SynthesisRequest) -> SynthesisResult:
        self.require_model(request.model_id)
        self.generate_calls += 1
        if self.session is None:
            self.load_calls += 1
            self.session = FakeSession()
            self._notify_runtime_loaded(
                LoadedRuntime(
                    self.descriptor.id,
                    self._model.id,
                    self.device,
                    precision="fp32",
                )
            )
        if self.generation_started is not None:
            self.generation_started.set()
        if self.generation_release is not None:
            self.generation_release.wait(timeout=5)
        return SynthesisResult(
            engine_id=self.descriptor.id,
            model_id=self._model.id,
            sample_rate=24_000,
            duration_seconds=1.0,
            generation_seconds=0.1,
        )

    def runtime_status(self):
        if self.session is None:
            return ()
        return (
            LoadedRuntime(
                self.descriptor.id,
                self._model.id,
                self.device,
                precision="fp32",
            ),
        )

    def unload_runtime(self, model_id=None):
        if model_id is not None:
            self.require_model(model_id)
        self.unload_calls += 1
        session = self.session
        self.session = None
        return (session,) if session is not None else ()


class FakeCacheBackend:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    def __init__(self) -> None:
        self.xpu = FakeCacheBackend()
        self.cuda = FakeCacheBackend()


def request(adapter: ManagedAdapter) -> SynthesisRequest:
    return SynthesisRequest(
        engine_id=adapter.descriptor.id,
        model_id=adapter.descriptor.default_model_id,
        text="Lifecycle test",
    )


class RuntimeManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kokoro = ManagedAdapter("kokoro", "kokoro-model")
        self.f5 = ManagedAdapter("f5", "f5-model", device="xpu:0")
        self.qwen = ManagedAdapter("qwen", "qwen-model", device="xpu:0")
        self.registry = EngineRegistry()
        for adapter in (self.kokoro, self.f5, self.qwen):
            self.registry.register(adapter)
        self.fake_torch = FakeTorch()
        self.gc_calls = 0
        self.cleaned: list[tuple[RuntimeKey, tuple[object, ...]]] = []
        self.manager = RuntimeManager(
            self.registry,
            cleanup_callbacks=(
                lambda key, sessions: self.cleaned.append((key, sessions)),
            ),
            cleanup_scheduler=lambda task: task(),
            garbage_collector=self._collect,
            torch_loader=lambda: self.fake_torch,
        )

    def _collect(self) -> int:
        self.gc_calls += 1
        return 7

    def test_metadata_selection_does_not_load_and_generation_reuses_runtime(self) -> None:
        for adapter in (self.kokoro, self.f5, self.qwen):
            self.manager.select(
                adapter.descriptor.id,
                adapter.descriptor.default_model_id,
            )
        self.assertEqual(
            [adapter.load_calls for adapter in (self.kokoro, self.f5, self.qwen)],
            [0, 0, 0],
        )

        self.manager.select("kokoro", "kokoro-model")
        self.manager.generate(request(self.kokoro))
        self.manager.generate(request(self.kokoro))

        self.assertEqual(self.kokoro.load_calls, 1)
        self.assertEqual(self.kokoro.generate_calls, 2)
        self.assertEqual(self.manager.snapshot().phase, RuntimePhase.READY)

    def test_switching_kokoro_to_f5_releases_runtime_and_caches(self) -> None:
        self.manager.select("kokoro", "kokoro-model")
        self.manager.generate(request(self.kokoro))
        session = self.kokoro.session

        self.manager.select("f5", "f5-model")
        self.assertTrue(self.manager.wait_for_idle(timeout=1))

        self.assertEqual(self.kokoro.unload_calls, 1)
        self.assertIsNone(self.kokoro.session)
        self.assertEqual(session.close_calls, 1)
        self.assertEqual(self.gc_calls, 1)
        self.assertEqual(self.fake_torch.xpu.empty_cache_calls, 1)
        self.assertEqual(self.fake_torch.cuda.empty_cache_calls, 1)
        self.assertEqual(self.manager.snapshot().cleanup_state, CleanupState.COMPLETED)

    def test_switching_f5_to_qwen_releases_runtime_and_runs_callbacks(self) -> None:
        self.manager.select("f5", "f5-model")
        self.manager.generate(request(self.f5))
        session = self.f5.session

        self.manager.select("qwen", "qwen-model")
        self.assertTrue(self.manager.wait_for_idle(timeout=1))

        self.assertEqual(self.f5.unload_calls, 1)
        self.assertIsNone(self.f5.session)
        self.assertEqual(session.close_calls, 1)
        self.assertEqual(self.cleaned[0][0], RuntimeKey("f5", "f5-model"))
        self.assertEqual(self.cleaned[0][1], (session,))
        self.assertEqual(self.manager.snapshot().phase, RuntimePhase.UNLOADED)

    def test_selection_during_generation_returns_before_cleanup(self) -> None:
        generation_started = Event()
        generation_release = Event()
        kokoro = ManagedAdapter(
            "kokoro",
            "kokoro-model",
            generation_started=generation_started,
            generation_release=generation_release,
        )
        f5 = ManagedAdapter("f5", "f5-model", device="xpu:0")
        registry = EngineRegistry()
        registry.register(kokoro)
        registry.register(f5)
        manager = RuntimeManager(
            registry,
            garbage_collector=lambda: 0,
            torch_loader=lambda: self.fake_torch,
        )
        manager.select("kokoro", "kokoro-model")
        generation = Thread(target=manager.generate, args=(request(kokoro),))
        generation.start()
        self.assertTrue(generation_started.wait(timeout=1))

        switched = manager.select("f5", "f5-model")

        self.assertEqual(switched.phase, RuntimePhase.UNLOADING)
        self.assertEqual(switched.cleanup_state, CleanupState.PENDING)
        self.assertTrue(generation.is_alive())
        self.assertEqual(f5.load_calls, 0)

        generation_release.set()
        generation.join(timeout=2)
        self.assertFalse(generation.is_alive())
        self.assertTrue(manager.wait_for_idle(timeout=2))
        self.assertEqual(kokoro.unload_calls, 1)
        self.assertEqual(manager.snapshot().selected, RuntimeKey("f5", "f5-model"))
        self.assertIsNone(manager.snapshot().active)


if __name__ == "__main__":
    unittest.main()
