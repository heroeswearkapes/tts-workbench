"""Application composition for built-in engine adapters."""

from tts_workbench.engines.f5 import F5Adapter
from tts_workbench.engines.kokoro import KokoroAdapter
from tts_workbench.engines.qwen import QwenAdapter
from tts_workbench.engines.registry import EngineRegistry


def create_default_registry() -> EngineRegistry:
    """Create a registry containing the Phase 1 built-in adapters."""
    registry = EngineRegistry()
    registry.register(KokoroAdapter())
    registry.register(F5Adapter())
    registry.register(QwenAdapter())
    return registry
