"""Built-in TTS engine adapters."""

from tts_workbench.engines.base import (
    EngineAdapter,
    EngineDependencyError,
    EngineOutputError,
    EngineRuntimeError,
    InferenceNotImplementedError,
)
from tts_workbench.engines.f5 import F5Adapter
from tts_workbench.engines.kokoro import KokoroAdapter
from tts_workbench.engines.qwen import QwenAdapter
from tts_workbench.engines.registry import EngineRegistry

__all__ = [
    "EngineAdapter",
    "EngineDependencyError",
    "EngineOutputError",
    "EngineRegistry",
    "EngineRuntimeError",
    "F5Adapter",
    "InferenceNotImplementedError",
    "KokoroAdapter",
    "QwenAdapter",
]
