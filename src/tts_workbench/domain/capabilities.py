"""Semantic capabilities advertised by TTS engines."""

from enum import StrEnum


class Capability(StrEnum):
    """Stable capability identifiers used outside engine implementations."""

    PRESET_VOICES = "preset_voices"
    REFERENCE_AUDIO = "reference_audio"
    REFERENCE_TEXT = "reference_text"
    VOICE_CLONING = "voice_cloning"
    SPEED_CONTROL = "speed_control"
    FAST_NARRATION = "fast_narration"
    MULTILINGUAL = "multilingual"
    STREAMING = "streaming"
    SEED_CONTROL = "seed_control"
    VOICE_DESIGN = "voice_design"
    STYLE_CONTROL = "style_control"
    LANGUAGE_SELECTION = "language_selection"
