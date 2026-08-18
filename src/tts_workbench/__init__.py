"""Model-agnostic local text-to-speech workbench."""


def create_default_registry():
    """Create the built-in engine registry without eager package imports."""
    from tts_workbench.bootstrap import create_default_registry as create_registry

    return create_registry()

__all__ = ["create_default_registry"]
__version__ = "0.5.0"
