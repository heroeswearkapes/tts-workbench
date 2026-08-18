"""Map declarative domain fields to Gradio components."""

from typing import Any

from tts_workbench.domain import ChoiceSource, FieldKind, InputFieldSpec, SettingSpec
from tts_workbench.ui.controller import EngineForm


def _base_kwargs(label: str, help_text: str, key: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"label": label, "key": key}
    if help_text:
        kwargs["info"] = help_text
    return kwargs


def render_input(gr: Any, form: EngineForm, field: InputFieldSpec) -> Any:
    """Create one Gradio input from an engine-declared input field."""
    key = f"{form.engine.id}:{form.model.id}:input:{field.key}"
    kwargs = _base_kwargs(field.label, field.help_text, key)

    if field.kind is FieldKind.DROPDOWN:
        if field.choice_source is ChoiceSource.VOICES:
            choices = [(voice.name, voice.id) for voice in form.voices]
        else:
            choices = list(field.choices)
        return gr.Dropdown(
            choices=choices,
            value=field.default,
            interactive=True,
            preserved_by_key="value",
            **kwargs,
        )
    if field.kind is FieldKind.TEXT:
        return gr.Textbox(value=field.default, interactive=True, **kwargs)
    if field.kind is FieldKind.MULTILINE_TEXT:
        return gr.Textbox(value=field.default, lines=4, interactive=True, **kwargs)
    if field.kind is FieldKind.AUDIO:
        kwargs.pop("info", None)
        return gr.Audio(
            type="filepath",
            sources=["upload", "microphone"],
            interactive=True,
            **kwargs,
        )
    if field.kind is FieldKind.NUMBER:
        return gr.Number(value=field.default, interactive=True, **kwargs)
    if field.kind is FieldKind.CHECKBOX:
        return gr.Checkbox(value=bool(field.default), interactive=True, **kwargs)
    raise ValueError(f"unsupported input field kind: {field.kind.value}")


def render_setting(gr: Any, form: EngineForm, setting: SettingSpec) -> Any:
    """Create one Gradio component from an engine-declared setting."""
    key = f"{form.engine.id}:{form.model.id}:setting:{setting.key}"
    kwargs = _base_kwargs(setting.label, setting.help_text, key)

    if setting.kind is FieldKind.SLIDER:
        return gr.Slider(
            minimum=setting.minimum,
            maximum=setting.maximum,
            step=setting.step,
            value=setting.default,
            interactive=True,
            **kwargs,
        )
    if setting.kind is FieldKind.NUMBER:
        return gr.Number(
            minimum=setting.minimum,
            maximum=setting.maximum,
            step=setting.step,
            value=setting.default,
            interactive=True,
            **kwargs,
        )
    if setting.kind is FieldKind.DROPDOWN:
        return gr.Dropdown(
            choices=list(setting.choices),
            value=setting.default,
            interactive=True,
            preserved_by_key="value",
            **kwargs,
        )
    if setting.kind is FieldKind.CHECKBOX:
        return gr.Checkbox(value=bool(setting.default), interactive=True, **kwargs)
    raise ValueError(f"unsupported setting field kind: {setting.kind.value}")
