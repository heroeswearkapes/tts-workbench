"""Pure UI view-model and request translation functions."""

from dataclasses import dataclass
import logging
from typing import Any, Sequence

from tts_workbench.domain import (
    EngineDescriptor,
    InputFieldSpec,
    ModelDescriptor,
    SettingSpec,
    SynthesisRequest,
    SynthesisResult,
    VoiceDescriptor,
)
from tts_workbench.engines.registry import EngineRegistry
from tts_workbench.runtime_manager import RuntimeManager


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EngineForm:
    """Everything the generic renderer needs for an engine/model selection."""

    engine: EngineDescriptor
    model: ModelDescriptor
    voices: tuple[VoiceDescriptor, ...]
    inputs: tuple[InputFieldSpec, ...]
    settings: tuple[SettingSpec, ...]


def engine_choices(registry: EngineRegistry) -> list[tuple[str, str]]:
    return [(descriptor.name, descriptor.id) for descriptor in registry.descriptors()]


def model_choices(registry: EngineRegistry, engine_id: str) -> list[tuple[str, str]]:
    adapter = registry.get(engine_id)
    return [(model.name, model.id) for model in adapter.list_models()]


def get_engine_form(
    registry: EngineRegistry,
    engine_id: str,
    model_id: str | None = None,
) -> EngineForm:
    adapter = registry.get(engine_id)
    available_model_ids = {model.id for model in adapter.list_models()}
    selected_model_id = (
        model_id
        if model_id is not None and model_id in available_model_ids
        else adapter.descriptor.default_model_id
    )
    model = adapter.require_model(selected_model_id)
    return EngineForm(
        engine=adapter.descriptor,
        model=model,
        voices=adapter.list_voices(selected_model_id),
        inputs=adapter.input_schema(selected_model_id),
        settings=adapter.setting_schema(selected_model_id),
    )


def build_request(
    form: EngineForm,
    text: str,
    input_values: Sequence[Any],
    setting_values: Sequence[Any],
) -> SynthesisRequest:
    if len(input_values) != len(form.inputs):
        raise ValueError("input values do not match the engine input schema")
    if len(setting_values) != len(form.settings):
        raise ValueError("setting values do not match the engine setting schema")
    inputs = {field.key: value for field, value in zip(form.inputs, input_values, strict=True)}
    settings = {
        setting.key: value
        for setting, value in zip(form.settings, setting_values, strict=True)
    }
    missing = [
        field.label
        for field in form.inputs
        if field.required and _is_missing(inputs[field.key])
    ]
    if missing:
        raise ValueError(f"required inputs are missing: {', '.join(missing)}")
    return SynthesisRequest(
        engine_id=form.engine.id,
        model_id=form.model.id,
        text=text,
        inputs=inputs,
        settings=settings,
    )


def generate_from_form(
    registry: EngineRegistry,
    form: EngineForm,
    text: str,
    input_values: Sequence[Any],
    setting_values: Sequence[Any],
    runtime_manager: RuntimeManager | None = None,
) -> SynthesisResult:
    """Translate the rendered form values and dispatch the normalized request."""
    _LOGGER.debug(
        "Controller received engine=%s model=%s input_values=%r "
        "setting_values=%r",
        form.engine.id,
        form.model.id,
        list(input_values),
        list(setting_values),
    )
    request = build_request(form, text, input_values, setting_values)
    _LOGGER.debug(
        "SynthesisRequest engine=%s model=%s resolved_voice_id=%r "
        "inputs=%r settings=%r",
        request.engine_id,
        request.model_id,
        request.inputs.get("voice"),
        dict(request.inputs),
        dict(request.settings),
    )
    if runtime_manager is not None:
        return runtime_manager.generate(request)
    return registry.get(form.engine.id).generate(request)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return not value
    return False
