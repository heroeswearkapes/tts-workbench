"""Generic schema-driven Gradio application."""

import logging
from typing import Any, Callable

from tts_workbench.bootstrap import create_default_registry
from tts_workbench.device_diagnostics import (
    DeviceAvailability,
    probe_device_availability,
)
from tts_workbench.engines.base import EngineRuntimeError
from tts_workbench.engines.registry import EngineRegistry
from tts_workbench.ui.controller import (
    EngineForm,
    engine_choices,
    generate_from_form,
    get_engine_form,
    model_choices,
)
from tts_workbench.ui.form_renderer import render_input, render_setting
from tts_workbench.runtime_manager import CleanupState, RuntimeManager
from tts_workbench.runtime_status import (
    RuntimePhase,
    RuntimeStatus,
    format_runtime_status,
    status_after_error,
    status_after_generation,
    status_after_runtime_load,
    status_before_generation,
    status_for_selection,
    status_with_runtime_lifecycle,
    status_with_warning,
)


_LOGGER = logging.getLogger(__name__)


class _FormEventBindings:
    """Components needed to bind one schema-generated form after layout creation."""

    def __init__(
        self,
        form: EngineForm,
        generate: Any,
        audio: Any,
        generation_status: Any,
        input_components: list[Any],
        setting_components: list[Any],
        ordered_settings: list[Any],
    ) -> None:
        self.form = form
        self.generate = generate
        self.audio = audio
        self.generation_status = generation_status
        self.input_components = input_components
        self.setting_components = setting_components
        self.ordered_settings = ordered_settings


def _load_gradio() -> Any:
    try:
        import gradio as gr
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Gradio is not installed. Install the project dependencies before launching "
            "TTS Workbench."
        ) from error
    return gr


def _available_forms(registry: EngineRegistry) -> tuple[EngineForm, ...]:
    """Snapshot every registered engine/model schema for stable UI components."""
    return tuple(
        get_engine_form(registry, descriptor.id, model.id)
        for descriptor in registry.descriptors()
        for model in registry.get(descriptor.id).list_models()
    )


def _form_key(form: EngineForm) -> tuple[str, str]:
    return form.engine.id, form.model.id


def _make_submit_handler(
    registry: EngineRegistry,
    runtime_manager: RuntimeManager,
    form: EngineForm,
    ordered_settings: list[Any],
    availability: DeviceAvailability,
) -> Callable[..., tuple[str | None, str, RuntimeStatus, str]]:
    """Bind one persistent form's components to generic request translation."""
    input_count = len(form.inputs)

    def submit(
        text_value: str,
        *component_values: Any,
    ) -> tuple[str | None, str, RuntimeStatus, str]:
        submitted_status = component_values[-1]
        form_component_values = component_values[:-1]
        runtime_status = _runtime_status_for_form(
            registry,
            runtime_manager,
            form,
            availability,
            submitted_status,
        )
        try:
            submitted_input_values = form_component_values[:input_count]
            submitted_setting_values = form_component_values[input_count:]
            values_by_setting_key = {
                setting.key: value
                for setting, value in zip(
                    ordered_settings,
                    submitted_setting_values,
                    strict=True,
                )
            }
            settings_in_schema_order = [
                values_by_setting_key[setting.key] for setting in form.settings
            ]
            _LOGGER.debug(
                "UI submit engine=%s model=%s component_inputs=%r "
                "component_settings=%r",
                form.engine.id,
                form.model.id,
                submitted_input_values,
                submitted_setting_values,
            )
            result = generate_from_form(
                registry,
                form,
                text_value,
                submitted_input_values,
                settings_in_schema_order,
                runtime_manager,
            )
        except (ValueError, KeyError) as error:
            runtime_status = _status_after_dispatch(
                runtime_manager,
                availability,
                runtime_status,
            )
            runtime_status = status_after_error(runtime_status, error)
            return (
                None,
                f"**Validation error:** {error}",
                runtime_status,
                format_runtime_status(runtime_status),
            )
        except EngineRuntimeError as error:
            runtime_status = _status_after_dispatch(
                runtime_manager,
                availability,
                runtime_status,
            )
            runtime_status = status_after_error(runtime_status, error)
            return (
                None,
                f"**Generation error:** {error}",
                runtime_status,
                format_runtime_status(runtime_status),
            )
        selected_form = _selected_form(registry, runtime_manager, form)
        runtime_status = _runtime_status_for_form(
            registry,
            runtime_manager,
            selected_form,
            availability,
            runtime_status,
        )
        if _form_key(selected_form) == (result.engine_id, result.model_id):
            runtime_status = status_after_generation(runtime_status, result)
        if result.audio_path is None:
            runtime_status = status_with_warning(
                runtime_status,
                "Generation completed without an audio artifact.",
            )
            return (
                None,
                "Generation completed, but no audio artifact was returned.",
                runtime_status,
                format_runtime_status(runtime_status),
            )
        timing = (
            f" in {result.generation_seconds:.2f}s"
            if result.generation_seconds is not None
            else ""
        )
        duration = (
            f"{result.duration_seconds:.2f}s of audio"
            if result.duration_seconds is not None
            else "audio"
        )
        return (
            str(result.audio_path),
            f"Generated {duration}{timing}.",
            runtime_status,
            format_runtime_status(runtime_status),
        )

    return submit


def _make_begin_handler(
    registry: EngineRegistry,
    runtime_manager: RuntimeManager,
    form: EngineForm,
    availability: DeviceAvailability,
) -> Callable[[RuntimeStatus | None], tuple[RuntimeStatus, str]]:
    def begin(status: RuntimeStatus | None) -> tuple[RuntimeStatus, str]:
        current = _runtime_status_for_form(
            registry,
            runtime_manager,
            form,
            availability,
            status,
        )
        if current.phase is not RuntimePhase.UNLOADING:
            current = status_before_generation(current)
        return current, format_runtime_status(current)

    return begin


def _runtime_status_for_form(
    registry: EngineRegistry,
    runtime_manager: RuntimeManager,
    form: EngineForm,
    availability: DeviceAvailability,
    current: Any = None,
) -> RuntimeStatus:
    if (
        isinstance(current, RuntimeStatus)
        and current.active_engine_id == form.engine.id
        and current.active_model_id == form.model.id
    ):
        status = current
    else:
        status = status_for_selection(
            form.engine,
            form.model,
            loaded_runtimes=runtime_manager.loaded_runtimes(),
            availability=availability,
        )
    return _status_after_dispatch(runtime_manager, availability, status)


def _status_after_dispatch(
    runtime_manager: RuntimeManager,
    availability: DeviceAvailability,
    current: RuntimeStatus,
) -> RuntimeStatus:
    status = status_after_runtime_load(
        current,
        runtime_manager.loaded_runtimes(),
        availability=availability,
    )
    snapshot = runtime_manager.snapshot()
    active = snapshot.active
    cleanup_error = (
        snapshot.cleanup_detail
        if snapshot.cleanup_state is CleanupState.FAILED
        else None
    )
    return status_with_runtime_lifecycle(
        status,
        phase=snapshot.phase,
        active_runtime_engine_id=active.engine_id if active else None,
        active_runtime_model_id=active.model_id if active else None,
        device_cleanup_status=snapshot.cleanup_status,
        cleanup_error=cleanup_error,
    )


def _selected_form(
    registry: EngineRegistry,
    runtime_manager: RuntimeManager,
    fallback: EngineForm,
) -> EngineForm:
    selected = runtime_manager.snapshot().selected
    if selected is None or (selected.engine_id, selected.model_id) == _form_key(fallback):
        return fallback
    return get_engine_form(registry, selected.engine_id, selected.model_id)


def _log_component_value(
    engine_id: str,
    model_id: str,
    group: str,
    key: str,
    value: Any,
) -> None:
    """Provide opt-in diagnostics without inserting engine-specific UI behavior."""
    _LOGGER.debug(
        "UI input event engine=%s model=%s group=%s field=%s value=%r",
        engine_id,
        model_id,
        group,
        key,
        value,
    )


def build_app(
    registry: EngineRegistry | None = None,
    runtime_manager: RuntimeManager | None = None,
) -> Any:
    """Build the Gradio Blocks app from registered engine schemas."""
    gr = _load_gradio()
    registry = registry or create_default_registry()
    runtime_manager = runtime_manager or RuntimeManager(registry)
    choices = engine_choices(registry)
    if not choices:
        raise RuntimeError("cannot build the UI without a registered engine")

    forms = _available_forms(registry)
    default_engine_id = choices[0][1]
    default_adapter = registry.get(default_engine_id)
    default_model_id = default_adapter.descriptor.default_model_id
    default_form_key = (default_engine_id, default_model_id)
    availability = probe_device_availability()
    default_form = next(form for form in forms if _form_key(form) == default_form_key)
    runtime_manager.select(default_engine_id, default_model_id)
    initial_runtime_status = _runtime_status_for_form(
        registry,
        runtime_manager,
        default_form,
        availability,
    )

    with gr.Blocks(title="TTS Workbench") as app:
        gr.Markdown("# TTS Workbench\nA model-agnostic local text-to-speech interface.")
        with gr.Row():
            engine = gr.Dropdown(
                label="Engine",
                choices=choices,
                value=default_engine_id,
                interactive=True,
            )
            model = gr.Dropdown(
                label="Model",
                choices=model_choices(registry, default_engine_id),
                value=default_model_id,
                interactive=True,
            )

        runtime_state = gr.State(initial_runtime_status)
        status_refresh = gr.Timer(value=1.0, active=True)

        text = gr.Textbox(
            label="Text",
            lines=8,
            placeholder="Enter text to synthesize...",
            interactive=True,
        )

        panels: list[Any] = []
        panel_keys: list[tuple[str, str]] = []
        form_event_bindings: list[_FormEventBindings] = []

        for form in forms:
            form_key = _form_key(form)
            with gr.Column(
                visible=form_key == default_form_key,
                key=f"{form.engine.id}:{form.model.id}:panel",
            ) as panel:
                capabilities = ", ".join(
                    capability.value.replace("_", " ")
                    for capability in sorted(
                        form.engine.capabilities,
                        key=lambda item: item.value,
                    )
                )
                gr.Markdown(
                    f"**{form.engine.description}**  \nCapabilities: {capabilities}"
                )

                input_components = [
                    render_input(gr, form, field) for field in form.inputs
                ]
                basic_settings = [
                    setting for setting in form.settings if not setting.advanced
                ]
                advanced_settings = [
                    setting for setting in form.settings if setting.advanced
                ]
                setting_components = [
                    render_setting(gr, form, setting) for setting in basic_settings
                ]
                if advanced_settings:
                    with gr.Accordion("Advanced settings", open=False):
                        setting_components.extend(
                            render_setting(gr, form, setting)
                            for setting in advanced_settings
                        )

                generate = gr.Button("Generate", variant="primary")
                audio = gr.Audio(label="Generated audio")
                generation_status = gr.Markdown()

                ordered_settings = basic_settings + advanced_settings
                for field, component in zip(
                    form.inputs,
                    input_components,
                    strict=True,
                ):
                    _LOGGER.debug(
                        "UI rendered input engine=%s model=%s field=%s "
                        "component_id=%s component_key=%s default=%r",
                        form.engine.id,
                        form.model.id,
                        field.key,
                        getattr(component, "_id", None),
                        getattr(component, "key", None),
                        field.default,
                    )
                    component.input(
                        lambda value, form=form, field=field: _log_component_value(
                            form.engine.id,
                            form.model.id,
                            "inputs",
                            field.key,
                            value,
                        ),
                        inputs=component,
                        outputs=None,
                        queue=False,
                    )
                for setting, component in zip(
                    ordered_settings,
                    setting_components,
                    strict=True,
                ):
                    component.input(
                        lambda value, form=form, setting=setting: _log_component_value(
                            form.engine.id,
                            form.model.id,
                            "settings",
                            setting.key,
                            value,
                        ),
                        inputs=component,
                        outputs=None,
                        queue=False,
                    )

                form_event_bindings.append(
                    _FormEventBindings(
                        form,
                        generate,
                        audio,
                        generation_status,
                        input_components,
                        setting_components,
                        ordered_settings,
                    )
                )

            panels.append(panel)
            panel_keys.append(form_key)

        with gr.Accordion(
            "Runtime status",
            open=False,
            key="runtime-status-accordion",
        ):
            runtime_panel = gr.Markdown(
                format_runtime_status(initial_runtime_status),
                key="runtime-status-panel",
            )

        for bindings in form_event_bindings:
            form = bindings.form
            generation_started = bindings.generate.click(
                _make_begin_handler(registry, runtime_manager, form, availability),
                inputs=runtime_state,
                outputs=[runtime_state, runtime_panel],
                queue=False,
                key=f"{form.engine.id}:{form.model.id}:begin-generation",
            )
            generation_started.then(
                _make_submit_handler(
                    registry,
                    runtime_manager,
                    form,
                    bindings.ordered_settings,
                    availability,
                ),
                inputs=[
                    text,
                    *bindings.input_components,
                    *bindings.setting_components,
                    runtime_state,
                ],
                outputs=[
                    bindings.audio,
                    bindings.generation_status,
                    runtime_state,
                    runtime_panel,
                ],
                concurrency_limit=1,
                concurrency_id="tts-generation",
                key=f"{form.engine.id}:{form.model.id}:generate-listener",
            )

        def select_engine(engine_id: str) -> list[Any]:
            adapter = registry.get(engine_id)
            selected_model_id = adapter.descriptor.default_model_id
            selected_key = (engine_id, selected_model_id)
            form = get_engine_form(registry, engine_id, selected_model_id)
            runtime_manager.select(engine_id, selected_model_id)
            selected_status = _runtime_status_for_form(
                registry,
                runtime_manager,
                form,
                availability,
            )
            return [
                gr.Dropdown(
                    choices=model_choices(registry, engine_id),
                    value=selected_model_id,
                ),
                *(
                    gr.Column(visible=panel_key == selected_key)
                    for panel_key in panel_keys
                ),
                selected_status,
                format_runtime_status(selected_status),
            ]

        def select_model(engine_id: str, model_id: str | None) -> list[Any]:
            form = get_engine_form(registry, engine_id, model_id)
            selected_key = _form_key(form)
            runtime_manager.select(*selected_key)
            selected_status = _runtime_status_for_form(
                registry,
                runtime_manager,
                form,
                availability,
            )
            return [
                *(
                    gr.Column(visible=panel_key == selected_key)
                    for panel_key in panel_keys
                ),
                selected_status,
                format_runtime_status(selected_status),
            ]

        def refresh_runtime_status(
            current: RuntimeStatus | None,
        ) -> tuple[RuntimeStatus, str]:
            selected_form = _selected_form(
                registry,
                runtime_manager,
                default_form,
            )
            selected_status = _runtime_status_for_form(
                registry,
                runtime_manager,
                selected_form,
                availability,
                current,
            )
            return selected_status, format_runtime_status(selected_status)

        engine.change(
            select_engine,
            inputs=engine,
            outputs=[model, *panels, runtime_state, runtime_panel],
            queue=False,
            key="engine-selection-listener",
        )
        model.change(
            select_model,
            inputs=[engine, model],
            outputs=[*panels, runtime_state, runtime_panel],
            queue=False,
            key="model-selection-listener",
        )
        status_refresh.tick(
            refresh_runtime_status,
            inputs=runtime_state,
            outputs=[runtime_state, runtime_panel],
            queue=False,
            key="runtime-status-refresh",
        )

    return app
