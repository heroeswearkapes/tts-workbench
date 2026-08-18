# Qwen3-TTS runtime integration and validation notes

This document records the Qwen3-TTS investigation that preceded integration and
the resulting runtime decisions. Qwen is now registered directly in TTS
Workbench; it is no longer blocked on a separate worker or dependency-isolation
decision.

## Current status

The active runtime model is:

```text
Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

The implementation provides:

- lazy Qwen imports and checkpoint loading;
- one cached session per model, device, and precision;
- nine preset speakers;
- language selection and advanced sampling controls;
- CPU FP32 fallback;
- Intel XPU BF16 or FP32 selection through the shared diagnostics layer;
- normalized WAV and JSON metadata output; and
- generic `RuntimeStatus` reporting.

The application still starts when Qwen dependencies are absent. A missing or
incompatible dependency produces an engine-scoped runtime error only when Qwen
generation is requested.

## Declared model family

The adapter exposes five official repository IDs so the generic UI and engine
registry can describe the complete current family:

| Model descriptor | Schema exposed | Inference status |
| --- | --- | --- |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Preset voice, language, style instruction, sampling | Active and integrated |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Reference audio/text, embedding-only option, sampling | Descriptor only |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Preset voice, language, style instruction, sampling | Descriptor only |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Reference audio/text, embedding-only option, sampling | Descriptor only |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Language, voice description, sampling | Descriptor only |

The adapter rejects generation for every descriptor-only model with an explicit
not-implemented error. Their schemas are foundations for later runtime work,
not claims of current inference support.

## Validated dependency set

| Component | Validated value | Notes |
| --- | --- | --- |
| Python | 3.11 | Copied from the official Python image into Ubuntu 26.04 |
| `qwen-tts` | 0.1.1 | Provides `qwen_tts.Qwen3TTSModel` |
| Transformers | 4.57.3 | Required exact runtime version |
| Accelerate | 1.12.0 | Required exact runtime version |
| PyTorch | 2.11.0+xpu | Main development image build |
| Torchaudio | 2.11.0+xpu | Must match PyTorch |
| Native audio | FFmpeg, libsndfile, SoX | Included in the development image |
| Attention | PyTorch SDPA | Flash Attention is intentionally absent |

The earlier investigation found that Qwen's Transformers constraint displaced a
newer Transformers/Hugging Face Hub combination. The final Docker build pins the
known-good versions explicitly, installs all Kokoro/F5/Qwen extras together,
and completes `pip check` without broken requirements. A separate Qwen worker
was therefore not required for the current combination.

Dependency availability is checked from distribution metadata without importing
Qwen or Torch. Actual Qwen and Hugging Face imports occur only at session load.

## Atlas Intel XPU validation

Atlas uses an Intel Arc Pro B70. Validation established that:

- the Ubuntu container can see the selected `/dev/dri` render node;
- the installed Intel OpenCL and Level Zero libraries allow
  `torch.xpu.is_available()` to succeed on Atlas;
- `qwen_tts` and `Qwen3TTSModel` import successfully;
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` loads successfully with the validated
  PyTorch/Transformers combination; and
- the application reports the Qwen session through the common runtime-status
  path.

This was a model-load validation. Do not describe the other Qwen descriptors as
hardware-validated inference runtimes. CPU-only dockerbuilder checks can verify
imports, dependency pins, app startup, fake generation persistence, and CPU
fallback, but they cannot substitute for XPU hardware testing.

The container requires these Intel userspace packages in addition to a working
host driver and device passthrough:

```text
intel-opencl-icd
libze1
libze-intel-gpu1
```

## Runtime flow

1. The registry exposes the Qwen engine and all model descriptors without
   importing `qwen_tts`.
2. The schema-driven UI renders the selected model's generic fields.
3. The controller builds a normalized `SynthesisRequest`.
4. The adapter validates that the selected model is the active 0.6B CustomVoice
   runtime and resolves device and precision.
5. CPU resolves to FP32; automatic XPU resolves to BF16 unless FP32 is requested.
6. The runtime imports Qwen dependencies and downloads an incomplete or missing
   checkpoint into `models/qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/`.
7. `Qwen3TTSModel.from_pretrained()` loads with explicit device mapping, dtype,
   and SDPA attention.
8. Generation calls the published CustomVoice API and persists `audio.wav` and
   `metadata.json` under `output/<date>/<job-id>/`.
9. The shared status lifecycle records the loaded device, precision, generation
   duration, audio duration, real-time factor, warnings, and errors.

## Current limitations

### Style instructions

The generic UI collects a style instruction and the runtime forwards it as
`instruct`. The published 0.6B CustomVoice implementation currently clears that
argument internally. The field therefore does not provide reliable acoustic
style control for the active checkpoint. Persisted metadata records
`style_instruction_supported: false`.

The larger 1.7B CustomVoice descriptor retains the same schema because that
family is intended to support instruction control, but its inference path is not
integrated.

### Model scope

Base-model cloning and VoiceDesign are schema-only. They still require:

- real reference-audio normalization and clone-prompt integration for Base;
- reference-transcript and speaker-embedding behavior decisions;
- VoiceDesign generation integration;
- output and error-path validation; and
- Atlas memory and operator-coverage tests.

### Streaming and batching

The engine descriptor can advertise upstream family capabilities, but the
active workbench generation path is serialized and non-streaming. Batching,
reusable clone prompts, cancellation, and streamed audio should be added only
after each corresponding model runtime is validated.

### CUDA

Official examples commonly target CUDA, but this workbench image is built for
Intel XPU. The domain and runtime-status contracts can represent CUDA; the Qwen
adapter currently accepts only CPU and XPU. A CUDA image and Qwen runtime path
must be validated separately. Flash Attention should not be installed in the
current XPU image.

## Model order and memory planning

Repository size is not peak inference memory. Checkpoint files include speech
tokenizer assets, while runtime memory also includes activations, caches, and
backend workspaces. The metadata probe measured approximately 2.50 GB for the
0.6B repositories and 4.52-4.54 GB for the 1.7B repositories.

Use this order for future integration work:

| Order | Model | Repository size | Planning BF16 VRAM | Purpose |
| --- | --- | --- | --- | --- |
| 1 | 0.6B CustomVoice | about 2.50 GB | 3-5 GiB | Current preset-voice runtime |
| 2 | 0.6B Base | about 2.52 GB | 4-6 GiB | Smallest cloning runtime |
| 3 | 1.7B CustomVoice | about 4.52 GB | 6-9 GiB | Larger preset/style runtime |
| 4 | 1.7B Base | about 4.54 GB | 7-10 GiB | Larger cloning runtime |
| 5 | 1.7B VoiceDesign | about 4.52 GB | 7-10 GiB | Free-form voice design |

These are engineering planning ranges, not official guarantees. Measure actual
allocated and reserved memory on Atlas before allowing concurrent generation or
keeping multiple Qwen checkpoints resident. FP32 can require substantially more
memory than BF16.

## Standalone probe

The probe under `experiments/qwen/` remains useful for import, metadata, and
device checks outside the main application. It does not replace the integrated
runtime or load/generate validation. See
[`experiments/qwen/README.md`](../experiments/qwen/README.md).

## Primary references

- [Official Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS)
- [Published dependency metadata](https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml)
- [Official model wrapper and inference modes](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py)
- [Published qwen-tts package](https://pypi.org/project/qwen-tts/)
