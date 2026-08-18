# Standalone Qwen3-TTS runtime probe

This probe predates the integrated Qwen adapter and remains available as a
standalone diagnostic. It checks package imports, official repository metadata,
and CPU/XPU/CUDA visibility without constructing a model or generating audio.

For normal TTS Workbench use, install the `qwen` extra or use the development
Docker image. The main application now includes lazy loading and generation for
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`; this probe is not a separate worker and
is not required by the application.

## What it checks

- Python and package versions;
- imports for Torch, Torchaudio, Transformers, Hugging Face Hub, Accelerate, and
  `qwen_tts`;
- the `Qwen3TTSModel` import;
- visible CPU, Intel XPU, and CUDA devices; and
- `config.json` and repository metadata for one or all five official Qwen3-TTS
  12 Hz model descriptors.

The default online run fetches metadata and `config.json` for the 0.6B
CustomVoice repository. `--all-models` checks all five descriptors. Neither mode
downloads checkpoint weights.

## Setup

Use Python 3.11 in a disposable virtual environment. Install matching,
backend-specific PyTorch and Torchaudio wheels before the probe requirement.

For Intel XPU:

```bash
python3.11 -m venv .venv-qwen-probe
. .venv-qwen-probe/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.11.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r experiments/qwen/requirements.txt
python experiments/qwen/probe.py --all-models
```

The validated dependency set resolves to:

```text
qwen-tts==0.1.1
transformers==4.57.3
accelerate==1.12.0
torch==2.11.0+xpu
torchaudio==2.11.0+xpu
```

The host should also provide FFmpeg, libsndfile, and SoX because importing the
Qwen audio stack can inspect those dependencies. The probe itself does not
decode audio.

## Offline device/import check

Use `--skip-metadata` when network access is unavailable:

```bash
python experiments/qwen/probe.py --skip-metadata
```

This reports package imports and devices only. An import or metadata error gives
the process a nonzero exit status and is included in the JSON report.

## Scope and interpretation

A successful probe demonstrates packaging and device visibility. It does not
prove that checkpoint loading, operator placement, memory consumption, or audio
generation works on the detected accelerator. Those behaviors belong to the
integrated runtime and Atlas hardware validation.

Current integration and hardware notes are maintained in
[`docs/qwen-runtime-research.md`](../../docs/qwen-runtime-research.md).
