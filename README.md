# TTS Workbench

TTS Workbench is a self-hosted, multi-engine text-to-speech platform with one
local Gradio interface for narration, voice cloning, and emerging TTS models.
It exists to make engines with very different controls and runtime requirements
usable through a consistent workflow without sending text, reference recordings,
or generated audio to a hosted application.

The current application supports fast preset-voice narration with Kokoro,
reference-audio voice cloning with F5-TTS, and an integrated Qwen3-TTS
CustomVoice runtime. Models load only when first used and generated audio and
metadata remain in local storage.

[Quick start](#quick-start) · [Engines](#engine-support) ·
[Hardware](#hardware-support) · [Timezone](#timezone-and-output-folders) ·
[Storage](#storage-layout) ·
[Architecture](#development-architecture) · [Troubleshooting](#troubleshooting)

## Feature overview

- **One self-hosted UI:** choose an engine and model, then use the controls that
  adapter declares.
- **Multiple synthesis workflows:** preset voices, reference-audio cloning,
  optional transcription, language selection, and model-specific settings.
- **Local artifacts:** model caches, voice assets, source material, generated
  WAV files, and metadata stay in mounted storage.
- **CPU and validated Intel XPU paths:** CPU fallback works without a GPU;
  F5-TTS synthesis and Qwen model loading have been exercised on Intel Arc Pro
  B70 hardware.
- **Runtime visibility:** a collapsible status panel reports model load state,
  device, precision, cleanup progress, timing, real-time factor, warnings, and
  errors.
- **Extensible by design:** adapters and schemas add engines without embedding
  engine-specific behavior in the Gradio frontend.

## Quick start

You need Docker Engine with the Compose plugin, an x86-64 CPU, enough disk space
for the selected models, and network access for first-use model downloads. A GPU
is optional. The default Compose configuration is CPU-compatible and does not
require `/dev/dri`.

```bash
git clone https://github.com/heroeswearkapes/tts-workbench.git
cd tts-workbench
test -f .env || cp .env.example .env
```

For this first-time setup, edit `.env` before starting the container. Set
`TTS_DATA_ROOT` to the parent directory for your models, voices, input, and
output, and set `CONFIG_PATH` to your configuration directory. Set `TZ` to your
preferred timezone if you want local-date output folders; Compose defaults to
`Etc/UTC`. For example:

```dotenv
TZ=America/New_York
```

Pull the published image and start the service:

```bash
docker compose pull
docker compose up -d
```

Open <http://localhost:7860> (or the port selected with `TTS_PORT`), select an
engine and model, enter text, and
generate audio. The first request for an engine may take longer while its model
and voice assets download and the runtime loads.

The standard Compose configuration mounts the host paths selected in `.env`:

| Host path | Container path | Contains |
| --- | --- | --- |
| `${TTS_DATA_ROOT}/models` | `/models` | Model weights and framework caches |
| `${TTS_DATA_ROOT}/voices` | `/voices` | Preset voice assets |
| `${TTS_DATA_ROOT}/input` | `/input` | Persistent reference material |
| `${TTS_DATA_ROOT}/output` | `/output` | Generated audio and metadata |
| `${CONFIG_PATH}` | `/config` | Persistent application configuration |

Use `docker compose logs -f` to follow logs and `docker compose down` to stop
and remove the service container. Data in the configured host directories is
retained. Intel XPU users can use the same published image with the
[device-passthrough command](#intel-xpu-docker-run).

## Engine support

| Engine | Maturity | Active runtime | Devices |
| --- | --- | --- | --- |
| Kokoro | **Fully supported** | Kokoro 82M | CPU (FP32) |
| F5-TTS | **Fully supported** | F5-TTS v1 Base | CPU or Intel XPU (FP32) |
| Qwen3-TTS | **Integrated** | Qwen3-TTS 12Hz 0.6B CustomVoice | CPU (FP32) or Intel XPU (BF16/FP32) |

“Fully supported” means the listed workflow has an end-to-end runtime and test
coverage. “Integrated” means the listed active model has a runtime in the main
application, while additional models may be descriptors only. Descriptor-only
Qwen models are visible foundations for future work and cannot generate audio.

### Kokoro

Kokoro is the fast narration engine. It provides:

- a catalog of 54 preset voices;
- voice selection and speech-speed control;
- voice descriptors with stable IDs, display names, language, gender,
  category, and accent where known;
- American and British English, Japanese, Mandarin Chinese, Spanish, French,
  Hindi, Italian, and Brazilian Portuguese voice groups;
- serialized CPU inference using FP32; and
- lazy download of the selected voice asset.

Kokoro model files are stored under `models/kokoro/kokoro-82m/`, while voice
assets are stored independently under `voices/kokoro/`.

### F5-TTS

F5-TTS is the reference-audio voice-cloning engine. It supports:

- a required reference recording;
- an optional exact reference transcript;
- automatic reference transcription when the transcript is omitted;
- speech-speed control;
- CPU synthesis; and
- Intel XPU synthesis.

Reference uploads include:

- WAV
- M4A
- MP3
- WebM
- Ogg

FFmpeg performs the initial decode and converts the source into mono, 24 kHz,
16-bit PCM WAV. SoundFile is the final decoder into the numeric data passed to
F5. FFmpeg-supported formats such as FLAC also work, although the formats above
cover the normal upload and browser-recording paths.

If reference text is supplied, it is passed directly to F5 and Whisper is not
loaded. If it is blank, the explicit `WhisperProcessor` and
`WhisperForConditionalGeneration` path transcribes normalized 16 kHz audio on
CPU. F5 synthesis remains on its selected CPU or XPU device while transcription
stays on CPU.

### Qwen3-TTS

Qwen3-TTS is integrated into the main adapter registry. The active inference
runtime is:

```text
Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

It supports nine preset speakers, language selection, sampling controls, lazy
checkpoint download, CPU fallback, Intel XPU selection, and runtime status
reporting.

The adapter also exposes schemas and model descriptors for:

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`

Those four descriptors let the generic UI represent voice cloning, larger
CustomVoice, and VoiceDesign inputs, but their inference paths are not yet
integrated. Only the 0.6B CustomVoice model should be treated as active.

The UI records and forwards a style instruction for the active model. The
published 0.6B CustomVoice implementation currently clears `instruct`
internally, so the instruction does not provide reliable acoustic style control.
Output metadata explicitly records that limitation. The field remains available
for schema compatibility and future model support.

Qwen imports are lazy. Missing Qwen packages do not prevent the application,
Kokoro, or F5-TTS from starting.

## Example workflow

1. Select **Kokoro** for fast preset narration, **F5-TTS** to clone a reference
   recording, or the active **Qwen3-TTS** model for a multilingual preset voice.
2. The UI displays only the fields declared by that engine/model schema.
3. Enter synthesis text and provide the selected workflow's voice or reference
   inputs.
4. Generate. The adapter validates the request, lazily loads or reuses its
   runtime session, and writes a normalized WAV plus JSON metadata.
5. Play or download the audio and expand **Runtime status** to inspect the
   device, precision, load state, timing, real-time factor, or any error.

## Hardware support

| Backend/hardware | Current support | Validation boundary |
| --- | --- | --- |
| x86-64 CPU | Supported | Kokoro, F5-TTS, and active Qwen runtime paths; performance depends on the host |
| Intel Arc Pro B70 | Validated where noted | F5-TTS XPU synthesis; Qwen 0.6B CustomVoice import/model load |
| Other Intel XPU devices | Not hardware-validated | May work with compatible drivers and runtimes, but no compatibility claim is made |
| NVIDIA CUDA | Architecture only | No CUDA runtime or CUDA Docker image is currently provided |

### CPU

CPU is always available as the fallback backend. Kokoro is CPU-only. F5-TTS and
the active Qwen runtime also support CPU, although their larger models may be
slower than interactive workloads require.

### Intel XPU

The development image uses `torch==2.11.0+xpu` and matching Torchaudio wheels.
F5-TTS and Qwen select XPU automatically when `torch.xpu.is_available()` is
true; otherwise they fall back to CPU. Kokoro remains on CPU.

Atlas validation with an Intel Arc Pro B70 confirmed:

- Intel device passthrough into the Ubuntu container;
- PyTorch XPU detection with the Intel OpenCL and Level Zero runtimes;
- F5-TTS model loading and synthesis on XPU; and
- Qwen3-TTS 0.6B CustomVoice package import and model loading on XPU.

The CPU-only dockerbuilder cannot validate XPU hardware, so XPU generation
checks remain an Atlas responsibility.

### CUDA architecture

The domain contracts and generic runtime-status model can represent a `cuda`
backend without UI changes. The current adapters and development image do not
ship a CUDA runtime: Kokoro accepts CPU, and F5/Qwen currently accept CPU or
XPU. CUDA engine enablement requires a compatible PyTorch image and explicit
adapter runtime validation; it is an architectural extension point, not a
currently validated execution mode.

## Installation

### Docker Compose

Docker is the recommended way to run all current engine dependencies together.
The image uses Ubuntu 26.04 with Python 3.11 copied from the official Python
runtime stage. It includes Gradio, Kokoro, F5-TTS, Qwen3-TTS, FFmpeg,
libsndfile, SoX, eSpeak NG, the PyTorch XPU build, the Intel GPU userspace
runtime, and `tzdata` for named timezone support.

After configuring `.env` as described in [Quick start](#quick-start), start the
CPU-compatible service with the published image:

```bash
docker compose pull
docker compose up -d
```

The standard Compose file uses `${IMAGE_NAME}:${IMAGE_TAG}` from `.env`,
persists all five storage paths, and exposes port `7860` through the configured
host port. It runs the source packaged in the image. The same image starts
normally without `/dev/dri` and falls back to CPU.

For a local image build, run `docker build -t tts-workbench:dev .` from the
repository root. The separate `docker-compose.dev.yml` is intended for source
development and mounts `./src` read-only into the container. Its storage
variables differ from the standard Compose file, so review its volume mappings
before using it with an existing `.env`.

Stop it with:

```bash
docker compose down
```

### Unraid

The Unraid template is maintained in
[`unraid-community-apps`](https://github.com/heroeswearkapes/unraid-community-apps/blob/main/templates/TTS-Workbench.xml)
and uses `heroeswearkapes/tts-workbench:latest`. Configure the WebUI port,
persistent storage paths, and optional Intel GPU device for your server.

Unraid's Docker manager normally supplies the server's timezone to the
container automatically. A custom `TZ` variable is only needed to override
that timezone. See [Timezone and output folders](#timezone-and-output-folders)
for configuration and upgrade guidance.

### Intel XPU Docker run

The host needs a working Intel graphics driver and a usable `/dev/dri` render
device. This standalone example stores data in directories under the current
working directory. For an existing installation, substitute your established
host storage paths in the volume arguments; `docker run` does not read the
Compose `.env` file. Pass DRI into the container:

```bash
docker run --rm \
  --name tts-workbench \
  --env TZ=Etc/UTC \
  --device=/dev/dri:/dev/dri \
  --publish 7860:7860 \
  --volume "$PWD/models:/models" \
  --volume "$PWD/voices:/voices" \
  --volume "$PWD/input:/input" \
  --volume "$PWD/output:/output" \
  --volume "$PWD/config:/config" \
  heroeswearkapes/tts-workbench:latest
```

Replace `Etc/UTC` with your preferred IANA timezone, such as
`America/New_York` or `America/Los_Angeles`, to use local-date output folders.

To pass only the render node used during Atlas validation, replace the device
argument with:

```text
--device=/dev/dri/renderD131:/dev/dri/renderD131
```

Device discovery can be checked without starting Gradio:

```bash
docker run --rm \
  --device=/dev/dri:/dev/dri \
  heroeswearkapes/tts-workbench:latest \
  tts-workbench-devices --device xpu --xpu-index 0 --json
```

### Local CPU development

Use Python 3.11. Native audio packages are required for the full runtime; on a
Debian or Ubuntu development host:

```bash
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1 sox espeak-ng
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --editable .
```

Install only the engines needed in that environment:

```bash
python -m pip install --editable '.[kokoro]'
python -m pip install --editable '.[f5]'
python -m pip install --editable '.[qwen]'
```

Or install all engine extras together:

```bash
python -m pip install --editable '.[kokoro,f5,qwen]'
```

Install the backend-appropriate matching Torch and Torchaudio wheels before the
engine extras. Qwen requires `qwen-tts==0.1.1`, `transformers==4.57.3`, and
`accelerate==1.12.0`; the Docker image pins that validated combination with
Torch/Torchaudio `2.11.0+xpu`. Flash Attention is intentionally not installed.

Launch the application from the repository root:

```bash
tts-workbench-devices
tts-workbench
```

The base install contains Gradio and the engine descriptors. Optional engine
dependencies are imported only when their runtime is requested.

## Docker runtime details

| Item | Current value |
| --- | --- |
| Base OS | Ubuntu 26.04 |
| Python | 3.11 |
| HTTP port | 7860 |
| Torch | 2.11.0 XPU build by default |
| Intel userspace packages | `intel-opencl-icd`, `libze1`, `libze-intel-gpu1` |
| Timezone database | `tzdata`; named timezones are selected through `TZ` |
| Persistent paths | `/models`, `/voices`, `/input`, `/output`, `/config` |
| Healthcheck | HTTP request to `127.0.0.1:7860` |

The application is configured with `TTS_WORKBENCH_ROOT=/` in the image. Model
and framework caches also live below `/models`, so recreating a container does
not require downloading every artifact again.

The entrypoint checks that all storage paths are writable, prints CPU/XPU
diagnostics, and then starts Gradio. Models are downloaded lazily rather than
baked into the image.

## Timezone and output folders

Kokoro, F5-TTS, and Qwen3-TTS organize new audio and metadata under:

```text
/output/YYYY-MM-DD/<job-id>/
```

`YYYY-MM-DD` is the container's local date when the completed job is saved.
For example, at `2026-11-13 16:30` in `America/Los_Angeles`, UTC has already
reached November 14. The local-date output folder is still `2026-11-13`.

The image includes `tzdata`, so IANA timezone names such as
`America/Los_Angeles` and `America/New_York` handle daylight saving time
automatically according to the installed timezone database. Use `Etc/UTC` to
keep UTC dates. This changes date selection for new jobs; existing output
folders and files are not renamed, moved, or deleted.

### Choose a timezone

| Deployment | Default behavior | Set a different timezone |
| --- | --- | --- |
| Unraid Docker manager | Uses the Unraid server's timezone unless overridden | Add or edit a container Variable with Key `TZ` and an IANA timezone as its Value |
| Supplied Docker Compose files | Uses `TZ` from the shell or `.env`, with `Etc/UTC` as the fallback when unset or empty | Set `TZ=America/Los_Angeles` in `.env` and recreate the service |
| Direct `docker run` | Uses the image's local timezone, normally UTC without an override | Add `--env TZ=America/Los_Angeles` when creating the container |

`TZ` sets the container's local timezone generally, including output date
folders. It is not a separate output-only setting. The standard Unraid
template leaves the host timezone inheritance intact instead of forcing a
fixed timezone. Environment variables can be managed in Unraid's container
configuration screen; see the
[Unraid container documentation](https://docs.unraid.net/unraid-os/using-unraid-to/run-docker-containers/managing-and-customizing-containers/).

For Compose, edit the existing `.env` file rather than replacing it during an
upgrade. A `TZ` exported in the shell takes precedence over the value in
`.env`; check that value if the container uses an unexpected timezone.

### Updating an existing installation

The timezone fix requires the updated image. On Unraid, update TTS Workbench
from the Docker tab so the container is recreated with the new image.

For a Compose installation created before the timezone fix, also update the
Compose configuration while preserving your local settings. Alternatively,
add this entry under the TTS Workbench service's existing `environment`
mapping so the `TZ` value in `.env` reaches the container:

```yaml
TZ: ${TZ:-Etc/UTC}
```

After the configuration is updated, pull the image and recreate the standard
Compose service:

```bash
docker compose pull
docker compose up -d
```

For another Compose configuration using the published image, supply the same
`-f` option you normally use for that installation. For source development,
update your source checkout and rebuild the local image with
`docker compose -f docker-compose.dev.yml up -d --build` instead.

Recreating the container applies the selected image and environment while
retaining data in the configured host storage paths. Restarting an existing
container does not install a new image or apply changes made only to `.env`.

With the updated image and a valid named `TZ`, a host `/etc/localtime` bind
mount is not required. If you previously used `TZ=/etc/localtime`, keep that
workaround until you are ready to switch to a named timezone or Unraid's
automatic host setting, then verify the result before removing the old mount.

### Verify the timezone

The standard Compose container is named `tts-workbench`. For Unraid or a
custom deployment, replace that name below with the actual container name:

```bash
docker exec tts-workbench dpkg-query -W tzdata
docker exec tts-workbench printenv TZ
docker exec tts-workbench date
docker exec tts-workbench python -c 'from datetime import datetime; print(datetime.now().astimezone().isoformat())'
```

Check that `tzdata` is installed and the operating system and Python agree on
the intended local time and UTC offset. Generate a short audio sample and
confirm that its new output folder matches the local date. Testing after UTC
midnight but before local midnight makes the change easy to verify for
timezones west of UTC.

## Runtime Status panel

The bottom of the Gradio interface contains one collapsible, engine-independent
Runtime Status panel. It consumes immutable `RuntimeStatus` snapshots and is
updated on engine/model selection, before generation, after a runtime session
loads, during asynchronous unload/device cleanup, after generation, and after
handled failures.

It reports:

- active engine and model;
- active loaded runtime, independently from the selected form;
- lifecycle state (`unloaded`, `loading`, `generating`, `ready`, `unloading`,
  or `error`);
- device backend (`cpu`, `xpu`, or future `cuda`);
- device name when available;
- precision when available;
- whether the runtime is loaded;
- device cleanup state and cache-clear result;
- last generation duration;
- generated audio duration;
- real-time factor; and
- warnings and errors.

Real-time factor is generation time divided by generated audio duration. A value
below `1.0×` means generation was faster than real time.

```text
Active engine: F5-TTS (f5)
Active model: F5-TTS v1 Base (f5-tts-v1-base)
State: ready
Device backend: xpu
Device name: Intel Arc Pro B70
Precision: fp32
Runtime loaded: yes
Last generation: 2.000 s
Generated audio: 5.000 s
Real-time factor: 0.400×
```

## Storage layout

With a repository-local development root, runtime storage looks like:

```text
models/
├── .cache/
├── f5/
│   └── f5-tts-v1-base/
├── kokoro/
│   └── kokoro-82m/
└── qwen/
    └── Qwen3-TTS-12Hz-0.6B-CustomVoice/
voices/
└── kokoro/
    ├── af_heart.pt
    └── ...
input/
output/
└── YYYY-MM-DD/
    └── <job-id>/
        ├── audio.wav
        └── metadata.json
config/
```

| Path | Purpose |
| --- | --- |
| `models/` | Engine weights and persistent framework caches |
| `voices/` | Independently downloaded preset voice assets |
| `input/` | Persistent user-managed reference audio and text |
| `output/` | Generated WAV files and JSON metadata grouped by local date and job ID |
| `config/` | Persistent configuration storage reserved for application settings |

In Docker these map directly to `/models`, `/voices`, `/input`, `/output`, and
`/config`. Gradio is allowed to serve generated files from `/output` and trusted
inputs from `/voices` and `/input`; `/models` is intentionally not exposed.

The dated output folder follows the configured container timezone. See
[Timezone and output folders](#timezone-and-output-folders) for deployment
defaults and configuration examples.

## Troubleshooting

### Output folders use an unexpected date

Confirm that the container uses the updated image, then check its `TZ` value
and local time with the [verification commands](#verify-the-timezone). Compose
defaults to UTC when `TZ` is unset or empty, while normal Unraid Docker manager
installations receive the server's timezone automatically. Check a newly
generated job: older output folders retain their original names.

### XPU is not detected

First verify that the render device exists and is accessible on the host:

```bash
ls -la /dev/dri
```

Then verify that it was passed into the container and that the XPU-enabled Torch
build is installed:

```bash
docker run --rm --device=/dev/dri:/dev/dri heroeswearkapes/tts-workbench:latest \
  python -c 'import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.device_count())'
```

If the device is visible but `torch.xpu.is_available()` is false, check host
driver compatibility, render-node permissions, container device mapping, and
the Intel userspace packages described below. CPU fallback is expected when no
usable XPU exists.

### Intel Level Zero runtime is missing

The development image installs the packages needed by the known-good Atlas
environment:

- `intel-opencl-icd`
- `libze1`
- `libze-intel-gpu1`

Verify them inside the image:

```bash
docker run --rm --entrypoint dpkg-query heroeswearkapes/tts-workbench:latest \
  --show intel-opencl-icd libze1 libze-intel-gpu1
```

An image based on a distribution that lacks `libze-intel-gpu1` can expose a DRI
render node while still failing PyTorch XPU detection. Use the provided Ubuntu
image or install a matching Intel Level Zero runtime.

### Optional engine dependencies are missing

The application intentionally starts without every model package. An engine
dependency error means the selected adapter reached its lazy runtime boundary.
Install the corresponding extra:

```bash
python -m pip install --editable '.[kokoro]'
python -m pip install --editable '.[f5]'
python -m pip install --editable '.[qwen]'
python -m pip check
```

For Qwen, also confirm that the exact validated versions are present:

```bash
python -c 'import importlib.metadata as m; print(m.version("qwen-tts")); print(m.version("transformers")); print(m.version("accelerate"))'
```

Expected values are `0.1.1`, `4.57.3`, and `1.12.0` respectively.

### Reference audio fails to decode

Confirm that FFmpeg is installed and can inspect the upload:

```bash
ffmpeg -hide_banner -i reference.m4a
```

To create a known-good F5 reference manually:

```bash
ffmpeg -i reference.m4a -vn -ac 1 -ar 24000 -c:a pcm_s16le reference.wav
```

Use a short, clean recording with audible speech. F5 reads at most the first 12
seconds. Browser recordings may use WebM or Ogg containers even when the file
extension is unexpected; FFmpeg performs format detection from the content.

### Model download fails

Kokoro, F5-TTS, Qwen3-TTS, and Whisper obtain missing artifacts on first use.
Check:

- internet and DNS access to the model host;
- available disk space under `models/` or `/models`;
- write permissions on `models/`, `voices/`, and their mounted counterparts;
- proxy or Hugging Face authentication requirements; and
- whether an interrupted download left an incomplete local checkpoint.

Container logs and the Runtime Status panel contain the underlying load error:

```bash
docker logs tts-workbench
```

Do not put model storage behind Gradio's allowed paths; model files are runtime
assets and should remain private.

## Development architecture

```mermaid
flowchart LR
    Browser[Browser] --> UI[Schema-driven Gradio UI]
    UI --> Controller[Generic request controller]
    Controller --> Registry[Engine registry]
    Registry --> Kokoro[Kokoro adapter]
    Registry --> F5[F5-TTS adapter]
    Registry --> Qwen[Qwen3-TTS adapter]
    Kokoro --> Sessions[Lazy runtime sessions]
    F5 --> Sessions
    Qwen --> Sessions
    Sessions --> Storage[(Local models, voices, input, output)]
    Sessions --> Status[RuntimeStatus snapshots]
    Status --> UI
```

- **Adapter registry:** each engine implements the common adapter interface and
  registers an engine descriptor, model descriptors, capabilities, voices,
  inputs, settings, generation, and optional runtime-status reporting.
- **Schema-driven UI:** Gradio creates persistent generic components from engine
  schemas. Switching engines or models changes visible forms without adding
  `if engine == ...` logic to the frontend.
- **Normalized requests/results:** the controller translates component values
  into `SynthesisRequest`; every runtime returns `SynthesisResult` with engine,
  model, audio path, sample rate, duration, and generation timing.
- **Runtime status:** immutable snapshots merge selected descriptors, hardware
  availability, loaded adapter sessions, timings, warnings, and errors. The UI
  only formats this shared status contract.
- **Lazy loading and lifecycle management:** engine packages import at session
  creation, and the active session is reused until the selected engine/model
  changes. Selection itself remains descriptor-only; a generic runtime manager
  unloads stale sessions, releases references, runs garbage collection, and
  clears XPU/CUDA allocator caches without blocking the selection callback.
- **Engine isolation:** Kokoro, F5-TTS, and Qwen3-TTS runtime code remains
  separate from the generic Gradio renderer.

Run the test suite from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The standalone Qwen probe and its historical validation notes are documented in
[`experiments/qwen/README.md`](experiments/qwen/README.md) and
[`docs/qwen-runtime-research.md`](docs/qwen-runtime-research.md).

## Acknowledgments

Thank you to [@Acheron4171](https://github.com/Acheron4171) for the detailed
timezone bug report in [issue #1](https://github.com/heroeswearkapes/tts-workbench/issues/1)
and for contributing the fixes in PRs
[#2](https://github.com/heroeswearkapes/tts-workbench/pull/2),
[#3](https://github.com/heroeswearkapes/tts-workbench/pull/3),
[#4](https://github.com/heroeswearkapes/tts-workbench/pull/4),
[#5](https://github.com/heroeswearkapes/tts-workbench/pull/5), and
[#6](https://github.com/heroeswearkapes/tts-workbench/pull/6).

Their contributions added timezone data to the Docker image, configurable
timezone support to the Compose files, and local-date output folders across
Kokoro, F5-TTS, and Qwen3-TTS. Thank you for helping improve TTS Workbench for
everyone who uses it!
