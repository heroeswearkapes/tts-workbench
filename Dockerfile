FROM python:3.11-bookworm AS python-runtime

FROM ubuntu:26.04

ARG PYTORCH_VERSION=2.11.0
ARG DEBIAN_FRONTEND=noninteractive

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TTS_WORKBENCH_ROOT=/ \
    TTS_WORKBENCH_LOG_LEVEL=INFO \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/models/.cache/huggingface \
    TORCH_HOME=/models/.cache/torch \
    XDG_CACHE_HOME=/models/.cache

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        espeak-ng \
        ffmpeg \
        intel-opencl-icd \
        libze-intel-gpu1 \
        libze1 \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-runtime /usr/local /usr/local

WORKDIR /app

RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        "torch==${PYTORCH_VERSION}" \
        "torchaudio==${PYTORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/xpu

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN python -m pip install --no-cache-dir --editable ".[kokoro,f5,qwen]" \
        "qwen-tts==0.1.1" \
        "transformers==4.57.3" \
        "accelerate==1.12.0" \
    && python -m pip check \
    && python -c "import torch; from qwen_tts import Qwen3TTSModel; assert torch.__version__.startswith('${PYTORCH_VERSION}+xpu'); print('Qwen runtime available:', Qwen3TTSModel.__name__, 'torch', torch.__version__)" \
    && dpkg-query --show intel-opencl-icd libze1 libze-intel-gpu1 sox \
    && tts-workbench-devices --json

COPY entrypoint.sh /usr/local/bin/tts-workbench-entrypoint

RUN chmod 0755 /usr/local/bin/tts-workbench-entrypoint \
    && mkdir -p /models /voices /input /output /config

VOLUME ["/models", "/voices", "/input", "/output", "/config"]

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3).read(1)" || exit 1

ENTRYPOINT ["/usr/local/bin/tts-workbench-entrypoint"]
CMD ["tts-workbench"]
