#!/usr/bin/env python3
"""Inspect a Qwen3-TTS environment without loading weights or generating audio."""

import argparse
import importlib
from importlib import metadata as importlib_metadata
import json
import platform
from pathlib import Path
import sys
from typing import Any, Sequence


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_IDS = (
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)
PACKAGE_IMPORTS = (
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("transformers", "transformers"),
    ("huggingface-hub", "huggingface_hub"),
    ("accelerate", "accelerate"),
    ("qwen-tts", "qwen_tts"),
)


def inspect_imports() -> tuple[dict[str, Any], Any | None]:
    packages: dict[str, Any] = {}
    torch_module = None
    for distribution, module_name in PACKAGE_IMPORTS:
        record: dict[str, Any] = {"version": _package_version(distribution)}
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            record.update(
                {
                    "imported": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        else:
            record["imported"] = True
            if module_name == "torch":
                torch_module = module
        packages[distribution] = record

    try:
        from qwen_tts import Qwen3TTSModel
    except Exception as error:
        packages["Qwen3TTSModel"] = {
            "imported": False,
            "error": f"{type(error).__name__}: {error}",
        }
    else:
        packages["Qwen3TTSModel"] = {
            "imported": True,
            "class": Qwen3TTSModel.__name__,
        }
    return packages, torch_module


def inspect_devices(torch_module: Any | None) -> dict[str, Any]:
    devices: dict[str, Any] = {
        "cpu": {
            "available": True,
            "name": platform.processor() or platform.machine() or "unknown",
        }
    }
    if torch_module is None:
        devices["xpu"] = {"available": False, "devices": []}
        devices["cuda"] = {"available": False, "devices": []}
        return devices

    for backend_name in ("xpu", "cuda"):
        backend = getattr(torch_module, backend_name, None)
        devices[backend_name] = _inspect_accelerator(backend)
    return devices


def inspect_model_metadata(model_id: str) -> dict[str, Any]:
    """Fetch repository metadata and config.json, never checkpoint weights."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(model_id, files_metadata=True)
    config_path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="config.json",
        )
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    siblings = tuple(info.siblings or ())
    safetensors = [
        sibling
        for sibling in siblings
        if sibling.rfilename.endswith(".safetensors")
    ]
    weight_bytes = sum((sibling.size or 0) for sibling in safetensors)
    repository_bytes = sum((sibling.size or 0) for sibling in siblings)
    talker = config.get("talker_config", {})
    return {
        "model_id": model_id,
        "revision": info.sha,
        "file_count": len(siblings),
        "repository_bytes": repository_bytes,
        "safetensors_bytes": weight_bytes,
        "safetensors_files": [sibling.rfilename for sibling in safetensors],
        "architecture": (config.get("architectures") or [None])[0],
        "model_type": config.get("model_type"),
        "tts_model_type": config.get("tts_model_type"),
        "tts_model_size": config.get("tts_model_size"),
        "tokenizer_type": config.get("tokenizer_type"),
        "transformers_version": config.get("transformers_version"),
        "speaker_count": len(talker.get("spk_id", {})),
        "language_ids": sorted(talker.get("codec_language_id", {})),
    }


def build_report(
    model_ids: Sequence[str],
    *,
    skip_metadata: bool = False,
) -> tuple[dict[str, Any], bool]:
    packages, torch_module = inspect_imports()
    report: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": packages,
        "devices": inspect_devices(torch_module),
        "models": [],
    }
    success = all(record.get("imported") for record in packages.values())
    if not skip_metadata:
        for model_id in model_ids:
            try:
                model = inspect_model_metadata(model_id)
            except Exception as error:
                model = {
                    "model_id": model_id,
                    "error": f"{type(error).__name__}: {error}",
                }
                success = False
            report["models"].append(model)
    return report, success


def _inspect_accelerator(backend: Any | None) -> dict[str, Any]:
    if backend is None:
        return {"available": False, "devices": [], "reason": "not exposed by torch"}
    try:
        available = bool(backend.is_available())
    except Exception as error:
        return {
            "available": False,
            "devices": [],
            "error": f"{type(error).__name__}: {error}",
        }
    if not available:
        return {"available": False, "devices": []}

    found = []
    try:
        count = int(backend.device_count())
    except Exception as error:
        return {
            "available": True,
            "devices": [],
            "error": f"{type(error).__name__}: {error}",
        }
    for index in range(count):
        try:
            name = str(backend.get_device_name(index))
        except Exception:
            name = "unknown"
        found.append({"index": index, "name": name})
    return {"available": True, "devices": found}


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Qwen3-TTS imports, checkpoint metadata, and device visibility "
            "without loading model weights"
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Hugging Face model id to inspect; may be repeated",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Inspect metadata for all currently released 12Hz checkpoints",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Run package imports and device detection without network access",
    )
    args = parser.parse_args(argv)
    model_ids = MODEL_IDS if args.all_models else tuple(args.models or (DEFAULT_MODEL,))
    report, success = build_report(model_ids, skip_metadata=args.skip_metadata)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
