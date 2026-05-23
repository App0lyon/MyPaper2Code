from __future__ import annotations

from pathlib import Path
from typing import Any

from mypaper2code.core.io import read_json, write_json
from mypaper2code.providers.base import (
    NVIDIA_DEFAULT_MODEL,
    OLLAMA_DEFAULT_MODEL,
    SUPPORTED_PROVIDERS,
)

DEFAULT_CONFIG = {
    "provider": "nvidia",
    "model": NVIDIA_DEFAULT_MODEL,
    "ollama_base_url": "http://localhost:11434",
    "nvidia_base_url": "https://integrate.api.nvidia.com/v1",
    "nvidia_api_key_env": "NVIDIA_API_KEY",
}


def config_path() -> Path:
    return Path.home() / ".mypaper2code" / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return DEFAULT_CONFIG.copy()
    return _normalize_config(DEFAULT_CONFIG | read_json(path))


def set_config(key: str, value: str) -> dict[str, Any]:
    if key not in DEFAULT_CONFIG:
        raise KeyError(f"Unsupported config key: {key}")
    if key == "provider" and value.lower() not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider `{value}`. Supported providers are: "
            f"{', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )
    config = load_config()
    config[key] = value.lower() if key == "provider" else value
    config = _normalize_config(config)
    write_json(config_path(), config)
    return config


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    provider = str(config.get("provider") or "nvidia").lower()
    if provider not in SUPPORTED_PROVIDERS:
        provider = "nvidia"
    config["provider"] = provider
    if not config.get("model") or config.get("model") == "stub":
        config["model"] = OLLAMA_DEFAULT_MODEL if provider == "ollama" else NVIDIA_DEFAULT_MODEL
    return config
