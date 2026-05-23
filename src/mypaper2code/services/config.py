from __future__ import annotations

from pathlib import Path
from typing import Any

from mypaper2code.core.io import read_json, write_json

DEFAULT_CONFIG = {"provider": "stub", "model": "stub"}


def config_path() -> Path:
    return Path.home() / ".mypaper2code" / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG | read_json(path)


def set_config(key: str, value: str) -> dict[str, Any]:
    if key not in DEFAULT_CONFIG:
        raise KeyError(f"Unsupported config key: {key}")
    config = load_config()
    config[key] = value
    write_json(config_path(), config)
    return config
