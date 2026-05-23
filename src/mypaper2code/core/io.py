from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_model(path: Path, model: BaseModel) -> None:
    write_json(path, model.model_dump(mode="json"))


def load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(read_json(path))


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
