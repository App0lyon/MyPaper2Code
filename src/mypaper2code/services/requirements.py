from __future__ import annotations

from pathlib import Path
from typing import Any

from mypaper2code.core.io import read_yaml, write_yaml
from mypaper2code.core.models import ImplementationRequirements


def requirements_path(workspace: Path) -> Path:
    return workspace / "analysis" / "requirements.yaml"


def load_requirements(workspace: Path) -> ImplementationRequirements:
    path = requirements_path(workspace)
    if not path.exists():
        requirements = ImplementationRequirements()
        write_yaml(path, requirements.model_dump(mode="json"))
        return requirements
    data = read_yaml(path) or {}
    return ImplementationRequirements.model_validate(data)


def save_requirements(workspace: Path, requirements: ImplementationRequirements) -> None:
    write_yaml(requirements_path(workspace), requirements.model_dump(mode="json"))


def set_requirement(workspace: Path, key: str, value: str) -> ImplementationRequirements:
    requirements = load_requirements(workspace)
    data: dict[str, Any] = requirements.model_dump()
    if key not in data:
        raise KeyError(f"Unsupported requirement key: {key}")
    current = data[key]
    if isinstance(current, bool):
        data[key] = value.lower() in {"1", "true", "yes", "y", "on"}
    else:
        data[key] = value
    updated = ImplementationRequirements.model_validate(data)
    save_requirements(workspace, updated)
    return updated
