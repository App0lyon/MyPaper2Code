from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from mypaper2code.core.models import ValidationResult


class ExperimentRunner:
    def validate(self, workspace: Path) -> list[ValidationResult]:
        generated = workspace / "generated_code"
        runs = workspace / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        if not generated.exists():
            raise FileNotFoundError(f"Generated code not found: {generated}")

        checks = [
            ("imports", [sys.executable, "-m", "compileall", "-q", str(generated)]),
            ("pytest", [sys.executable, "-m", "pytest", str(generated / "tests")]),
        ]
        ruff = shutil.which("ruff")
        if ruff:
            checks.insert(1, ("ruff", [ruff, "check", "."]))
        results: list[ValidationResult] = []
        for name, command in checks:
            log_path = runs / f"{name}.log"
            completed = subprocess.run(
                command,
                cwd=generated,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_path.write_text(completed.stdout, encoding="utf-8")
            results.append(
                ValidationResult(
                    name=name,
                    command=" ".join(command),
                    returncode=completed.returncode,
                    log_path=str(log_path),
                )
            )
        return results
