from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from mypaper2code.core.io import read_json, write_json
from mypaper2code.core.models import ValidationResult, ValidationSuiteResult


class ExperimentRunner:
    def validate(self, workspace: Path, level: str = "smoke") -> list[ValidationResult]:
        return self.validate_suite(workspace, level=level).results

    def validate_suite(self, workspace: Path, level: str = "smoke") -> ValidationSuiteResult:
        if level not in {"smoke", "contract", "repro"}:
            raise ValueError("Validation level must be one of: smoke, contract, repro.")
        generated = workspace / "generated"
        if not generated.exists():
            generated = workspace / "generated_code"
        runs = workspace / "validation"
        if not runs.exists():
            runs = workspace / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        if not generated.exists():
            raise FileNotFoundError(f"Generated code not found: {generated}")

        checks = [
            ("imports", [sys.executable, "-m", "compileall", "-q", str(generated)]),
            (
                "pytest_smoke",
                [sys.executable, "-m", "pytest", str(generated / "tests" / "test_smoke.py")],
            ),
        ]
        if level in {"contract", "repro"}:
            checks.append(
                (
                    "pytest_contract",
                    [sys.executable, "-m", "pytest", str(generated / "tests" / "test_contract.py")],
                )
            )
        if level == "repro":
            checks.append(
                (
                    "toy_repro",
                    [
                        sys.executable,
                        "scripts/run_experiment.py",
                        "--config",
                        "configs/default.yaml",
                    ],
                )
            )
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
        suite = ValidationSuiteResult(
            level=level,
            results=results,
            fidelity_score=_fidelity_score(workspace, results, level),
            reasons=_fidelity_reasons(workspace, results, level),
        )
        write_json(runs / "validation_suite.json", suite.model_dump(mode="json"))
        return suite


def _fidelity_score(workspace: Path, results: list[ValidationResult], level: str) -> str:
    if any(not result.passed for result in results):
        return "blocked"
    understanding_path = workspace / "understanding" / "research_understanding.json"
    trace_path = workspace / "trace" / "implementation_trace.json"
    if not understanding_path.exists() or not trace_path.exists():
        return "low"
    understanding = read_json(understanding_path)
    trace = read_json(trace_path)
    evidence_count = len(understanding.get("evidence", []))
    trace_count = len(trace.get("entries", []))
    has_protocol = bool(understanding.get("protocols"))
    has_metrics = bool(understanding.get("metrics"))
    has_method = bool(understanding.get("algorithms") or understanding.get("contributions"))
    if (
        level == "repro"
        and evidence_count
        and trace_count
        and has_protocol
        and has_metrics
        and has_method
    ):
        return "reproducible"
    if evidence_count and trace_count and has_metrics and has_method:
        return "high"
    if evidence_count and trace_count:
        return "medium"
    return "low"


def _fidelity_reasons(workspace: Path, results: list[ValidationResult], level: str) -> list[str]:
    reasons = [f"Validation level: {level}."]
    failed = [result.name for result in results if not result.passed]
    if failed:
        reasons.append(f"Failed checks: {', '.join(failed)}.")
    understanding_path = workspace / "understanding" / "research_understanding.json"
    if not understanding_path.exists():
        reasons.append("No rich research understanding artifact was found.")
        return reasons
    understanding = read_json(understanding_path)
    for key in ("evidence", "algorithms", "contributions", "metrics", "protocols"):
        if not understanding.get(key):
            reasons.append(f"Missing or empty understanding field: {key}.")
    return reasons
