from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mypaper2code.cli import app
from mypaper2code.core.io import write_json
from mypaper2code.core.models import (
    ImplementationRequirements,
    PaperChunk,
    PaperMetadata,
    WorkspaceMetadata,
)
from mypaper2code.services.generation import CodeWriter
from mypaper2code.services.planning import ImplementationPlanner
from mypaper2code.services.search.hybrid import HybridRetriever
from mypaper2code.services.validation import ExperimentRunner


def make_workspace(path: Path) -> Path:
    (path / "paper").mkdir(parents=True)
    (path / "analysis").mkdir()
    (path / "generated_code").mkdir()
    (path / "runs").mkdir()
    metadata = WorkspaceMetadata(
        workspace_id="paper_20260101-000000",
        root=str(path),
        paper=PaperMetadata(
            paper_id="paper",
            title="Paper",
            source_path=str(path / "paper" / "original.pdf"),
        ),
    )
    write_json(path / "metadata.json", metadata.model_dump(mode="json"))
    chunks = [
        PaperChunk(
            chunk_id="paper-00000",
            paper_id="paper",
            section="method",
            page=1,
            text="The method uses a compact neural architecture with cross entropy loss.",
        )
    ]
    write_json(path / "paper" / "chunks.json", [chunk.model_dump() for chunk in chunks])
    HybridRetriever.build(path, chunks)
    return path


def test_plan_and_generation(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path / "workspace")

    ImplementationPlanner().create_plan(workspace, requirements=ImplementationRequirements())
    generated = CodeWriter().generate(workspace)

    assert (workspace / "analysis" / "implementation_plan.md").exists()
    assert (generated / "configs" / "default.yaml").exists()
    assert (generated / "scripts" / "train.py").exists()


def test_validation_runs_on_generated_project(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path / "workspace")
    ImplementationPlanner().create_plan(workspace, requirements=ImplementationRequirements())
    CodeWriter().generate(workspace)

    results = ExperimentRunner().validate(workspace)

    assert results
    assert all(result.passed for result in results)


def test_cli_config_get() -> None:
    result = CliRunner().invoke(app, ["config", "get"])

    assert result.exit_code == 0
    assert "provider" in result.output


def test_cli_plan_generate_ask(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path / "workspace")
    runner = CliRunner()

    plan_result = runner.invoke(app, ["plan", "--workspace", str(workspace)])
    generate_result = runner.invoke(app, ["implement", "--workspace", str(workspace)])
    ask_result = runner.invoke(app, ["ask", "cross entropy loss", "--workspace", str(workspace)])

    assert plan_result.exit_code == 0, plan_result.output
    assert generate_result.exit_code == 0, generate_result.output
    assert ask_result.exit_code == 0, ask_result.output


def test_requirements_implement_ask_code_and_report(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path / "workspace")
    runner = CliRunner()

    requirements_result = runner.invoke(
        app,
        ["requirements", "set", "dataset", "mnist", "--workspace", str(workspace)],
    )
    plan_result = runner.invoke(app, ["plan", "--workspace", str(workspace)])
    implement_result = runner.invoke(app, ["implement", "--workspace", str(workspace)])
    ask_code_result = runner.invoke(
        app,
        ["ask-code", "where is the loss implemented", "--workspace", str(workspace)],
    )
    report_result = runner.invoke(app, ["report", "--workspace", str(workspace)])

    assert requirements_result.exit_code == 0, requirements_result.output
    assert plan_result.exit_code == 0, plan_result.output
    assert implement_result.exit_code == 0, implement_result.output
    assert ask_code_result.exit_code == 0, ask_code_result.output
    assert report_result.exit_code == 0, report_result.output
    assert (workspace / "analysis" / "implementation_trace.json").exists()
    assert (workspace / "analysis" / "fidelity_report.md").exists()
