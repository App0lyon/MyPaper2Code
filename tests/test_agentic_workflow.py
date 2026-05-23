from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mypaper2code.cli import app
from mypaper2code.core.io import write_json
from mypaper2code.core.models import (
    EvidenceRef,
    PaperChunk,
    PaperMetadata,
    ResearchAmbiguity,
    ResearchUnderstanding,
    WorkspaceMetadata,
)
from mypaper2code.services.search.hybrid import HybridRetriever


def make_agentic_workspace(path: Path, blocking: bool = False) -> Path:
    for child in (
        "paper",
        "understanding",
        "decisions",
        "plan",
        "generated",
        "validation",
        "trace",
        "analysis",
        "generated_code",
        "runs",
    ):
        (path / child).mkdir(parents=True, exist_ok=True)
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
            text="The method defines an iterative optimization algorithm and reports accuracy.",
        )
    ]
    write_json(path / "paper" / "chunks.json", [chunk.model_dump() for chunk in chunks])
    HybridRetriever.build(path, chunks)
    evidence = [
        EvidenceRef(
            evidence_id="paper-00000",
            kind="chunk",
            page=1,
            section="method",
            text=chunks[0].text,
        )
    ]
    ambiguities = []
    if blocking:
        ambiguities.append(
            ResearchAmbiguity(
                ambiguity_id="amb-001",
                question="Which dataset split should be used?",
                severity="blocking",
            )
        )
    understanding = ResearchUnderstanding(
        paper_type="optimization",
        contributions=["Iterative optimization method"],
        algorithms=["Iterative update rule"],
        datasets=["toy"],
        protocols=["run toy experiment"],
        metrics=["accuracy"],
        expected_artifacts=["script"],
        ambiguities=ambiguities,
        evidence=evidence,
    )
    write_json(
        path / "understanding" / "research_understanding.json",
        understanding.model_dump(mode="json"),
    )
    return path


def test_blocking_ambiguity_prevents_implementation_until_decided(tmp_path: Path) -> None:
    workspace = make_agentic_workspace(tmp_path / "workspace", blocking=True)
    runner = CliRunner()

    plan_result = runner.invoke(app, ["plan", "--workspace", str(workspace)])
    blocked = runner.invoke(app, ["implement", "--workspace", str(workspace)])
    decision = runner.invoke(
        app,
        ["decide", "--workspace", str(workspace), "--id", "amb-001", "--value", "use toy split"],
    )
    implemented = runner.invoke(app, ["implement", "--workspace", str(workspace)])

    assert plan_result.exit_code == 0, plan_result.output
    assert blocked.exit_code == 1, blocked.output
    assert "Blocking decisions" in blocked.output
    assert decision.exit_code == 0, decision.output
    assert implemented.exit_code == 0, implemented.output
    assert (workspace / "generated" / "src" / "method" / "core.py").exists()
    assert (workspace / "trace" / "implementation_trace.json").exists()


def test_contract_validation_writes_suite_result(tmp_path: Path) -> None:
    workspace = make_agentic_workspace(tmp_path / "workspace")
    runner = CliRunner()

    assert runner.invoke(app, ["plan", "--workspace", str(workspace)]).exit_code == 0
    assert runner.invoke(app, ["implement", "--workspace", str(workspace)]).exit_code == 0
    result = runner.invoke(app, ["validate", str(workspace), "--level", "contract"])

    assert result.exit_code == 0, result.output
    assert (workspace / "validation" / "validation_suite.json").exists()
