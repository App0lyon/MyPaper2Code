from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mypaper2code.core.models import ImplementationRequirements
from mypaper2code.services.analysis import MethodAnalyzer
from mypaper2code.services.code_qa import CodeQuestionAnswerer
from mypaper2code.services.config import load_config, set_config
from mypaper2code.services.generation import CodeWriter
from mypaper2code.services.ingestion import IngestionService
from mypaper2code.services.planning import ImplementationPlanner
from mypaper2code.services.report import ReportWriter
from mypaper2code.services.requirements import (
    load_requirements,
    save_requirements,
    set_requirement,
)
from mypaper2code.services.search.hybrid import HybridRetriever
from mypaper2code.services.validation import ExperimentRunner
from mypaper2code.services.workspace import WorkspaceManager

app = typer.Typer(help="Local assistant for paper-to-code workspaces.")
config_app = typer.Typer(help="Read and update MyPaper2Code configuration.")
requirements_app = typer.Typer(help="Read and update workspace implementation requirements.")
app.add_typer(config_app, name="config")
app.add_typer(requirements_app, name="requirements")
console = Console()


@app.command()
def ingest(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    provider: str | None = typer.Option(None, help="LLM provider name."),
    model: str | None = typer.Option(None, help="LLM model name."),
) -> None:
    """Ingest a PDF and create a workspace."""
    config = load_config()
    workspace = IngestionService().ingest(
        pdf,
        provider=provider or config["provider"],
        model=model or config["model"],
    )
    console.print(str(workspace))


@app.command()
def ask_paper(
    question: Annotated[str, typer.Argument()],
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
    limit: int = typer.Option(5, help="Number of fused passages to return."),
) -> None:
    """Ask a sourced question against an ingested workspace."""
    WorkspaceManager.ensure(workspace)
    hits = HybridRetriever(workspace).search(question, limit=limit)
    if not hits:
        console.print("No relevant passage found.")
        raise typer.Exit(code=1)
    table = Table(title="Hybrid RRF results")
    table.add_column("Score")
    table.add_column("Page")
    table.add_column("Section")
    table.add_column("Passage")
    for hit in hits:
        table.add_row(f"{hit.score:.4f}", str(hit.page), hit.section, hit.text)
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
    limit: int = typer.Option(5, help="Number of fused passages to return."),
) -> None:
    """Backward-compatible alias for ask-paper."""
    ask_paper(question, workspace, limit)


@app.command()
def ask_code(
    question: Annotated[str, typer.Argument()],
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
    limit: int = typer.Option(5, help="Number of code/trace matches to return."),
) -> None:
    """Ask about the generated implementation using trace and direct file search."""
    WorkspaceManager.ensure(workspace)
    answers = CodeQuestionAnswerer().answer(workspace, question, limit=limit)
    if not answers:
        console.print("No implementation evidence found.")
        raise typer.Exit(code=1)
    for answer in answers:
        console.print(f"- {answer}")


@app.command()
def analyze(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Extract a method summary and uncertainty report."""
    WorkspaceManager.ensure(workspace)
    summary, assumptions = MethodAnalyzer().analyze(workspace)
    console.print(f"Wrote {summary}")
    console.print(f"Wrote {assumptions}")


@app.command(name="plan")
def plan_command(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
    framework: str | None = None,
    dataset: str | None = None,
    style: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Create an implementation plan for a workspace."""
    WorkspaceManager.ensure(workspace)
    config = load_config()
    stored = load_requirements(workspace)
    requirements = ImplementationRequirements(
        **(
            stored.model_dump()
            | {
                "framework": framework or stored.framework,
                "dataset": dataset or stored.dataset,
                "style": style or stored.style,
                "provider": provider or stored.provider or config["provider"],
                "model": model or stored.model or config["model"],
            }
        )
    )
    save_requirements(workspace, requirements)
    ImplementationPlanner().create_plan(workspace, requirements)
    console.print(str(workspace / "analysis" / "implementation_plan.md"))


@app.command()
def implement(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Implement the paper-specific plan step by step."""
    WorkspaceManager.ensure(workspace)
    generated = CodeWriter().implement(workspace)
    console.print(str(generated))


@app.command()
def generate(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Backward-compatible alias for implement."""
    implement(workspace)


@app.command()
def report(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Write a fidelity and assumption report."""
    WorkspaceManager.ensure(workspace)
    path = ReportWriter().write(workspace)
    console.print(str(path))


@requirements_app.command("get")
def requirements_get(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Read workspace implementation requirements."""
    WorkspaceManager.ensure(workspace)
    console.print_json(data=load_requirements(workspace).model_dump(mode="json"))


@requirements_app.command("set")
def requirements_set(
    key: str,
    value: str,
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Set one workspace implementation requirement."""
    WorkspaceManager.ensure(workspace)
    requirements = set_requirement(workspace, key, value)
    console.print_json(data=requirements.model_dump(mode="json"))


@app.callback()
def main() -> None:
    return None


@app.command()
def validate(
    workspace: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Run minimal validation checks for generated code."""
    WorkspaceManager.ensure(workspace)
    results = ExperimentRunner().validate(workspace)
    failed = False
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        failed = failed or not result.passed
        console.print(f"{status} {result.name}: {result.log_path}")
    if failed:
        raise typer.Exit(code=1)


@config_app.command("get")
def config_get(key: str | None = None) -> None:
    """Read configuration."""
    config = load_config()
    if key:
        console.print(config[key])
    else:
        console.print_json(data=config)


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set configuration value."""
    config = set_config(key, value)
    console.print_json(data=config)


if __name__ == "__main__":
    app()
