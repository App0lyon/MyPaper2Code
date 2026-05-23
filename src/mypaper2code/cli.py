from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mypaper2code.core.models import ImplementationRequirements
from mypaper2code.providers.base import (
    NVIDIA_DEFAULT_MODEL,
    OLLAMA_DEFAULT_MODEL,
    ProviderError,
    provider_for,
)
from mypaper2code.services.analysis import MethodAnalyzer
from mypaper2code.services.code_qa import CodeQuestionAnswerer
from mypaper2code.services.config import load_config, set_config
from mypaper2code.services.decisions import (
    load_approvals,
    load_decisions,
    save_approval,
    save_decision,
)
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
providers_app = typer.Typer(help="Test configured LLM providers.")
app.add_typer(config_app, name="config")
app.add_typer(requirements_app, name="requirements")
app.add_typer(providers_app, name="providers")
console = Console()


@app.command()
def run(
    pdf: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional PDF to ingest before running the workflow.",
        ),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            exists=True,
            file_okay=False,
            help="Existing workspace to continue.",
        ),
    ] = None,
    ask_paper_question: str | None = typer.Option(
        None,
        "--ask-paper",
        help="Ask a sourced question about the paper after ingestion.",
    ),
    ask_code_question: str | None = typer.Option(
        None,
        "--ask-code",
        help="Ask a question about the implementation after implement.",
    ),
    framework: str | None = None,
    dataset: str | None = None,
    style: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    analyze_: bool = typer.Option(True, "--analyze/--no-analyze", help="Run paper analysis."),
    plan_: bool = typer.Option(True, "--plan/--no-plan", help="Create/update implementation plan."),
    implement_: bool = typer.Option(
        True,
        "--implement/--no-implement",
        help="Implement the plan.",
    ),
    validate_: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate generated code.",
    ),
    validation_level: str = typer.Option(
        "smoke",
        "--level",
        help="Validation level: smoke, contract, or repro.",
    ),
    report_: bool = typer.Option(True, "--report/--no-report", help="Write fidelity report."),
) -> None:
    """Single entrypoint for the full paper-to-code workflow."""
    if pdf is None and workspace is None:
        raise typer.BadParameter("Provide either a PDF argument or --workspace.")
    config = load_config()
    selected_provider = provider or config["provider"]
    selected_model = model or config["model"]
    _print_header("MyPaper2Code workflow")
    _print_context(
        "Run context",
        {
            "PDF": str(pdf) if pdf else "not provided",
            "Workspace": str(workspace) if workspace else "will be created",
            "Provider": selected_provider,
            "Model": selected_model,
            "Analyze": _yes_no(analyze_),
            "Plan": _yes_no(plan_),
            "Implement": _yes_no(implement_),
            "Validate": f"{_yes_no(validate_)} ({validation_level})",
            "Report": _yes_no(report_),
        },
    )

    if pdf is not None:
        _print_step("Ingest", "Extracting PDF text, chunking passages, and building indexes.")
        console.print(f"[dim]Input PDF:[/dim] {pdf}")
        with console.status("Creating workspace and retrieval indexes...", spinner="dots"):
            workspace = IngestionService().ingest(
                pdf,
                provider=selected_provider,
                model=selected_model,
            )
        _print_success("Workspace created")
        _print_path("Workspace", workspace)
    else:
        workspace = WorkspaceManager.ensure(workspace)
        _print_step("Workspace", "Continuing from an existing workspace.")
        _print_path("Workspace", workspace)

    assert workspace is not None
    requirements = _merged_requirements(
        workspace,
        framework=framework,
        dataset=dataset,
        style=style,
        provider=selected_provider,
        model=selected_model,
    )
    _print_requirements(requirements)

    if ask_paper_question:
        _print_step("Paper question", "Retrieving sourced passages from the paper index.")
        _print_paper_hits(workspace, ask_paper_question)

    if analyze_:
        _print_step("Analyze", "Asking the provider chain for structured paper understanding.")
        _print_provider_chain(selected_provider, selected_model)
        with console.status(
            "Reading retrieved passages and calling the provider...",
            spinner="dots",
        ):
            summary, assumptions = MethodAnalyzer().analyze(workspace)
        _print_success("Paper analysis written")
        _print_path("Summary", summary)
        _print_path("Assumptions", assumptions)
        _print_path("Understanding JSON", workspace / "analysis" / "paper_understanding.json")

    if plan_:
        _print_step("Plan", "Creating the implementation plan from requirements and paper facts.")
        with console.status("Writing plan artifacts...", spinner="dots"):
            plan = ImplementationPlanner().create_plan(workspace, requirements)
        _print_success(f"Implementation plan ready with {len(plan.steps)} steps")
        _print_path("Plan Markdown", workspace / "plan" / "implementation_plan.md")
        _print_path("Plan JSON", workspace / "plan" / "research_plan.json")

    if implement_:
        _print_step("Implement", "Generating the planned source tree and implementation trace.")
        try:
            with console.status("Writing generated code...", spinner="dots"):
                generated = CodeWriter().implement(workspace)
        except RuntimeError as exc:
            _print_failure(str(exc))
            raise typer.Exit(code=1) from exc
        _print_success("Generated implementation written")
        _print_path("Generated code", generated)
        _print_path("Trace", workspace / "trace" / "implementation_trace.json")

    if validate_:
        _print_step("Validate", "Running validation checks.")
        _print_validation_results(workspace, level=validation_level)

    if report_:
        _print_step("Report", "Writing the fidelity and assumptions report.")
        with console.status("Building report...", spinner="dots"):
            report_path = ReportWriter().write(workspace)
        _print_success("Report written")
        _print_path("Report", report_path)

    if ask_code_question:
        _print_step("Code question", "Searching the trace and generated files.")
        _print_code_answers(workspace, ask_code_question)

    _print_success("Workflow complete")


@app.command()
def ingest(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    provider: str | None = typer.Option(None, help="LLM provider name."),
    model: str | None = typer.Option(None, help="LLM model name."),
) -> None:
    """Ingest a PDF and create a workspace."""
    config = load_config()
    selected_provider = provider or config["provider"]
    selected_model = model or config["model"]
    _print_header("Ingest paper")
    _print_context(
        "Ingestion context",
        {"PDF": str(pdf), "Provider": selected_provider, "Model": selected_model},
    )
    with console.status(
        "Extracting PDF, chunking text, and building search indexes...",
        spinner="dots",
    ):
        workspace = IngestionService().ingest(
            pdf,
            provider=selected_provider,
            model=selected_model,
        )
    _print_success("Workspace created")
    _print_path("Workspace", workspace)
    _print_path("Chunks", workspace / "paper" / "chunks.json")
    _print_path("Retrieval config", workspace / "paper" / "retrieval_config.json")


@app.command()
def ask_paper(
    question: Annotated[str, typer.Argument()],
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
    limit: int = typer.Option(5, help="Number of fused passages to return."),
) -> None:
    """Ask a sourced question against an ingested workspace."""
    WorkspaceManager.ensure(workspace)
    _print_header("Paper question")
    _print_context(
        "Question context",
        {"Workspace": str(workspace), "Question": question, "Limit": limit},
    )
    _print_paper_hits(workspace, question, limit=limit)


def _print_paper_hits(workspace: Path, question: str, limit: int = 5) -> None:
    console.print(f"[dim]Searching paper passages for:[/dim] {question}")
    with console.status("Running hybrid retrieval...", spinner="dots"):
        hits = HybridRetriever(workspace).search(question, limit=limit)
    if not hits:
        _print_failure("No relevant paper passage found.")
        raise typer.Exit(code=1)
    _print_success(f"Found {len(hits)} sourced passage(s)")
    table = Table(title="Hybrid retrieval results")
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
    _print_header("Code question")
    _print_context(
        "Question context",
        {"Workspace": str(workspace), "Question": question, "Limit": limit},
    )
    _print_code_answers(workspace, question, limit=limit)


def _print_code_answers(workspace: Path, question: str, limit: int = 5) -> None:
    console.print(f"[dim]Searching implementation evidence for:[/dim] {question}")
    with console.status("Searching trace and generated files...", spinner="dots"):
        answers = CodeQuestionAnswerer().answer(workspace, question, limit=limit)
    if not answers:
        _print_failure("No implementation evidence found.")
        raise typer.Exit(code=1)
    _print_success(f"Found {len(answers)} implementation evidence item(s)")
    for answer in answers:
        console.print(f"[bold]-[/bold] {answer}")


@app.command()
def analyze(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Extract a method summary and uncertainty report."""
    WorkspaceManager.ensure(workspace)
    requirements = load_requirements(workspace)
    _print_header("Analyze paper")
    _print_context(
        "Analysis context",
        {
            "Workspace": str(workspace),
            "Provider": requirements.provider,
            "Model": requirements.model,
        },
    )
    _print_provider_chain(requirements.provider, requirements.model)
    with console.status("Retrieving passages and extracting structured facts...", spinner="dots"):
        summary, assumptions = MethodAnalyzer().analyze(workspace)
    _print_success("Paper analysis written")
    _print_path("Summary", summary)
    _print_path("Assumptions", assumptions)
    _print_path("Understanding JSON", workspace / "analysis" / "paper_understanding.json")


@app.command()
def understand(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Extract rich, evidence-backed research understanding."""
    WorkspaceManager.ensure(workspace)
    requirements = load_requirements(workspace)
    _print_header("Understand paper")
    _print_context(
        "Understanding context",
        {
            "Workspace": str(workspace),
            "Provider": requirements.provider,
            "Model": requirements.model,
        },
    )
    with console.status("Extracting rich structured understanding...", spinner="dots"):
        understanding = MethodAnalyzer().understand(workspace)
    _print_success("Research understanding written")
    _print_context(
        "Understanding summary",
        {
            "Paper type": understanding.paper_type,
            "Contributions": len(understanding.contributions),
            "Algorithms": len(understanding.algorithms),
            "Equations": len(understanding.equations),
            "Ambiguities": len(understanding.ambiguities),
        },
    )
    _print_path("Understanding JSON", workspace / "understanding" / "research_understanding.json")
    _print_path("Review", workspace / "understanding" / "review.md")


@app.command()
def review(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """List open ambiguities and recorded human decisions."""
    from mypaper2code.core.io import read_json
    from mypaper2code.core.models import ResearchUnderstanding

    WorkspaceManager.ensure(workspace)
    path = workspace / "understanding" / "research_understanding.json"
    if not path.exists():
        _print_failure("Run `understand` before review.")
        raise typer.Exit(code=1)
    understanding = ResearchUnderstanding.model_validate(read_json(path))
    decisions = load_decisions(workspace)
    approvals = load_approvals(workspace)
    _print_header("Human review")
    table = Table(title="Ambiguities")
    table.add_column("ID")
    table.add_column("Severity")
    table.add_column("Status")
    table.add_column("Question")
    for item in understanding.ambiguities:
        status = "answered" if item.ambiguity_id in decisions else "open"
        if approvals.get(item.ambiguity_id):
            status = "approved"
        table.add_row(item.ambiguity_id, item.severity, status, item.question)
    console.print(table)
    if decisions:
        _print_context("Recorded decisions", {key: value.value for key, value in decisions.items()})
    else:
        console.print("[dim]No human decisions recorded yet.[/dim]")


@app.command()
def decide(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
    decision_id: Annotated[str, typer.Option("--id", help="Ambiguity or decision id.")],
    value: Annotated[str, typer.Option("--value", help="Decision value.")],
) -> None:
    """Record a human decision for an ambiguity."""
    WorkspaceManager.ensure(workspace)
    decision = save_decision(workspace, decision_id, value)
    _print_success(f"Decision `{decision.decision_id}` saved")


@app.command("approve-plan")
def approve_plan(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Approve the current implementation plan."""
    WorkspaceManager.ensure(workspace)
    save_approval(workspace, "plan")
    _print_success("Plan approved")


@app.command("approve-assumption")
def approve_assumption(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
    assumption_id: Annotated[str, typer.Option("--id", help="Ambiguity or assumption id.")],
) -> None:
    """Approve one extracted assumption or ambiguity."""
    WorkspaceManager.ensure(workspace)
    save_approval(workspace, assumption_id)
    _print_success(f"Assumption `{assumption_id}` approved")


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
    requirements = _merged_requirements(
        workspace,
        framework=framework,
        dataset=dataset,
        style=style,
        provider=provider or config["provider"],
        model=model or config["model"],
    )
    _print_header("Plan implementation")
    _print_path("Workspace", workspace)
    _print_requirements(requirements)
    with console.status("Creating implementation plan...", spinner="dots"):
        plan = ImplementationPlanner().create_plan(workspace, requirements)
    _print_success(f"Implementation plan ready with {len(plan.steps)} steps")
    _print_path("Plan Markdown", workspace / "plan" / "implementation_plan.md")
    _print_path("Plan JSON", workspace / "plan" / "research_plan.json")


@app.command()
def implement(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Implement the paper-specific plan step by step."""
    WorkspaceManager.ensure(workspace)
    _print_header("Implement plan")
    _print_path("Workspace", workspace)
    try:
        with console.status("Writing generated code and trace...", spinner="dots"):
            generated = CodeWriter().implement(workspace)
    except RuntimeError as exc:
        _print_failure(str(exc))
        raise typer.Exit(code=1) from exc
    _print_success("Generated implementation written")
    _print_path("Generated code", generated)
    _print_path("Trace", workspace / "trace" / "implementation_trace.json")


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
    _print_header("Fidelity report")
    _print_path("Workspace", workspace)
    with console.status("Writing report...", spinner="dots"):
        path = ReportWriter().write(workspace)
    _print_success("Report written")
    _print_path("Report", path)


@requirements_app.command("get")
def requirements_get(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Read workspace implementation requirements."""
    WorkspaceManager.ensure(workspace)
    _print_header("Requirements")
    _print_path("Workspace", workspace)
    _print_requirements(load_requirements(workspace))


@requirements_app.command("set")
def requirements_set(
    key: str,
    value: str,
    workspace: Annotated[Path, typer.Option("--workspace", "-w", exists=True, file_okay=False)],
) -> None:
    """Set one workspace implementation requirement."""
    WorkspaceManager.ensure(workspace)
    _print_header("Update requirement")
    console.print(f"[dim]Setting[/dim] {key} = {value}")
    requirements = set_requirement(workspace, key, value)
    _print_success("Requirement saved")
    _print_requirements(requirements)


@app.callback()
def main() -> None:
    return None


@app.command()
def validate(
    workspace: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    level: str = typer.Option("smoke", "--level", help="smoke, contract, or repro."),
) -> None:
    """Run minimal validation checks for generated code."""
    WorkspaceManager.ensure(workspace)
    _print_header("Validate generated code")
    _print_path("Workspace", workspace)
    _print_validation_results(workspace, level=level)


def _print_validation_results(workspace: Path, level: str = "smoke") -> None:
    with console.status("Running validation commands...", spinner="dots"):
        suite = ExperimentRunner().validate_suite(workspace, level=level)
    results = suite.results
    failed = False
    table = Table(title="Validation results")
    table.add_column("Status")
    table.add_column("Check")
    table.add_column("Command")
    table.add_column("Log")
    for result in results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        failed = failed or not result.passed
        table.add_row(status, result.name, result.command, result.log_path)
    console.print(table)
    _print_context(
        "Fidelity",
        {
            "Level": suite.level,
            "Score": suite.fidelity_score,
            "Reasons": " ".join(suite.reasons),
        },
    )
    if failed:
        _print_failure("Validation failed. Open the log paths above for details.")
        raise typer.Exit(code=1)
    _print_success("All validation checks passed")


def _merged_requirements(
    workspace: Path,
    framework: str | None,
    dataset: str | None,
    style: str | None,
    provider: str | None,
    model: str | None,
) -> ImplementationRequirements:
    stored = load_requirements(workspace)
    requirements = ImplementationRequirements(
        **(
            stored.model_dump()
            | {
                "framework": framework or stored.framework,
                "dataset": dataset or stored.dataset,
                "style": style or stored.style,
                "provider": provider or stored.provider,
                "model": model or stored.model,
            }
        )
    )
    save_requirements(workspace, requirements)
    return requirements


@config_app.command("get")
def config_get(key: str | None = None) -> None:
    """Read configuration."""
    config = load_config()
    _print_header("Configuration")
    if key:
        console.print(f"[bold]{key}[/bold]: {config[key]}")
    else:
        _print_context("Current config", {str(k): str(v) for k, v in config.items()})


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set configuration value."""
    _print_header("Update configuration")
    console.print(f"[dim]Setting[/dim] {key} = {value}")
    try:
        config = set_config(key, value)
    except (KeyError, ValueError) as exc:
        _print_failure(f"Config error: {exc}")
        raise typer.Exit(code=1) from exc
    _print_success("Configuration saved")
    _print_context("Current config", {str(k): str(v) for k, v in config.items()})


@providers_app.command("test")
def providers_test(
    prompt: str = typer.Option(
        "Reply with the single word ok.",
        help="Prompt sent to the provider.",
    ),
    provider: str | None = typer.Option(None, help="Override configured provider."),
    model: str | None = typer.Option(None, help="Override configured model."),
) -> None:
    """Send a small completion request to the selected provider."""
    config = load_config()
    provider_name = (provider or config["provider"]).lower()
    errors: list[str] = []
    _print_header("Provider test")
    _print_context(
        "Provider context",
        {"Requested provider": provider_name, "Requested model": model or config["model"]},
    )
    try:
        candidates = _provider_order(provider_name)
    except ProviderError as exc:
        _print_failure(f"Provider error: {exc}")
        raise typer.Exit(code=1) from exc
    for candidate in candidates:
        selected = provider_for(
            candidate,
            _model_for_provider(candidate, model or config["model"]),
            ollama_base_url=config["ollama_base_url"],
            nvidia_base_url=config["nvidia_base_url"],
            nvidia_api_key_env=config["nvidia_api_key_env"],
        )
        try:
            console.print(f"[dim]Trying provider:[/dim] {candidate} ({selected.model})")
            response = selected.complete(prompt, max_tokens=64)
            _print_success(f"Provider `{candidate}` responded")
            console.print(Panel(response, title="Provider response", border_style="green"))
            return
        except ProviderError as exc:
            errors.append(f"{candidate}: {exc}")
            _print_failure(f"Provider `{candidate}` failed: {exc}")
    _print_failure(f"Provider error: {'; '.join(errors)}")
    raise typer.Exit(code=1)


def _provider_order(provider_name: str) -> list[str]:
    if provider_name == "nvidia":
        return ["nvidia", "ollama"]
    if provider_name == "ollama":
        return ["ollama"]
    raise ProviderError(f"Unsupported provider `{provider_name}`. Use `nvidia` or `ollama`.")


def _model_for_provider(provider_name: str, model: str) -> str:
    if provider_name == "ollama" and model == NVIDIA_DEFAULT_MODEL:
        return OLLAMA_DEFAULT_MODEL
    if provider_name == "nvidia" and model in {"", OLLAMA_DEFAULT_MODEL, "stub"}:
        return NVIDIA_DEFAULT_MODEL
    if provider_name == "ollama" and model in {"", "stub"}:
        return OLLAMA_DEFAULT_MODEL
    return model


def _print_header(title: str) -> None:
    console.print()
    console.print(Panel.fit(f"[bold]{title}[/bold]", border_style="cyan"))


def _print_step(title: str, detail: str) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}")
    console.print(f"[dim]{detail}[/dim]")


def _print_success(message: str) -> None:
    console.print(f"[green]OK[/green] {message}")


def _print_failure(message: str) -> None:
    console.print(f"[red]ERROR[/red] {message}")


def _print_path(label: str, path: Path) -> None:
    console.print(f"[bold]{label}:[/bold] {path}")


def _print_context(title: str, values: dict[str, object]) -> None:
    table = Table(title=title, show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(key, str(value))
    console.print(table)


def _print_requirements(requirements: ImplementationRequirements) -> None:
    _print_context(
        "Implementation requirements",
        {
            "Framework": requirements.framework,
            "Dataset": requirements.dataset,
            "Style": requirements.style,
            "Implementation level": requirements.implementation_level,
            "Include tests": _yes_no(requirements.include_tests),
            "Training script": _yes_no(requirements.include_training_script),
            "Evaluation script": _yes_no(requirements.include_evaluation_script),
            "Provider": requirements.provider,
            "Model": requirements.model,
        },
    )


def _print_provider_chain(provider_name: str, model: str) -> None:
    chain = " -> ".join(_provider_order(provider_name))
    console.print(f"[bold]Provider chain:[/bold] {chain}")
    console.print(f"[bold]Requested model:[/bold] {model}")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    app()
