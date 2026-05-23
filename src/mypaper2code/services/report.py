from __future__ import annotations

from pathlib import Path

from mypaper2code.core.io import load_model, read_json
from mypaper2code.core.models import (
    AgenticImplementationTrace,
    ImplementationPlan,
    ImplementationTrace,
    ResearchImplementationPlan,
)


class ReportWriter:
    def write(self, workspace: Path) -> Path:
        research_plan_path = workspace / "plan" / "research_plan.json"
        if research_plan_path.exists():
            return self._write_research_report(workspace, research_plan_path)
        plan = load_model(workspace / "analysis" / "implementation_plan.json", ImplementationPlan)
        trace_path = workspace / "analysis" / "implementation_trace.json"
        trace = (
            ImplementationTrace.model_validate(read_json(trace_path))
            if trace_path.exists()
            else ImplementationTrace()
        )
        lines = [
            "# Fidelity and Assumption Report",
            "",
            "## Paper Understanding",
            "",
            f"- Architecture: `{plan.understanding.architecture}`",
            f"- Loss: `{plan.understanding.loss}`",
            f"- Datasets: {', '.join(plan.understanding.datasets) or 'unspecified'}",
            f"- Metrics: {', '.join(plan.understanding.metrics) or 'unspecified'}",
            "",
            "## Implemented Steps",
            "",
        ]
        for entry in trace.entries:
            locations = ", ".join(location.file for location in entry.implemented_in)
            lines.append(f"- `{entry.step_id}`: {entry.paper_claim} -> {locations}")
        lines.extend(["", "## Assumptions and Unclear Items", ""])
        lines.extend(f"- {item}" for item in plan.assumptions)
        fidelity = "Medium" if trace.entries and plan.understanding.ambiguities else "High"
        if not trace.entries:
            fidelity = "Low"
        lines.extend(["", f"Fidelity level: {fidelity}", ""])
        path = workspace / "analysis" / "fidelity_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_research_report(self, workspace: Path, plan_path: Path) -> Path:
        plan = load_model(plan_path, ResearchImplementationPlan)
        trace_path = workspace / "trace" / "implementation_trace.json"
        trace = (
            AgenticImplementationTrace.model_validate(read_json(trace_path))
            if trace_path.exists()
            else AgenticImplementationTrace()
        )
        validation_path = workspace / "validation" / "validation_suite.json"
        validation = read_json(validation_path) if validation_path.exists() else {}
        fidelity = validation.get("fidelity_score") or _report_fidelity(plan, trace)
        reasons = validation.get("reasons") or _report_reasons(plan, trace)
        lines = [
            "# Fidelity and Assumption Report",
            "",
            f"Fidelity level: {fidelity}",
            "",
            "## Paper Understanding",
            "",
            f"- Paper type: `{plan.paper_type}`",
            f"- Contributions: {', '.join(plan.understanding.contributions) or 'unspecified'}",
            f"- Algorithms: {', '.join(plan.understanding.algorithms) or 'unspecified'}",
            f"- Metrics: {', '.join(plan.understanding.metrics) or 'unspecified'}",
            f"- Protocols: {', '.join(plan.understanding.protocols) or 'unspecified'}",
            "",
            "## Implemented Steps",
            "",
        ]
        for entry in trace.entries:
            locations = ", ".join(location.file for location in entry.implemented_in)
            lines.append(f"- `{entry.step_id}`: {entry.claim} -> {locations}")
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- {item}" for item in reasons)
        lines.extend(["", "## Open Assumptions and Ambiguities", ""])
        if plan.understanding.ambiguities:
            lines.extend(
                f"- `{item.ambiguity_id}` ({item.severity}): {item.question}"
                for item in plan.understanding.ambiguities
            )
        else:
            lines.append("- None recorded.")
        path = workspace / "validation" / "fidelity_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        (workspace / "analysis").mkdir(parents=True, exist_ok=True)
        (workspace / "analysis" / "fidelity_report.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        return path


def _report_fidelity(plan: ResearchImplementationPlan, trace: AgenticImplementationTrace) -> str:
    if plan.blocking_decisions:
        return "blocked"
    if not trace.entries:
        return "low"
    if plan.understanding.metrics and plan.understanding.evidence and plan.understanding.protocols:
        return "high"
    if plan.understanding.evidence:
        return "medium"
    return "low"


def _report_reasons(
    plan: ResearchImplementationPlan,
    trace: AgenticImplementationTrace,
) -> list[str]:
    reasons = []
    if plan.blocking_decisions:
        reasons.append(f"Blocking decisions remain: {', '.join(plan.blocking_decisions)}.")
    if not trace.entries:
        reasons.append("No implementation trace entries were found.")
    for key, value in {
        "evidence": plan.understanding.evidence,
        "metrics": plan.understanding.metrics,
        "protocols": plan.understanding.protocols,
        "algorithms_or_contributions": (
            plan.understanding.algorithms or plan.understanding.contributions
        ),
    }.items():
        if not value:
            reasons.append(f"Missing coverage for {key}.")
    return reasons or ["Evidence, trace, metrics, and protocol coverage are present."]
