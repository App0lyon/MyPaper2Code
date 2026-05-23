from __future__ import annotations

from pathlib import Path

from mypaper2code.core.io import load_model, read_json
from mypaper2code.core.models import ImplementationPlan, ImplementationTrace


class ReportWriter:
    def write(self, workspace: Path) -> Path:
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
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
