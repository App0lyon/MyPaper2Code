from __future__ import annotations

from pathlib import Path

from mypaper2code.core.io import load_model, write_json, write_yaml
from mypaper2code.core.models import (
    ImplementationPlan,
    ImplementationRequirements,
    ImplementationStep,
    PaperUnderstanding,
)
from mypaper2code.services.analysis import MethodAnalyzer, build_understanding, sources_for_plan
from mypaper2code.services.requirements import load_requirements, save_requirements


class ImplementationPlanner:
    def create_plan(
        self,
        workspace: Path,
        requirements: ImplementationRequirements | None = None,
    ) -> ImplementationPlan:
        requirements = requirements or load_requirements(workspace)
        save_requirements(workspace, requirements)
        understanding = self._load_or_analyze_understanding(workspace)
        sources = sources_for_plan(workspace)
        plan = ImplementationPlan(
            requirements=requirements,
            understanding=understanding,
            steps=self._steps_for(understanding, sources, requirements),
            assumptions=self._assumptions_for(understanding, requirements),
            sources=sources,
        )
        self.write(workspace, plan)
        return plan

    @staticmethod
    def _load_or_analyze_understanding(workspace: Path) -> PaperUnderstanding:
        path = workspace / "analysis" / "paper_understanding.json"
        if not path.exists():
            MethodAnalyzer().analyze(workspace)
        if path.exists():
            return load_model(path, PaperUnderstanding)
        return build_understanding(workspace)

    @staticmethod
    def write(workspace: Path, plan: ImplementationPlan) -> None:
        analysis_dir = workspace / "analysis"
        write_yaml(analysis_dir / "requirements.yaml", plan.requirements.model_dump(mode="json"))
        write_json(analysis_dir / "implementation_plan.json", plan.model_dump(mode="json"))
        lines = ["# Implementation Plan", "", "## Requirements", ""]
        for key, value in plan.requirements.model_dump().items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Paper Understanding", ""])
        lines.append(f"- `architecture`: {plan.understanding.architecture}")
        lines.append(f"- `loss`: {plan.understanding.loss}")
        lines.append(f"- `datasets`: {', '.join(plan.understanding.datasets) or 'unspecified'}")
        lines.append(f"- `metrics`: {', '.join(plan.understanding.metrics) or 'unspecified'}")
        lines.extend(["", "## Implementation Steps", ""])
        for step in plan.steps:
            lines.append(f"### {step.step_id}. {step.title}")
            lines.append(f"- Purpose: {step.purpose}")
            lines.append(f"- Files: {', '.join(f'`{file}`' for file in step.files)}")
            if step.symbols:
                lines.append(f"- Symbols: {', '.join(f'`{symbol}`' for symbol in step.symbols)}")
            if step.assumptions:
                lines.append(f"- Assumptions: {'; '.join(step.assumptions)}")
            lines.append("")
        lines.extend(["", "## Assumptions", ""])
        lines.extend(f"- {item}" for item in plan.assumptions)
        lines.extend(["", "## Sources", ""])
        if plan.sources:
            lines.extend(
                f"- Page {source.page}, section `{source.section}`: {source.text}"
                for source in plan.sources
            )
        else:
            lines.append("- No source passage available.")
        (analysis_dir / "implementation_plan.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    @staticmethod
    def _steps_for(
        understanding: PaperUnderstanding,
        sources,
        requirements: ImplementationRequirements,
    ) -> list[ImplementationStep]:
        model_symbol = _model_symbol(understanding.architecture)
        loss_symbol = _loss_symbol(understanding.loss)
        return [
            ImplementationStep(
                step_id="config",
                title="Persist user requirements and runnable defaults",
                files=["configs/default.yaml", "requirements.txt", "README.md"],
                purpose=(
                    f"Create project metadata for {requirements.framework}/"
                    f"{requirements.dataset}."
                ),
                source_refs=sources[:2],
            ),
            ImplementationStep(
                step_id="data",
                title="Implement the data entrypoint requested by the user",
                files=["src/data/datasets.py"],
                purpose=(
                    f"Expose a loader compatible with `{requirements.dataset}` and smoke tests."
                ),
                symbols=["make_data_loader"],
                source_refs=sources[:2],
                assumptions=[
                    "Use synthetic fallback data when the requested dataset is unavailable locally."
                ],
            ),
            ImplementationStep(
                step_id="model",
                title=f"Implement paper-derived architecture: {understanding.architecture}",
                files=["src/models/model.py"],
                purpose=(
                    "Translate the architecture found in the paper understanding into "
                    "PyTorch modules."
                ),
                symbols=[model_symbol],
                source_refs=sources[:3],
                assumptions=(
                    ["Architecture is underspecified; choose a small configurable module."]
                    if understanding.architecture == "unspecified"
                    else []
                ),
            ),
            ImplementationStep(
                step_id="loss",
                title=f"Implement paper-derived loss: {understanding.loss}",
                files=["src/losses/losses.py"],
                purpose="Expose the objective identified in the paper understanding.",
                symbols=[loss_symbol, "build_loss"],
                source_refs=sources[:3],
                assumptions=(
                    ["Loss is underspecified; expose a documented supervised fallback."]
                    if understanding.loss == "unspecified"
                    else []
                ),
            ),
            ImplementationStep(
                step_id="training",
                title="Implement training and evaluation scripts around the planned components",
                files=[
                    "src/training/trainer.py",
                    "src/evaluation/metrics.py",
                    "src/utils/config.py",
                    "scripts/train.py",
                    "scripts/evaluate.py",
                ],
                purpose="Wire the data, model, loss, metrics, and config into executable scripts.",
                symbols=["train_one_epoch", "accuracy", "load_config"],
                source_refs=sources[:3],
            ),
            ImplementationStep(
                step_id="tests",
                title="Add smoke tests for the planned implementation",
                files=["tests/test_smoke.py"],
                purpose="Verify generated modules import and execute a forward/loss/metric path.",
                symbols=["test_generated_pipeline_smoke"],
            ),
        ]

    @staticmethod
    def _assumptions_for(
        understanding: PaperUnderstanding,
        requirements: ImplementationRequirements,
    ) -> list[str]:
        assumptions = list(understanding.ambiguities)
        assumptions.append(
            f"Implementation follows user requirements: framework={requirements.framework}, "
            f"dataset={requirements.dataset}, style={requirements.style}."
        )
        assumptions.append(
            "Each implementation step records a trace entry linking sources to files."
        )
        return assumptions


def _model_symbol(architecture: str) -> str:
    return {
        "transformer": "PaperTransformer",
        "cnn": "PaperCNN",
        "diffusion": "PaperDiffusionStub",
        "gan": "PaperGANStub",
        "vae": "PaperVAE",
        "mlp": "PaperMLP",
    }.get(architecture, "PaperModel")


def _loss_symbol(loss: str) -> str:
    return {
        "contrastive": "ContrastiveLoss",
        "cross_entropy": "CrossEntropyLoss",
        "mse": "MSELoss",
        "triplet": "TripletLoss",
    }.get(loss, "PaperLoss")
