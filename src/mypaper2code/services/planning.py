from __future__ import annotations

from pathlib import Path

from mypaper2code.core.io import load_model, write_json, write_yaml
from mypaper2code.core.models import (
    AgenticImplementationStep,
    ImplementationPlan,
    ImplementationRequirements,
    ImplementationStep,
    PaperUnderstanding,
    ResearchImplementationPlan,
    ResearchUnderstanding,
)
from mypaper2code.services.analysis import (
    MethodAnalyzer,
    legacy_understanding,
    load_research_understanding,
)
from mypaper2code.services.decisions import load_approvals, unresolved_blocking_ambiguities
from mypaper2code.services.requirements import load_requirements, save_requirements


class ImplementationPlanner:
    def create_plan(
        self,
        workspace: Path,
        requirements: ImplementationRequirements | None = None,
    ) -> ResearchImplementationPlan:
        requirements = requirements or load_requirements(workspace)
        save_requirements(workspace, requirements)
        understanding = self._load_or_analyze_understanding(workspace)
        blocking = unresolved_blocking_ambiguities(understanding, workspace)
        approvals = load_approvals(workspace)
        plan = ResearchImplementationPlan(
            requirements=requirements,
            understanding=understanding,
            paper_type=understanding.paper_type,
            steps=self._agentic_steps_for(understanding, requirements),
            blocking_decisions=[item.ambiguity_id for item in blocking],
            assumptions=self._assumptions_for(understanding, requirements),
            provider=understanding.provider,
            model=understanding.model,
            approved=approvals.get("plan", False),
        )
        self.write(workspace, plan)
        return plan

    @staticmethod
    def _load_or_analyze_understanding(workspace: Path) -> ResearchUnderstanding:
        path = workspace / "understanding" / "research_understanding.json"
        legacy_path = workspace / "analysis" / "paper_understanding.json"
        if not path.exists() and legacy_path.exists():
            understanding = load_research_understanding(workspace)
            from mypaper2code.services.analysis import write_research_understanding

            write_research_understanding(workspace, understanding)
            return understanding
        if not path.exists():
            MethodAnalyzer().understand(workspace)
        if path.exists():
            return load_model(path, ResearchUnderstanding)
        return load_research_understanding(workspace)

    @staticmethod
    def write(workspace: Path, plan: ResearchImplementationPlan) -> None:
        plan_dir = workspace / "plan"
        analysis_dir = workspace / "analysis"
        write_yaml(plan_dir / "requirements.yaml", plan.requirements.model_dump(mode="json"))
        write_json(plan_dir / "research_plan.json", plan.model_dump(mode="json"))
        write_yaml(analysis_dir / "requirements.yaml", plan.requirements.model_dump(mode="json"))
        legacy = _legacy_plan(plan)
        write_json(analysis_dir / "implementation_plan.json", legacy.model_dump(mode="json"))
        lines = ["# Implementation Plan", "", "## Requirements", ""]
        for key, value in plan.requirements.model_dump().items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Paper Understanding", ""])
        lines.append(f"- `paper_type`: {plan.paper_type}")
        lines.append(
            f"- `contributions`: {', '.join(plan.understanding.contributions) or 'unspecified'}"
        )
        lines.append(f"- `algorithms`: {', '.join(plan.understanding.algorithms) or 'unspecified'}")
        lines.append(f"- `datasets`: {', '.join(plan.understanding.datasets) or 'unspecified'}")
        lines.append(f"- `metrics`: {', '.join(plan.understanding.metrics) or 'unspecified'}")
        if plan.blocking_decisions:
            lines.extend(["", "## Blocking Decisions", ""])
            lines.extend(f"- `{item}`" for item in plan.blocking_decisions)
        lines.extend(["", "## Implementation Steps", ""])
        for step in plan.steps:
            lines.append(f"### {step.step_id}. {step.title}")
            lines.append(f"- Purpose: {step.purpose}")
            lines.append(f"- Files: {', '.join(f'`{file}`' for file in step.target_files)}")
            if step.acceptance_criteria:
                lines.append(f"- Acceptance: {'; '.join(step.acceptance_criteria)}")
            if step.risks:
                lines.append(f"- Risks: {'; '.join(step.risks)}")
            lines.append("")
        lines.extend(["", "## Assumptions", ""])
        lines.extend(f"- {item}" for item in plan.assumptions)
        (plan_dir / "implementation_plan.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        (analysis_dir / "implementation_plan.md").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _agentic_steps_for(
        understanding: ResearchUnderstanding,
        requirements: ImplementationRequirements,
    ) -> list[AgenticImplementationStep]:
        evidence = understanding.evidence[:5]
        steps = [
            AgenticImplementationStep(
                step_id="project_scaffold",
                title="Create reproducible project scaffold",
                purpose="Create configs, README, dependency notes, and a paper evidence manifest.",
                target_files=["README.md", "configs/default.yaml", "evidence/manifest.json"],
                acceptance_criteria=[
                    "The generated project documents paper type and unresolved assumptions.",
                    "The config includes datasets, metrics, protocols, and hyperparameters.",
                ],
                evidence=evidence,
                risks=["Paper details may be incomplete if extraction missed tables or equations."],
            ),
            AgenticImplementationStep(
                step_id="method_core",
                title="Implement method core",
                purpose="Implement the main algorithmic interface described by the paper.",
                target_files=["src/method/core.py"],
                acceptance_criteria=[
                    "The module exposes a MethodCore class with deterministic toy execution.",
                    "Evidence references are preserved in comments and trace metadata.",
                ],
                evidence=evidence,
                risks=[
                    "Complex math may require a human-confirmed derivation before full fidelity."
                ],
            ),
            AgenticImplementationStep(
                step_id="experiment_protocol",
                title="Implement experiment protocol",
                purpose=(
                    "Wire datasets, metrics, hyperparameters, and protocol into runnable scripts."
                ),
                target_files=[
                    "src/experiments/protocol.py",
                    "scripts/run_experiment.py",
                    "tests/test_contract.py",
                ],
                acceptance_criteria=[
                    "Smoke execution runs without external datasets.",
                    "Contract tests check method output shape and metric types.",
                ],
                evidence=evidence,
                risks=["Real reproduction requires real datasets and compute resources."],
            ),
        ]
        if understanding.paper_type == "ml":
            steps.insert(
                2,
                AgenticImplementationStep(
                    step_id="ml_components",
                    title="Implement ML-specific components",
                    purpose="Provide a PyTorch-ready adapter when the paper is machine learning.",
                    target_files=["src/method/ml_adapter.py"],
                    dependencies=["torch"],
                    acceptance_criteria=[
                        "Adapter is optional and the core method remains testable without "
                        "training.",
                    ],
                    evidence=evidence,
                    risks=["Architecture details may still need human review."],
                ),
            )
        if requirements.include_tests:
            steps.append(
                AgenticImplementationStep(
                    step_id="validation_assets",
                    title="Add validation assets",
                    purpose=(
                        "Add smoke and contract tests generated from the plan acceptance criteria."
                    ),
                    target_files=["tests/test_smoke.py", "tests/test_contract.py"],
                    acceptance_criteria=[
                        "Tests cover imports, config loading, method execution, and trace "
                        "presence.",
                    ],
                    evidence=evidence,
                )
            )
        return steps

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
        understanding: ResearchUnderstanding,
        requirements: ImplementationRequirements,
    ) -> list[str]:
        assumptions = [item.question for item in understanding.ambiguities]
        assumptions.append(
            f"Implementation follows user requirements: framework={requirements.framework}, "
            f"dataset={requirements.dataset}, style={requirements.style}."
        )
        assumptions.append(
            "Each implementation step must record trace entries linking evidence to files."
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


def _legacy_plan(plan: ResearchImplementationPlan) -> ImplementationPlan:
    understanding = legacy_understanding(plan.understanding)
    return ImplementationPlan(
        requirements=plan.requirements,
        understanding=understanding,
        steps=[
            ImplementationStep(
                step_id=step.step_id,
                title=step.title,
                files=step.target_files,
                purpose=step.purpose,
                source_refs=[
                    source
                    for source in understanding.sources
                    if source.chunk_id in {evidence.evidence_id for evidence in step.evidence}
                ],
                assumptions=step.risks,
            )
            for step in plan.steps
        ],
        assumptions=plan.assumptions,
        sources=understanding.sources,
    )
