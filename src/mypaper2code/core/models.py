from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SourceSpan(BaseModel):
    section: str
    page: int
    text: str
    chunk_id: str | None = None


EvidenceKind = Literal[
    "page",
    "chunk",
    "section",
    "table",
    "figure",
    "equation",
    "algorithm",
    "appendix",
    "decision",
]


class EvidenceRef(BaseModel):
    evidence_id: str
    kind: EvidenceKind
    page: int | None = None
    section: str | None = None
    label: str | None = None
    text: str = ""
    source_path: str | None = None
    confidence: float = 1.0


class PaperSection(BaseModel):
    name: str
    title: str
    page_start: int
    page_end: int
    text: str


class PaperChunk(BaseModel):
    chunk_id: str
    paper_id: str
    section: str
    page: int
    text: str


class PaperPageArtifact(BaseModel):
    evidence_id: str
    page: int
    text: str


class PaperTableArtifact(BaseModel):
    evidence_id: str
    page: int
    label: str
    text: str


class PaperFigureArtifact(BaseModel):
    evidence_id: str
    page: int
    label: str
    caption: str


class PaperEquationArtifact(BaseModel):
    evidence_id: str
    page: int
    label: str
    text: str


class PaperAlgorithmArtifact(BaseModel):
    evidence_id: str
    page: int
    label: str
    text: str


class SearchHit(BaseModel):
    chunk_id: str
    section: str
    page: int
    text: str
    score: float
    lexical_rank: int | None = None
    vector_rank: int | None = None


class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    source_path: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkspaceMetadata(BaseModel):
    workspace_id: str
    paper: PaperMetadata
    root: str
    provider: str = "nvidia"
    model: str = "mistralai/mistral-medium-3.5-128b"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ImplementationRequirements(BaseModel):
    framework: str = "pytorch"
    dataset: str = "cifar10"
    style: str = "research"
    config_format: Literal["yaml"] = "yaml"
    target_gpu_memory: str | None = None
    implementation_level: str = "minimal"
    include_tests: bool = True
    include_training_script: bool = True
    include_evaluation_script: bool = True
    provider: str = "nvidia"
    model: str = "mistralai/mistral-medium-3.5-128b"


class ResearchAmbiguity(BaseModel):
    ambiguity_id: str
    question: str
    severity: Literal["blocking", "non_blocking"] = "non_blocking"
    evidence: list[EvidenceRef] = Field(default_factory=list)
    recommendation: str | None = None
    status: Literal["open", "answered", "approved"] = "open"
    answer: str | None = None


class UserDecision(BaseModel):
    decision_id: str
    value: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchUnderstanding(BaseModel):
    schema_version: str = "2.0"
    paper_type: Literal[
        "ml",
        "classical_algorithm",
        "simulation",
        "statistics",
        "optimization",
        "systems",
        "other",
    ] = "other"
    contributions: list[str] = Field(default_factory=list)
    definitions: list[str] = Field(default_factory=list)
    algorithms: list[str] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    protocols: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, str] = Field(default_factory=dict)
    resources_required: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    ambiguities: list[ResearchAmbiguity] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None


class PaperUnderstanding(BaseModel):
    architecture: str = "unspecified"
    loss: str = "unspecified"
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    training: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    sources: list[SourceSpan] = Field(default_factory=list)


class ImplementationStep(BaseModel):
    step_id: str
    title: str
    files: list[str]
    purpose: str
    symbols: list[str] = Field(default_factory=list)
    source_refs: list[SourceSpan] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class AgenticImplementationStep(BaseModel):
    step_id: str
    title: str
    purpose: str
    target_files: list[str]
    dependencies: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    test_strategy: str = "smoke"
    evidence: list[EvidenceRef] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    requires_decisions: list[str] = Field(default_factory=list)


class ImplementationPlan(BaseModel):
    requirements: ImplementationRequirements
    understanding: PaperUnderstanding
    steps: list[ImplementationStep]
    assumptions: list[str]
    sources: list[SourceSpan] = Field(default_factory=list)


class ResearchImplementationPlan(BaseModel):
    schema_version: str = "2.0"
    requirements: ImplementationRequirements
    understanding: ResearchUnderstanding
    paper_type: str
    steps: list[AgenticImplementationStep]
    blocking_decisions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    approved: bool = False


class CodeLocation(BaseModel):
    file: str
    symbol: str | None = None
    line_start: int | None = None
    line_end: int | None = None


class ImplementationTraceEntry(BaseModel):
    step_id: str
    paper_claim: str
    source: SourceSpan | None = None
    implemented_in: list[CodeLocation]
    assumptions: list[str] = Field(default_factory=list)


class AgenticTraceEntry(BaseModel):
    step_id: str
    claim: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    implemented_in: list[CodeLocation]
    assumptions: list[str] = Field(default_factory=list)
    reviewer_notes: list[str] = Field(default_factory=list)


class ImplementationTrace(BaseModel):
    entries: list[ImplementationTraceEntry] = Field(default_factory=list)


class AgenticImplementationTrace(BaseModel):
    schema_version: str = "2.0"
    entries: list[AgenticTraceEntry] = Field(default_factory=list)


class ValidationResult(BaseModel):
    name: str
    command: str
    returncode: int
    log_path: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


class ValidationSuiteResult(BaseModel):
    level: Literal["smoke", "contract", "repro"] = "smoke"
    results: list[ValidationResult] = Field(default_factory=list)
    fidelity_score: Literal["blocked", "low", "medium", "high", "reproducible"] = "low"
    reasons: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)


def path_text(path: Path) -> str:
    return str(path.as_posix())
