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


class ImplementationPlan(BaseModel):
    requirements: ImplementationRequirements
    understanding: PaperUnderstanding
    steps: list[ImplementationStep]
    assumptions: list[str]
    sources: list[SourceSpan] = Field(default_factory=list)


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


class ImplementationTrace(BaseModel):
    entries: list[ImplementationTraceEntry] = Field(default_factory=list)


class ValidationResult(BaseModel):
    name: str
    command: str
    returncode: int
    log_path: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def path_text(path: Path) -> str:
    return str(path.as_posix())
