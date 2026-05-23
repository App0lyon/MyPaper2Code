from __future__ import annotations

import json
from pathlib import Path

from mypaper2code.core.io import read_json, write_json
from mypaper2code.core.models import (
    EvidenceRef,
    PaperChunk,
    PaperUnderstanding,
    ResearchAmbiguity,
    ResearchUnderstanding,
    SourceSpan,
)
from mypaper2code.core.text import excerpt
from mypaper2code.providers.base import (
    NVIDIA_DEFAULT_MODEL,
    OLLAMA_DEFAULT_MODEL,
    ProviderError,
    provider_for,
)
from mypaper2code.services.config import load_config
from mypaper2code.services.requirements import load_requirements
from mypaper2code.services.search.hybrid import HybridRetriever
from mypaper2code.services.workspace import WorkspaceManager

ANALYSIS_QUERIES = {
    "Architecture": "model architecture proposed method modules network",
    "Loss": "loss objective optimization training criterion",
    "Datasets": "dataset datasets benchmark data experiments",
    "Metrics": "metrics evaluation accuracy f1 auc bleu",
    "Training": "training optimizer learning rate batch size epochs scheduler",
}


class MethodAnalyzer:
    def understand(self, workspace: Path) -> ResearchUnderstanding:
        understanding = build_research_understanding(workspace)
        write_research_understanding(workspace, understanding)
        return understanding

    def analyze(self, workspace: Path) -> tuple[Path, Path]:
        chunks = [
            PaperChunk.model_validate(item)
            for item in read_json(workspace / "paper" / "chunks.json")
        ]
        retriever = HybridRetriever(workspace)
        lines = ["# Method Summary", ""]
        assumptions = ["# Assumptions and Uncertainties", ""]

        for label, query in ANALYSIS_QUERIES.items():
            hits = retriever.search(query, limit=3)
            lines.append(f"## {label}")
            if hits:
                for hit in hits:
                    lines.append(
                        f"- Page {hit.page}, section `{hit.section}`: "
                        f"{excerpt(hit.text, 260)}"
                    )
            else:
                lines.append("- Not found in the extracted paper text.")
                assumptions.append(f"- {label}: not found explicitly in the extracted text.")
            lines.append("")

        if chunks:
            assumptions.append(
                "- Exact implementation details may be incomplete; generated code uses "
                "explicit defaults."
            )
        research_understanding = build_research_understanding(workspace)
        write_research_understanding(workspace, research_understanding)
        understanding = legacy_understanding(research_understanding)
        summary_path = workspace / "analysis" / "method_summary.md"
        assumptions_path = workspace / "analysis" / "assumptions.md"
        write_json(
            workspace / "analysis" / "paper_understanding.json",
            understanding.model_dump(mode="json"),
        )
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        assumptions_path.write_text("\n".join(assumptions), encoding="utf-8")
        return summary_path, assumptions_path


def build_understanding(workspace: Path) -> PaperUnderstanding:
    return legacy_understanding(build_research_understanding(workspace))


def build_research_understanding(workspace: Path) -> ResearchUnderstanding:
    retriever = HybridRetriever(workspace)
    hits = retriever.search(
        "method algorithm equation dataset protocol metric hyperparameter implementation",
        limit=12,
    )
    return _llm_understanding(workspace, hits)


def _llm_understanding(workspace: Path, hits) -> ResearchUnderstanding:
    metadata = WorkspaceManager.load_metadata(workspace)
    config = load_config()
    requirements = load_requirements(workspace)
    provider_name = _first_provider(
        requirements.provider,
        metadata.provider,
        config["provider"],
    )
    evidence = _evidence_for_hits(hits)
    supplemental_evidence = _load_multimodal_evidence(workspace)
    evidence.extend(supplemental_evidence[:12])
    source_text = "\n\n".join(
        f"[{idx}] id={source.evidence_id} kind={source.kind} page={source.page} "
        f"section={source.section}\n{source.text}"
        for idx, source in enumerate(evidence, start=1)
    )
    prompt = f"""Extract implementation-relevant facts from these paper passages.
Return strict JSON with exactly these keys:
paper_type: one of ["ml", "classical_algorithm", "simulation", "statistics",
"optimization", "systems", "other"]
contributions: array of strings
definitions: array of strings
algorithms: array of strings
equations: array of strings
datasets: array of strings
protocols: array of strings
metrics: array of strings
hyperparameters: object mapping strings to strings
resources_required: array of strings
expected_artifacts: array of strings
ambiguities: array of objects with keys question, severity, recommendation

Use only facts supported by the passages.
Put missing implementation-critical details in ambiguities.

Passages:
{source_text}
"""
    errors: list[str] = []
    for candidate in _provider_order(provider_name):
        model = _model_for_provider(candidate, requirements.model, metadata.model, config["model"])
        provider = provider_for(
            candidate,
            model,
            ollama_base_url=config["ollama_base_url"],
            nvidia_base_url=config["nvidia_base_url"],
            nvidia_api_key_env=config["nvidia_api_key_env"],
        )
        try:
            raw = provider.complete(
                prompt,
                system="You extract paper implementation details and return only valid JSON.",
            )
            data = _load_json_object(raw)
            understanding = _research_understanding_from_json(
                data,
                evidence=evidence,
                provider=candidate,
                model=model,
            )
            write_json(
                workspace / "understanding" / "provider_artifact.json",
                {
                    "provider": candidate,
                    "model": model,
                    "raw_response": raw,
                },
            )
            return understanding
        except (ProviderError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate}: {exc}")
            continue
    raise ProviderError(
        "No LLM provider is available for paper analysis. Tried "
        f"{', '.join(_provider_order(provider_name))}. Errors: {'; '.join(errors)}"
    )


def write_research_understanding(workspace: Path, understanding: ResearchUnderstanding) -> None:
    write_json(
        workspace / "understanding" / "research_understanding.json",
        understanding.model_dump(mode="json"),
    )
    lines = ["# Review Required", ""]
    blocking = [item for item in understanding.ambiguities if item.severity == "blocking"]
    non_blocking = [item for item in understanding.ambiguities if item.severity != "blocking"]
    if blocking:
        lines.extend(["## Blocking Questions", ""])
        lines.extend(f"- `{item.ambiguity_id}`: {item.question}" for item in blocking)
        lines.append("")
    if non_blocking:
        lines.extend(["## Non-Blocking Assumptions", ""])
        lines.extend(f"- `{item.ambiguity_id}`: {item.question}" for item in non_blocking)
        lines.append("")
    if not understanding.ambiguities:
        lines.append("- No ambiguities were extracted by the provider.")
    (workspace / "understanding" / "review.md").write_text("\n".join(lines), encoding="utf-8")


def load_research_understanding(workspace: Path) -> ResearchUnderstanding:
    path = workspace / "understanding" / "research_understanding.json"
    if path.exists():
        return ResearchUnderstanding.model_validate(read_json(path))
    legacy_path = workspace / "analysis" / "paper_understanding.json"
    if legacy_path.exists():
        return _research_understanding_from_json(read_json(legacy_path), evidence=[])
    return build_research_understanding(workspace)


def legacy_understanding(understanding: ResearchUnderstanding) -> PaperUnderstanding:
    architecture = _infer_architecture(understanding)
    loss = _infer_loss(understanding)
    return PaperUnderstanding(
        architecture=architecture,
        loss=loss,
        datasets=understanding.datasets,
        metrics=understanding.metrics,
        training=understanding.protocols + list(understanding.hyperparameters.values()),
        ambiguities=[item.question for item in understanding.ambiguities],
        sources=[
            SourceSpan(
                section=item.section or item.kind,
                page=item.page or 0,
                text=item.text,
                chunk_id=item.evidence_id,
            )
            for item in understanding.evidence
        ],
    )


def _research_understanding_from_json(
    data: dict,
    evidence: list[EvidenceRef],
    provider: str | None = None,
    model: str | None = None,
) -> ResearchUnderstanding:
    ambiguities = []
    for idx, item in enumerate(data.get("ambiguities", []), start=1):
        if isinstance(item, str):
            ambiguities.append(
                ResearchAmbiguity(
                    ambiguity_id=f"amb-{idx:03d}",
                    question=item,
                    severity="non_blocking",
                )
            )
        elif isinstance(item, dict):
            severity = str(item.get("severity") or "non_blocking")
            if severity not in {"blocking", "non_blocking"}:
                severity = "non_blocking"
            ambiguities.append(
                ResearchAmbiguity(
                    ambiguity_id=str(item.get("ambiguity_id") or f"amb-{idx:03d}"),
                    question=str(
                        item.get("question") or item.get("text") or "Unspecified ambiguity"
                    ),
                    severity=severity,
                    recommendation=(
                        str(item.get("recommendation")) if item.get("recommendation") else None
                    ),
                )
            )
    paper_type = str(data.get("paper_type") or _infer_paper_type(data))
    if paper_type not in {
        "ml",
        "classical_algorithm",
        "simulation",
        "statistics",
        "optimization",
        "systems",
        "other",
    }:
        paper_type = "other"
    hyperparameters = data.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        hyperparameters = {}
    return ResearchUnderstanding(
        paper_type=paper_type,
        contributions=_list_of_strings(data.get("contributions", [])),
        definitions=_legacy_seeded_list(data, "definitions", "architecture"),
        algorithms=_legacy_seeded_list(data, "algorithms", "architecture"),
        equations=_legacy_seeded_list(data, "equations", "loss"),
        datasets=_list_of_strings(data.get("datasets", [])),
        protocols=_list_of_strings(data.get("protocols", data.get("training", []))),
        metrics=_list_of_strings(data.get("metrics", [])),
        hyperparameters={str(key): str(value) for key, value in hyperparameters.items()},
        resources_required=_list_of_strings(data.get("resources_required", [])),
        expected_artifacts=_list_of_strings(data.get("expected_artifacts", [])),
        ambiguities=ambiguities,
        evidence=evidence,
        provider=provider,
        model=model,
    )


def _first_provider(*values: str) -> str:
    for value in values:
        normalized = value.lower()
        if normalized in {"nvidia", "ollama"}:
            return normalized
    return "nvidia"


def _provider_order(provider_name: str) -> list[str]:
    if provider_name == "nvidia":
        return ["nvidia", "ollama"]
    if provider_name == "ollama":
        return ["ollama"]
    raise ProviderError(f"Unsupported provider `{provider_name}`. Use `nvidia` or `ollama`.")


def _model_for_provider(provider_name: str, *values: str) -> str:
    for value in values:
        if not value or value == "stub":
            continue
        if provider_name == "ollama" and value == NVIDIA_DEFAULT_MODEL:
            continue
        if provider_name == "nvidia" and value == OLLAMA_DEFAULT_MODEL:
            continue
        return value
    return OLLAMA_DEFAULT_MODEL if provider_name == "ollama" else NVIDIA_DEFAULT_MODEL


def _load_json_object(raw: str) -> dict:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in provider response.")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise TypeError("Provider response must be a JSON object.")
    return parsed


def sources_for_plan(workspace: Path) -> list[SourceSpan]:
    retriever = HybridRetriever(workspace)
    hits = retriever.search("method architecture loss training dataset evaluation", limit=5)
    return [
        SourceSpan(section=hit.section, page=hit.page, text=hit.text, chunk_id=hit.chunk_id)
        for hit in hits
    ]


def _evidence_for_hits(hits) -> list[EvidenceRef]:
    return [
        EvidenceRef(
            evidence_id=hit.chunk_id,
            kind="chunk",
            page=hit.page,
            section=hit.section,
            text=hit.text,
            confidence=hit.score,
        )
        for hit in hits
    ]


def _load_multimodal_evidence(workspace: Path) -> list[EvidenceRef]:
    evidence: list[EvidenceRef] = []
    specs = [
        (workspace / "paper" / "tables" / "tables.json", "table", "text"),
        (workspace / "paper" / "figures" / "figures.json", "figure", "caption"),
        (workspace / "paper" / "equations" / "equations.json", "equation", "text"),
        (workspace / "paper" / "algorithms" / "algorithms.json", "algorithm", "text"),
    ]
    for path, kind, text_key in specs:
        if not path.exists():
            continue
        for item in read_json(path):
            evidence.append(
                EvidenceRef(
                    evidence_id=str(item["evidence_id"]),
                    kind=kind,
                    page=item.get("page"),
                    label=item.get("label"),
                    text=str(item.get(text_key) or ""),
                )
            )
    return evidence


def _list_of_strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _infer_paper_type(data: dict) -> str:
    values: list[str] = []
    for key in ("algorithms", "datasets", "metrics", "protocols", "training"):
        values.extend(_list_of_strings(data.get(key, [])))
    text = " ".join(values).lower()
    if any(token in text for token in ("dataset", "accuracy", "loss", "training", "neural")):
        return "ml"
    if any(token in text for token in ("simulate", "simulation")):
        return "simulation"
    if any(token in text for token in ("theorem", "estimator", "statistical")):
        return "statistics"
    return "other"


def _infer_architecture(understanding: ResearchUnderstanding) -> str:
    text = " ".join(
        understanding.algorithms + understanding.contributions + understanding.definitions
    ).lower()
    return _detect_architecture(text)


def _infer_loss(understanding: ResearchUnderstanding) -> str:
    text = " ".join(
        understanding.algorithms
        + understanding.contributions
        + understanding.equations
        + understanding.protocols
    ).lower()
    return _detect_loss(text)


def _detect_architecture(text: str) -> str:
    if "transformer" in text or "attention" in text:
        return "transformer"
    if "convolution" in text or "cnn" in text or "resnet" in text:
        return "cnn"
    if "diffusion" in text:
        return "diffusion"
    if "gan" in text or "adversarial" in text:
        return "gan"
    if "vae" in text or "variational autoencoder" in text:
        return "vae"
    if "mlp" in text or "feed-forward" in text or "neural network" in text:
        return "mlp"
    return "unspecified"


def _detect_loss(text: str) -> str:
    if "contrastive" in text:
        return "contrastive"
    if "cross entropy" in text or "cross-entropy" in text or "cross_entropy" in text:
        return "cross_entropy"
    if "mse" in text or "mean squared" in text:
        return "mse"
    if "triplet" in text:
        return "triplet"
    return "unspecified"


def _detect_any(text: str, candidates: list[str]) -> list[str]:
    return [candidate for candidate in candidates if candidate in text]


def _legacy_seeded_list(data: dict, key: str, legacy_key: str) -> list[str]:
    values = _list_of_strings(data.get(key, []))
    legacy_value = data.get(legacy_key)
    if legacy_value and str(legacy_value) != "unspecified":
        values.append(str(legacy_value))
    return values
