from __future__ import annotations

import json
from pathlib import Path

from mypaper2code.core.io import read_json, write_json
from mypaper2code.core.models import PaperChunk, PaperUnderstanding, SourceSpan
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
        understanding = build_understanding(workspace)
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
    retriever = HybridRetriever(workspace)
    hits = retriever.search("method architecture loss training dataset evaluation metric", limit=8)
    return _llm_understanding(workspace, hits)


def _llm_understanding(workspace: Path, hits) -> PaperUnderstanding:
    metadata = WorkspaceManager.load_metadata(workspace)
    config = load_config()
    requirements = load_requirements(workspace)
    provider_name = _first_provider(
        requirements.provider,
        metadata.provider,
        config["provider"],
    )
    sources = [
        SourceSpan(section=hit.section, page=hit.page, text=hit.text, chunk_id=hit.chunk_id)
        for hit in hits
    ]
    source_text = "\n\n".join(
        f"[{idx}] page={source.page} section={source.section}\n{source.text}"
        for idx, source in enumerate(sources, start=1)
    )
    prompt = f"""Extract implementation-relevant facts from these paper passages.
Return strict JSON with exactly these keys:
architecture: string
loss: string
datasets: array of strings
metrics: array of strings
training: array of strings
ambiguities: array of strings

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
            return PaperUnderstanding(
                architecture=str(data.get("architecture") or "unspecified"),
                loss=str(data.get("loss") or "unspecified"),
                datasets=[str(item) for item in data.get("datasets", [])],
                metrics=[str(item) for item in data.get("metrics", [])],
                training=[str(item) for item in data.get("training", [])],
                ambiguities=[str(item) for item in data.get("ambiguities", [])],
                sources=sources,
            )
        except (ProviderError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate}: {exc}")
            continue
    raise ProviderError(
        "No LLM provider is available for paper analysis. Tried "
        f"{', '.join(_provider_order(provider_name))}. Errors: {'; '.join(errors)}"
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
    if "cross entropy" in text or "cross-entropy" in text:
        return "cross_entropy"
    if "mse" in text or "mean squared" in text:
        return "mse"
    if "triplet" in text:
        return "triplet"
    return "unspecified"


def _detect_any(text: str, candidates: list[str]) -> list[str]:
    return [candidate for candidate in candidates if candidate in text]
