from __future__ import annotations

from pathlib import Path

from mypaper2code.core.io import read_json, write_json
from mypaper2code.core.models import PaperChunk, PaperUnderstanding, SourceSpan
from mypaper2code.core.text import excerpt
from mypaper2code.services.search.hybrid import HybridRetriever

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
    corpus = " ".join(hit.text.lower() for hit in hits)
    architecture = _detect_architecture(corpus)
    loss = _detect_loss(corpus)
    datasets = _detect_any(
        corpus,
        ["cifar-10", "cifar10", "imagenet", "mnist", "coco", "wikitext"],
    )
    metrics = _detect_any(corpus, ["accuracy", "f1", "auc", "precision", "recall", "bleu", "rouge"])
    training = _detect_any(
        corpus,
        ["adamw", "adam", "sgd", "cosine", "scheduler", "batch", "epoch"],
    )
    sources = [
        SourceSpan(section=hit.section, page=hit.page, text=hit.text, chunk_id=hit.chunk_id)
        for hit in hits
    ]
    ambiguities: list[str] = []
    if architecture == "unspecified":
        ambiguities.append("Architecture not found explicitly in retrieved passages.")
    if loss == "unspecified":
        ambiguities.append("Loss function not found explicitly in retrieved passages.")
    if not datasets:
        ambiguities.append("Dataset names not found explicitly in retrieved passages.")
    if not metrics:
        ambiguities.append("Evaluation metrics not found explicitly in retrieved passages.")
    return PaperUnderstanding(
        architecture=architecture,
        loss=loss,
        datasets=datasets,
        metrics=metrics,
        training=training,
        ambiguities=ambiguities,
        sources=sources,
    )


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
