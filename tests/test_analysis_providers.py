from __future__ import annotations

from pathlib import Path

import pytest

from mypaper2code.core.io import read_json, write_json
from mypaper2code.core.models import PaperChunk, PaperMetadata, WorkspaceMetadata
from mypaper2code.providers.base import NvidiaProvider, OllamaProvider, ProviderError
from mypaper2code.services.analysis import MethodAnalyzer
from mypaper2code.services.search.hybrid import HybridRetriever


def make_workspace(path: Path) -> Path:
    (path / "paper").mkdir(parents=True)
    (path / "analysis").mkdir()
    (path / "generated_code").mkdir()
    (path / "runs").mkdir()
    metadata = WorkspaceMetadata(
        workspace_id="paper_20260101-000000",
        root=str(path),
        paper=PaperMetadata(
            paper_id="paper",
            title="Paper",
            source_path=str(path / "paper" / "original.pdf"),
        ),
        provider="nvidia",
    )
    write_json(path / "metadata.json", metadata.model_dump(mode="json"))
    chunks = [
        PaperChunk(
            chunk_id="paper-00000",
            paper_id="paper",
            section="method",
            page=1,
            text="The method uses a compact neural architecture with cross entropy loss.",
        )
    ]
    write_json(path / "paper" / "chunks.json", [chunk.model_dump() for chunk in chunks])
    HybridRetriever.build(path, chunks)
    return path


def test_analyze_falls_back_to_ollama_when_nvidia_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_workspace(tmp_path / "workspace")
    monkeypatch.setattr(
        NvidiaProvider,
        "complete",
        lambda *args, **kwargs: (_ for _ in ()).throw(ProviderError("nvidia unavailable")),
    )
    monkeypatch.setattr(
        OllamaProvider,
        "complete",
        lambda *args, **kwargs: (
            '{"architecture":"mlp","loss":"cross_entropy","datasets":["cifar10"],'
            '"metrics":["accuracy"],"training":[],"ambiguities":[]}'
        ),
    )

    MethodAnalyzer().analyze(workspace)

    understanding = read_json(workspace / "analysis" / "paper_understanding.json")
    assert understanding["architecture"] == "mlp"
    assert understanding["loss"] == "cross_entropy"


def test_analyze_crashes_when_nvidia_and_ollama_are_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_workspace(tmp_path / "workspace")
    monkeypatch.setattr(
        NvidiaProvider,
        "complete",
        lambda *args, **kwargs: (_ for _ in ()).throw(ProviderError("nvidia unavailable")),
    )
    monkeypatch.setattr(
        OllamaProvider,
        "complete",
        lambda *args, **kwargs: (_ for _ in ()).throw(ProviderError("ollama unavailable")),
    )

    with pytest.raises(ProviderError, match="No LLM provider is available"):
        MethodAnalyzer().analyze(workspace)
