from __future__ import annotations

from pathlib import Path

from mypaper2code.core.io import write_json
from mypaper2code.core.models import PaperChunk, SearchHit
from mypaper2code.services.search.hybrid import HybridRetriever
from mypaper2code.services.search.rrf import reciprocal_rank_fusion


def sample_chunks() -> list[PaperChunk]:
    return [
        PaperChunk(
            chunk_id="paper-00000",
            paper_id="paper",
            section="method",
            page=1,
            text="The proposed transformer architecture uses contrastive loss for image retrieval.",
        ),
        PaperChunk(
            chunk_id="paper-00001",
            paper_id="paper",
            section="experiments",
            page=2,
            text="Experiments are conducted on CIFAR-10 with accuracy as the main metric.",
        ),
    ]


def test_rrf_deduplicates_chunk_ids() -> None:
    lexical = [SearchHit(chunk_id="a", section="s", page=1, text="x", score=10, lexical_rank=1)]
    vector = [SearchHit(chunk_id="a", section="s", page=1, text="x", score=0.9, vector_rank=1)]

    fused = reciprocal_rank_fusion([lexical, vector])

    assert len(fused) == 1
    assert fused[0].chunk_id == "a"
    assert fused[0].lexical_rank == 1
    assert fused[0].vector_rank == 1


def test_hybrid_retriever_builds_and_searches(tmp_path: Path) -> None:
    workspace = tmp_path
    (workspace / "paper").mkdir()
    chunks = sample_chunks()
    write_json(workspace / "paper" / "chunks.json", [chunk.model_dump() for chunk in chunks])

    HybridRetriever.build(workspace, chunks)
    hits = HybridRetriever(workspace).search("contrastive loss architecture", limit=2)

    assert hits
    assert hits[0].chunk_id == "paper-00000"
    assert (workspace / "paper" / "tantivy_index").exists()
    assert (workspace / "paper" / "vectors.npy").exists()
    assert (workspace / "paper" / "vector_metadata.json").exists()
    assert (workspace / "paper" / "retrieval_config.json").exists()
