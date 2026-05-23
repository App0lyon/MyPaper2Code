from __future__ import annotations

from pathlib import Path

from mypaper2code.core.io import read_json, write_json
from mypaper2code.core.models import PaperChunk, SearchHit
from mypaper2code.core.text import excerpt
from mypaper2code.services.search.rrf import reciprocal_rank_fusion
from mypaper2code.services.search.tantivy_index import TantivyIndex
from mypaper2code.services.search.vector_index import VectorIndex


class HybridRetriever:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.tantivy = TantivyIndex(workspace)
        self.vector = VectorIndex(workspace)

    @classmethod
    def build(cls, workspace: Path, chunks: list[PaperChunk]) -> None:
        retriever = cls(workspace)
        retriever.tantivy.build(chunks)
        retriever.vector.build(chunks)
        write_json(
            workspace / "paper" / "retrieval_config.json",
            {
                "lexical": "tantivy",
                "vector": "numpy-exact",
                "fusion": "rrf",
                "rrf_k": 60,
            },
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        top_k_lexical: int = 12,
        top_k_vector: int = 12,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        lexical_hits = self.tantivy.search(query, top_k=top_k_lexical)
        vector_hits = self.vector.search(query, top_k=top_k_vector)
        fused = reciprocal_rank_fusion([lexical_hits, vector_hits], rrf_k=rrf_k, limit=limit)
        return [hit.model_copy(update={"text": excerpt(hit.text)}) for hit in fused]

    @staticmethod
    def load_chunks(workspace: Path) -> list[PaperChunk]:
        return [
            PaperChunk.model_validate(item)
            for item in read_json(workspace / "paper" / "chunks.json")
        ]
