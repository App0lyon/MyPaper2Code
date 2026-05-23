from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from mypaper2code.core.io import read_json, write_json
from mypaper2code.core.models import PaperChunk, SearchHit
from mypaper2code.core.text import tokenize


class VectorIndex:
    def __init__(self, workspace: Path) -> None:
        self.paper_dir = workspace / "paper"
        self.vectors_path = self.paper_dir / "vectors.npy"
        self.metadata_path = self.paper_dir / "vector_metadata.json"

    def build(self, chunks: list[PaperChunk]) -> None:
        texts = [chunk.text for chunk in chunks]
        vectors, backend = embed_texts(texts)
        np.save(self.vectors_path, vectors.astype(np.float32))
        write_json(
            self.metadata_path,
            {
                "backend": backend,
                "chunks": [chunk.model_dump() for chunk in chunks],
            },
        )

    def search(self, query: str, top_k: int = 8) -> list[SearchHit]:
        if not self.vectors_path.exists() or not self.metadata_path.exists():
            return []
        vectors = np.load(self.vectors_path)
        metadata = read_json(self.metadata_path)
        chunks = [PaperChunk.model_validate(item) for item in metadata["chunks"]]
        query_vector, _ = embed_texts([query], backend_hint=metadata.get("backend"))
        scores = cosine_scores(vectors, query_vector[0])
        order = np.argsort(scores)[::-1][:top_k]
        hits: list[SearchHit] = []
        for rank, idx in enumerate(order, start=1):
            chunk = chunks[int(idx)]
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    section=chunk.section,
                    page=chunk.page,
                    text=chunk.text,
                    score=float(scores[int(idx)]),
                    vector_rank=rank,
                )
            )
        return hits


def embed_texts(texts: list[str], backend_hint: str | None = None) -> tuple[np.ndarray, str]:
    if backend_hint in (None, "sentence-transformers"):
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = model.encode(texts, normalize_embeddings=True)
            return np.asarray(embeddings, dtype=np.float32), "sentence-transformers"
        except Exception:
            if backend_hint == "sentence-transformers":
                pass
    vectors = np.vstack([hashing_embedding(text) for text in texts])
    return vectors.astype(np.float32), "hashing"


def hashing_embedding(text: str, dimensions: int = 384) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def cosine_scores(vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)
    denominator = np.maximum(vector_norms * query_norm, 1e-12)
    return (vectors @ query_vector) / denominator
