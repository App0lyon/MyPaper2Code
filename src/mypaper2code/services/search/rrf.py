from __future__ import annotations

from collections.abc import Iterable

from mypaper2code.core.models import SearchHit


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[SearchHit]],
    rrf_k: int = 60,
    limit: int = 8,
) -> list[SearchHit]:
    fused: dict[str, SearchHit] = {}

    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            existing = fused.get(hit.chunk_id)
            contribution = 1.0 / (rrf_k + rank)
            if existing is None:
                updated = hit.model_copy(update={"score": contribution})
                fused[hit.chunk_id] = updated
            else:
                existing.score += contribution
                existing.lexical_rank = existing.lexical_rank or hit.lexical_rank
                existing.vector_rank = existing.vector_rank or hit.vector_rank

    return sorted(fused.values(), key=lambda item: item.score, reverse=True)[:limit]
