from __future__ import annotations

from pathlib import Path
from typing import Any

from mypaper2code.core.io import read_json, write_json
from mypaper2code.core.models import PaperChunk, SearchHit
from mypaper2code.core.text import tokenize


class TantivyIndex:
    def __init__(self, workspace: Path) -> None:
        self.paper_dir = workspace / "paper"
        self.index_dir = self.paper_dir / "tantivy_index"
        self.native_dir = self.index_dir / "native"
        self.fallback_path = self.index_dir / "fallback_documents.json"

    def build(self, chunks: list[PaperChunk]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.fallback_path, [chunk.model_dump() for chunk in chunks])
        try:
            import tantivy

            schema_builder = tantivy.SchemaBuilder()
            schema_builder.add_text_field("chunk_id", stored=True)
            schema_builder.add_text_field("paper_id", stored=True)
            schema_builder.add_text_field("section", stored=True)
            schema_builder.add_integer_field("page", stored=True)
            schema_builder.add_text_field("text", stored=True)
            schema = schema_builder.build()
            self.native_dir.mkdir(parents=True, exist_ok=True)
            index = tantivy.Index(schema, path=str(self.native_dir))
            writer = index.writer()
            for chunk in chunks:
                writer.add_document(
                    tantivy.Document(
                        chunk_id=[chunk.chunk_id],
                        paper_id=[chunk.paper_id],
                        section=[chunk.section],
                        page=[chunk.page],
                        text=[chunk.text],
                    )
                )
            writer.commit()
        except Exception:
            return

    def search(self, query: str, top_k: int = 8) -> list[SearchHit]:
        try:
            hits = self._search_tantivy(query, top_k=top_k)
            if hits:
                return hits
        except Exception:
            pass
        return self._search_fallback(query, top_k=top_k)

    def _search_tantivy(self, query: str, top_k: int) -> list[SearchHit]:
        import tantivy

        index = tantivy.Index.open(str(self.native_dir))
        searcher = index.searcher()
        parsed = index.parse_query(query, ["text", "section"])
        raw_hits = searcher.search(parsed, top_k).hits
        hits: list[SearchHit] = []
        for rank, item in enumerate(raw_hits, start=1):
            score, doc_address = _split_tantivy_hit(item)
            doc = searcher.doc(doc_address)
            fields = _document_to_mapping(doc)
            hits.append(
                SearchHit(
                    chunk_id=_first(fields["chunk_id"]),
                    section=_first(fields["section"]),
                    page=int(_first(fields["page"])),
                    text=_first(fields["text"]),
                    score=float(score),
                    lexical_rank=rank,
                )
            )
        return hits

    def _search_fallback(self, query: str, top_k: int) -> list[SearchHit]:
        if not self.fallback_path.exists():
            return []
        chunks = [PaperChunk.model_validate(item) for item in read_json(self.fallback_path)]
        query_terms = set(tokenize(query))
        scored: list[tuple[float, PaperChunk]] = []
        for chunk in chunks:
            terms = tokenize(chunk.text + " " + chunk.section)
            if not terms:
                continue
            overlap = sum(1 for term in terms if term in query_terms)
            score = overlap / max(len(terms), 1)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                chunk_id=chunk.chunk_id,
                section=chunk.section,
                page=chunk.page,
                text=chunk.text,
                score=score,
                lexical_rank=rank,
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


def _split_tantivy_hit(item: Any) -> tuple[float, Any]:
    if isinstance(item, tuple) and len(item) == 2:
        return float(item[0]), item[1]
    return float(item.score), item.doc_address


def _document_to_mapping(doc: Any) -> dict[str, Any]:
    if isinstance(doc, dict):
        return doc
    if hasattr(doc, "to_dict"):
        return doc.to_dict()
    return dict(doc)


def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0]
    return value
