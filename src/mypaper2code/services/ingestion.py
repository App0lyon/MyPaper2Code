from __future__ import annotations

import re
from pathlib import Path

from mypaper2code.core.io import write_json
from mypaper2code.core.models import PaperChunk, PaperSection
from mypaper2code.core.text import normalize_space
from mypaper2code.services.search.hybrid import HybridRetriever
from mypaper2code.services.workspace import WorkspaceManager

SECTION_PATTERNS = {
    "abstract": re.compile(r"^\s*abstract\s*$", re.IGNORECASE),
    "introduction": re.compile(r"^\s*(\d+\.?\s*)?introduction\s*$", re.IGNORECASE),
    "related_work": re.compile(r"^\s*(\d+\.?\s*)?related work\s*$", re.IGNORECASE),
    "method": re.compile(
        r"^\s*(\d+\.?\s*)?(method|methodology|approach|model|proposed method)\s*$",
        re.IGNORECASE,
    ),
    "experiments": re.compile(r"^\s*(\d+\.?\s*)?(experiments|experimental setup)\s*$", re.I),
    "results": re.compile(r"^\s*(\d+\.?\s*)?(results|evaluation)\s*$", re.IGNORECASE),
    "appendix": re.compile(r"^\s*(appendix|supplementary material)\s*$", re.IGNORECASE),
    "references": re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE),
}


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("PyMuPDF is required to ingest PDF files. Run `uv sync`.") from exc

    doc = fitz.open(pdf_path)
    try:
        return [(page.number + 1, page.get_text("text")) for page in doc]
    finally:
        doc.close()


def detect_sections(pages: list[tuple[int, str]], paper_id: str) -> list[PaperSection]:
    del paper_id
    sections: list[PaperSection] = []
    current_name = "unknown"
    current_title = "Unknown"
    current_start = pages[0][0] if pages else 1
    current_text: list[str] = []

    def flush(end_page: int) -> None:
        if current_text:
            sections.append(
                PaperSection(
                    name=current_name,
                    title=current_title,
                    page_start=current_start,
                    page_end=end_page,
                    text=normalize_space("\n".join(current_text)),
                )
            )

    for page_number, page_text in pages:
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            matched = next(
                (name for name, pattern in SECTION_PATTERNS.items() if pattern.match(line)),
                None,
            )
            if matched:
                flush(page_number)
                current_name = matched
                current_title = line
                current_start = page_number
                current_text = []
            elif line:
                current_text.append(line)

    if pages:
        flush(pages[-1][0])
    return sections or [
        PaperSection(
            name="full_text",
            title="Full text",
            page_start=page,
            page_end=page,
            text=normalize_space(text),
        )
        for page, text in pages
        if normalize_space(text)
    ]


def chunk_sections(
    sections: list[PaperSection],
    paper_id: str,
    max_words: int = 180,
    overlap_words: int = 35,
) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    for section in sections:
        words = section.text.split()
        if not words:
            continue
        step = max(1, max_words - overlap_words)
        for start in range(0, len(words), step):
            window = words[start : start + max_words]
            if not window:
                continue
            chunk_id = f"{paper_id}-{len(chunks):05d}"
            chunks.append(
                PaperChunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    section=section.name,
                    page=section.page_start,
                    text=" ".join(window),
                )
            )
            if start + max_words >= len(words):
                break
    return chunks


class IngestionService:
    def __init__(self, workspace_manager: WorkspaceManager | None = None) -> None:
        self.workspace_manager = workspace_manager or WorkspaceManager()

    def ingest(
        self,
        pdf_path: Path,
        provider: str = "nvidia",
        model: str = "mistralai/mistral-medium-3.5-128b",
    ) -> Path:
        pdf_path = pdf_path.resolve()
        pages = extract_pdf_pages(pdf_path)
        title = _guess_title(pdf_path, pages)
        workspace = self.workspace_manager.create(
            pdf_path,
            title=title,
            provider=provider,
            model=model,
        )
        metadata = self.workspace_manager.load_metadata(workspace)
        sections = detect_sections(pages, metadata.paper.paper_id)
        chunks = chunk_sections(sections, metadata.paper.paper_id)

        write_json(
            workspace / "paper" / "extracted_sections.json",
            [s.model_dump() for s in sections],
        )
        write_json(workspace / "paper" / "chunks.json", [c.model_dump() for c in chunks])
        HybridRetriever.build(workspace, chunks)
        return workspace


def _guess_title(pdf_path: Path, pages: list[tuple[int, str]]) -> str:
    if not pages:
        return pdf_path.stem
    for line in pages[0][1].splitlines():
        cleaned = normalize_space(line)
        if 8 <= len(cleaned) <= 160 and not cleaned.lower().startswith(("abstract", "arxiv")):
            return cleaned
    return pdf_path.stem
