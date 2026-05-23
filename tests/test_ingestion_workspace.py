from __future__ import annotations

from pathlib import Path

import fitz

from mypaper2code.core.io import read_json
from mypaper2code.services.ingestion import (
    IngestionService,
    chunk_sections,
    detect_sections,
    extract_pdf_pages,
)
from mypaper2code.services.workspace import WorkspaceManager


def make_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "A Small Paper\nAbstract\nThis paper proposes a simple model.\n"
        "Method\nThe architecture uses a neural network and cross entropy loss.\n"
        "Experiments\nWe evaluate on CIFAR-10.",
    )
    doc.save(path)
    doc.close()


def test_extract_sections_and_chunks(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf)

    pages = extract_pdf_pages(pdf)
    sections = detect_sections(pages, "paper")
    chunks = chunk_sections(sections, "paper", max_words=20, overlap_words=5)

    assert pages
    assert any(section.name == "method" for section in sections)
    assert chunks


def test_ingestion_creates_workspace(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf)
    manager = WorkspaceManager(base_dir=tmp_path / "workspaces")

    workspace = IngestionService(manager).ingest(pdf)

    assert (workspace / "metadata.json").exists()
    assert (workspace / "paper" / "original.pdf").exists()
    assert read_json(workspace / "paper" / "chunks.json")
