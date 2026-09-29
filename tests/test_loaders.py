from __future__ import annotations

import pytest

from ragchat.chunker import chunk_segments
from ragchat.config import CONFIG
from ragchat.errors import IngestError
from ragchat.loaders import discover, load

DOCS_DIR = CONFIG.docs_path


def test_discovers_sample_corpus():
    names = {p.name for p in discover(DOCS_DIR)}
    assert {"week1_intro_to_rag.md", "week2_vector_stores.txt", "week3_prompting.pdf"} <= names


def test_markdown_splits_on_headings_with_section_anchors():
    segments = load(DOCS_DIR / "week1_intro_to_rag.md", root=DOCS_DIR)
    sections = [meta["section"] for _, meta in segments]
    assert "What RAG Is" in sections
    assert "Chunking" in sections
    assert all(meta["source_file"] == "week1_intro_to_rag.md" for _, meta in segments)
    assert all("page" not in meta for _, meta in segments)


def test_markdown_segment_includes_its_heading():
    segments = load(DOCS_DIR / "week1_intro_to_rag.md", root=DOCS_DIR)
    text, meta = next((t, m) for t, m in segments if m["section"] == "Chunking")
    assert text.startswith("Chunking")


def test_pdf_yields_one_segment_per_page():
    segments = load(DOCS_DIR / "week3_prompting.pdf", root=DOCS_DIR)
    assert [meta["page"] for _, meta in segments] == [1, 2]
    assert all("section" not in meta for _, meta in segments)
    assert all(text.strip() for text, _ in segments)


def test_text_file_is_one_segment():
    segments = load(DOCS_DIR / "week2_vector_stores.txt", root=DOCS_DIR)
    assert len(segments) == 1
    assert segments[0][1]["section"]


def test_unsupported_extension_message_names_file(tmp_path):
    target = tmp_path / "notes.docx"
    target.write_text("content", encoding="utf-8")
    with pytest.raises(IngestError) as excinfo:
        load(target)
    assert "notes.docx" in str(excinfo.value)


def test_empty_text_file_raises(tmp_path):
    target = tmp_path / "empty.txt"
    target.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(IngestError) as excinfo:
        load(target)
    assert "empty.txt" in str(excinfo.value)


def test_every_corpus_document_chunks_successfully():
    for path in discover(DOCS_DIR):
        segments = load(path, root=DOCS_DIR)
        chunks = chunk_segments(segments)
        assert chunks, f"{path.name} produced no chunks"
        assert all(c.text.strip() for c in chunks)
        assert all(c.metadata["source_file"] for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)
