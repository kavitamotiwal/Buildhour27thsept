from __future__ import annotations

import pytest

from ragchat.chunker import chunk_segments
from ragchat.config import CONFIG
from ragchat.errors import IngestError
from ragchat.loaders import discover, load

DOCS_DIR = CONFIG.docs_path


def test_discovers_sample_corpus():
    names = {p.name for p in discover(DOCS_DIR)}
    assert names == {
        "hdfc-large-cap-fund-direct-growth.md",
        "hdfc-flexi-cap-fund-direct-growth.md",
        "hdfc-elss-tax-saver-fund-direct-plan-growth.md",
        "hdfc-small-cap-fund-direct-growth.md",
        "hdfc-balanced-advantage-fund-direct-growth.md",
    }


def test_markdown_splits_on_headings_with_section_anchors():
    segments = load(DOCS_DIR / "hdfc-large-cap-fund-direct-growth.md", root=DOCS_DIR)
    sections = [meta["section"] for _, meta in segments]
    assert "Expense Ratio" in sections
    assert "Exit Load" in sections
    assert "Riskometer" in sections
    assert all(meta["source_file"] == "hdfc-large-cap-fund-direct-growth.md" for _, meta in segments)
    assert all("page" not in meta for _, meta in segments)


def test_markdown_segment_includes_its_heading():
    segments = load(DOCS_DIR / "hdfc-large-cap-fund-direct-growth.md", root=DOCS_DIR)
    text, meta = next((t, m) for t, m in segments if m["section"] == "Expense Ratio")
    assert text.startswith("Expense Ratio")


def test_front_matter_becomes_metadata_not_content():
    segments = load(DOCS_DIR / "hdfc-large-cap-fund-direct-growth.md", root=DOCS_DIR)
    assert all(
        meta["source_url"] == "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
        and meta["scheme"] == "HDFC Large Cap Fund - Direct Growth"
        and meta["fetch_date"] == "2026-09-29"
        for _, meta in segments
    )
    assert all("source_url" not in text.lower() and "fetch_date" not in text.lower() for text, _ in segments)


def test_text_file_is_one_segment(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("just one block of text", encoding="utf-8")
    segments = load(target)
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
