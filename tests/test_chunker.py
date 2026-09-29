from __future__ import annotations

import math

import pytest

from ragchat.chunker import _encode, chunk_segments
from ragchat.errors import IngestError

SOURCE = {"source_file": "week1.md", "section": "Embeddings"}


def _segments(body_tokens: int, section: str = "Embeddings", source_file: str = "week1.md"):
    text = " ".join(f"token{i}" for i in range(body_tokens))
    return [(text, {"source_file": source_file, "section": section})]


def test_no_chunk_exceeds_size():
    chunks = chunk_segments(_segments(1200), size=100, overlap=20)
    assert chunks
    assert all(len(_encode(c.text)) <= 100 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_segments(_segments(300), size=100, overlap=20)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text[-15:].split()[-1] in current.text


def test_window_count_matches_step_arithmetic():
    text, metadata = _segments(600)[0]
    total = len(_encode(text))
    step = 100 - 50
    chunks = chunk_segments([(text, metadata)], size=100, overlap=50)
    assert len(chunks) == math.ceil(total / step)


def test_no_chunk_crosses_segment_boundary():
    segments = [
        (" ".join(f"a{i}" for i in range(300)), {"source_file": "week1.md", "section": "Embeddings"}),
        (" ".join(f"b{i}" for i in range(300)), {"source_file": "week1.md", "section": "Retrieval"}),
    ]
    chunks = chunk_segments(segments, size=100, overlap=20)
    for chunk in chunks:
        body = chunk.text.split("\n", 1)[-1]
        prefix = "a" if chunk.metadata["section"] == "Embeddings" else "b"
        assert body.split()[0].startswith(prefix)


def test_ids_are_stable_across_runs():
    first = chunk_segments(_segments(500), size=100, overlap=20)
    second = chunk_segments(_segments(500), size=100, overlap=20)
    assert [c.id for c in first] == [c.id for c in second]


def test_ids_differ_per_section_and_index():
    chunks = chunk_segments(_segments(500), size=100, overlap=20)
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.id != chunk_segments(_segments(500, section="Retrieval"), size=100, overlap=20)[0].id for c in chunks[:1])


def test_metadata_always_populated():
    chunks = chunk_segments(_segments(500), size=100, overlap=20)
    for chunk in chunks:
        assert chunk.metadata["source_file"] == "week1.md"
        assert chunk.metadata["section"] == "Embeddings"
        assert isinstance(chunk.metadata["chunk_index"], int)


def test_no_empty_chunk_text():
    chunks = chunk_segments(_segments(500), size=100, overlap=20)
    assert all(chunk.text.strip() for chunk in chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(IngestError):
        chunk_segments(_segments(500), size=100, overlap=100)
    with pytest.raises(IngestError):
        chunk_segments(_segments(500), size=100, overlap=-1)


def test_page_anchor_uses_page_metadata():
    segments = [(" ".join(f"p{i}" for i in range(300)), {"source_file": "notes.pdf", "page": 2})]
    chunks = chunk_segments(segments, size=100, overlap=20)
    assert all(c.metadata["page"] == 2 for c in chunks)
    assert all("section" not in c.metadata for c in chunks)
    assert all(c.anchor == "p.2" for c in chunks)
