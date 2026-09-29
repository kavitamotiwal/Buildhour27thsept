from __future__ import annotations

import json
import random

import pytest

from ragchat.chunker import chunk_id
from ragchat.config import CONFIG
from ragchat.ingest import MANIFEST_FIELDS, manifest_path, run_ingest
from ragchat.loaders import discover, source_hash
from ragchat.models import Chunk
from ragchat.vectorstore import VectorStore, normalize

DOCS_DIR = CONFIG.docs_path


class FakeEmbedder:
    """Deterministic, offline stand-in so the real ingest path is exercised without a key."""

    def __init__(self, dims: int = 16) -> None:
        self.dims = dims
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            rng = random.Random(text)
            vectors.append([rng.random() for _ in range(self.dims)])
        return vectors


@pytest.fixture()
def chroma_dir(tmp_path):
    return tmp_path / "chroma"


@pytest.fixture()
def manifest(chroma_dir):
    return run_ingest(FakeEmbedder(), docs_dir=DOCS_DIR, chroma_dir=chroma_dir, verbose=False)


def test_manifest_has_all_required_fields(manifest):
    assert set(MANIFEST_FIELDS) <= set(manifest)
    for field in MANIFEST_FIELDS:
        assert manifest[field] is not None, field


def test_manifest_chunk_count_matches_index(manifest, chroma_dir):
    store = VectorStore(path=chroma_dir)
    assert store.count() == manifest["chunk_count"]
    assert manifest["chunk_count"] > 0


def test_reingest_of_unchanged_corpus_is_identical(chroma_dir):
    first = run_ingest(FakeEmbedder(), docs_dir=DOCS_DIR, chroma_dir=chroma_dir, verbose=False)
    second = run_ingest(FakeEmbedder(), docs_dir=DOCS_DIR, chroma_dir=chroma_dir, verbose=False)
    assert first == second
    assert first["source_hash"] == source_hash(DOCS_DIR)


def test_reingest_resets_instead_of_appending(chroma_dir):
    run_ingest(FakeEmbedder(), docs_dir=DOCS_DIR, chroma_dir=chroma_dir, verbose=False)
    store = VectorStore(path=chroma_dir)
    before = store.count()
    run_ingest(FakeEmbedder(), docs_dir=DOCS_DIR, chroma_dir=chroma_dir, verbose=False)
    assert store.count() == before


def test_embedding_is_batched(manifest, chroma_dir):
    embedder = FakeEmbedder()
    run_ingest(embedder, docs_dir=DOCS_DIR, chroma_dir=chroma_dir, verbose=False)
    assert len(embedder.calls) >= 1
    assert sum(len(batch) for batch in embedder.calls) == manifest["chunk_count"]


def test_source_hash_changes_when_corpus_changes(tmp_path, chroma_dir):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    before = source_hash(docs)
    (docs / "a.txt").write_text("beta", encoding="utf-8")
    assert source_hash(docs) != before


def test_source_hash_is_content_based_not_mtime(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "a.txt"
    target.write_text("alpha", encoding="utf-8")
    before = source_hash(docs)
    import os

    os.utime(target, (0, 0))
    assert source_hash(docs) == before


def test_stored_chunks_carry_metadata(manifest, chroma_dir):
    store = VectorStore(path=chroma_dir)
    chunks = store.all_chunks()
    assert chunks
    for chunk in chunks:
        assert chunk.metadata.get("source_file")
        assert ("page" in chunk.metadata) or ("section" in chunk.metadata)
        assert "chunk_index" in chunk.metadata
        assert chunk.text.strip()


def test_manifest_file_written_atomically(manifest, chroma_dir):
    path = manifest_path(chroma_dir)
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["chunk_count"] == manifest["chunk_count"]
    assert not list(path.parent.glob("*.tmp"))


def test_query_returns_ranked_scored_chunks(manifest, chroma_dir):
    store = VectorStore(path=chroma_dir)
    hits = store.query([0.5] * 16, k=3)
    assert 0 < len(hits) <= 3
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.chunk.text.strip() for h in hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_query_on_empty_index_returns_empty(tmp_path):
    store = VectorStore(path=tmp_path / "empty")
    assert store.query([0.1] * 16, k=4) == []


def test_query_k_is_clamped_to_collection_size(manifest, chroma_dir):
    store = VectorStore(path=chroma_dir)
    assert len(store.query([0.2] * 16, k=99)) == store.count()


def test_normalize_produces_unit_vectors():
    assert abs(sum(value**2 for value in normalize([3.0, 4.0])) - 1.0) < 1e-9


def test_add_rejects_mismatched_lengths(chroma_dir):
    store = VectorStore(path=chroma_dir)
    two_chunks = [
        Chunk(id="a", text="a", metadata={"source_file": "a.md"}),
        Chunk(id="b", text="b", metadata={"source_file": "a.md"}),
    ]
    with pytest.raises(ValueError):
        store.add(two_chunks, [[0.1, 0.2]])


def test_add_accepts_matching_lengths(chroma_dir):
    store = VectorStore(path=chroma_dir)
    store.add([Chunk(id="a", text="a", metadata={"source_file": "a.md"})], [[0.1, 0.2]])
    assert store.count() == 1


def test_chunk_ids_are_unique_in_built_index(manifest, chroma_dir):
    store = VectorStore(path=chroma_dir)
    ids = [chunk.id for chunk in store.all_chunks()]
    assert len(ids) == len(set(ids)) == manifest["chunk_count"]


def test_ingest_rejects_empty_corpus(tmp_path, chroma_dir):
    empty = tmp_path / "empty_docs"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        run_ingest(FakeEmbedder(), docs_dir=empty, chroma_dir=chroma_dir, verbose=False)


def test_manifest_records_chunking_parameters(manifest):
    assert manifest["chunk_size"] == CONFIG.CHUNK_SIZE
    assert manifest["chunk_overlap"] == CONFIG.CHUNK_OVERLAP
    assert manifest["embedding_model"] == CONFIG.EMBEDDING_MODEL


def test_all_corpus_documents_contribute_chunks(manifest):
    expected = len(discover(DOCS_DIR))
    assert manifest["chunk_count"] >= expected


def test_chunk_id_helper_is_stable():
    assert chunk_id("week1.md", "section:Chunking", 0) == chunk_id("week1.md", "section:Chunking", 0)
    assert chunk_id("week1.md", "section:Chunking", 0) != chunk_id("week1.md", "section:Embeddings", 0)
