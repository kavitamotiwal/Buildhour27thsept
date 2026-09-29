from __future__ import annotations

import pytest

from ragchat.chunker import _encode
from ragchat.export import build_report, export_chunks
from ragchat.ingest import run_ingest
from ragchat.models import Chunk
from ragchat.vectorstore import VectorStore


@pytest.fixture(scope="module")
def local_embedder():
    from ragchat.embedder import LocalEmbedder

    return LocalEmbedder()


@pytest.fixture(scope="module")
def index(tmp_path_factory, local_embedder):
    from ragchat.config import CONFIG

    chroma_dir = tmp_path_factory.mktemp("export") / "chroma"
    run_ingest(local_embedder, docs_dir=CONFIG.docs_path, chroma_dir=chroma_dir, verbose=False)
    return chroma_dir


@pytest.fixture(scope="module")
def exported(tmp_path_factory, index):
    out = tmp_path_factory.mktemp("out") / "chunks.txt"
    export_chunks(out_path=out, store=VectorStore(path=index))
    return out.read_text(encoding="utf-8")


def test_export_contains_every_chunk(index, exported):
    store = VectorStore(path=index)
    assert len(store.all_chunks()) == 41
    for chunk in store.all_chunks():
        assert chunk.id in exported


def test_export_includes_full_chunk_text(index, exported):
    flattened = " ".join(exported.split())
    store = VectorStore(path=index)
    for chunk in store.all_chunks():
        first_words = " ".join(chunk.text.split()[:8])
        assert first_words in flattened


def test_export_includes_section_anchors(exported):
    assert 'section "Expense Ratio"' in exported
    assert 'section "Exit Load"' in exported


def test_export_includes_embeddings(index, exported):
    vectors = [vector for _, vector in VectorStore(path=index).all_chunks_with_vectors()]
    assert all(len(vector) == 384 for vector in vectors)
    assert exported.count("embedding  :") == len(vectors)
    assert "vector dim      : 384" in exported


def test_export_reports_normalized_norms(exported):
    assert "norm       : 1.0000" in exported


def test_export_documents_the_running_config(exported):
    assert "chunk size      : 500 tokens" in exported
    assert "chunk overlap   : 50 tokens" in exported
    assert "top-k           : 4" in exported
    assert "RAG CHUNK + EMBEDDING EXPORT" in exported


def test_export_truncates_by_default_and_expands_with_full(index):
    pairs = VectorStore(path=index).all_chunks_with_vectors()
    assert "showing         : first 12 of 384 dimensions" in build_report(pairs)
    assert "(+372 more)" in build_report(pairs)
    full = build_report(pairs, full=True)
    assert "showing         : all of 384 dimensions" in full
    assert "(+372 more)" not in full
    assert full.count("strongest dimensions") == 0


def test_export_lists_strongest_dimensions(exported):
    assert "strongest dimensions:" in exported
    assert exported.count("strongest dimensions:") == 41


def test_export_rejects_empty_index(tmp_path):
    from ragchat.export import export_chunks as run_export

    with pytest.raises(ValueError):
        run_export(out_path=tmp_path / "empty.txt", store=VectorStore(path=tmp_path / "nope"))


def test_report_is_valid_utf8_text(exported):
    assert exported.startswith("=")
    assert exported.rstrip().endswith("=")
