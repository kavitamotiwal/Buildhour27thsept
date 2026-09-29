from __future__ import annotations

import math
from typing import Callable, TypeVar

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from .config import CONFIG
from .errors import IndexMissingError
from .models import Chunk, ScoredChunk

COLLECTION_NAME = "course_notes"
INGEST_COMMAND = "python -m ragchat.ingest --commit"
COSINE_METADATA = {"hnsw:space": "cosine"}

T = TypeVar("T")


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


class VectorStore:
    """Thin wrapper over a persistent local Chroma collection (architecture.md section 5.2)."""

    def __init__(self, path=None, collection_name: str = COLLECTION_NAME) -> None:
        self.path = path or CONFIG.chroma_path
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        self._handle = None

    def _open(self):
        try:
            return self._client.get_collection(self.collection_name)
        except NotFoundError:
            return self._client.get_or_create_collection(self.collection_name, metadata=COSINE_METADATA)

    def _run(self, action: Callable[[object], T]) -> T:
        """Run against the collection, re-resolving the handle if another store reset it."""
        for _ in range(2):
            try:
                return action(self._handle if self._handle is not None else self._open())
            except NotFoundError:
                self._handle = None
        raise IndexMissingError(f"The vector index is unavailable. Rebuild it with: {INGEST_COMMAND}")

    def count(self) -> int:
        return self._run(lambda collection: collection.count())

    def reset(self) -> None:
        """Drop every record. Ingestion always calls this first (architecture.md section 3.4).

        Idempotent: a collection that does not exist yet already holds no records.
        """
        try:
            self._client.delete_collection(self.collection_name)
        except NotFoundError:
            pass
        self._handle = None

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        self._run(
            lambda collection: collection.add(
                ids=[chunk.id for chunk in chunks],
                embeddings=[normalize(vector) for vector in vectors],
                documents=[chunk.text for chunk in chunks],
                metadatas=[_chroma_metadata(chunk) for chunk in chunks],
            )
        )

    def query(self, vector: list[float], k: int) -> list[ScoredChunk]:
        total = self.count()
        if total == 0:
            return []
        k = min(k, total)

        def action(collection) -> list[ScoredChunk]:
            result = collection.query(
                query_embeddings=[normalize(vector)],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            return _to_scored(result)

        return self._run(action)

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk, for verification and debugging."""
        return [chunk for chunk, _ in self.all_chunks_with_vectors()]

    def all_chunks_with_vectors(self) -> list[tuple[Chunk, list[float]]]:
        """Every stored chunk paired with its stored (normalized) embedding."""
        def action(collection) -> list[tuple[Chunk, list[float]]]:
            stored = collection.get(include=["documents", "metadatas", "embeddings"])
            documents = stored.get("documents")
            metadatas = stored.get("metadatas")
            embeddings = stored.get("embeddings")
            documents = documents if documents is not None else []
            metadatas = metadatas if metadatas is not None else []
            embeddings = embeddings if embeddings is not None else []
            pairs: list[tuple[Chunk, list[float]]] = []
            for index, chunk_id in enumerate(stored["ids"]):
                chunk = Chunk(
                    id=chunk_id,
                    text=(documents[index] or "") if index < len(documents) else "",
                    metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
                )
                vector = [float(v) for v in embeddings[index]] if index < len(embeddings) else []
                pairs.append((chunk, vector))
            return pairs

        return self._run(action)

    def require_index(self) -> None:
        if self.count() == 0:
            raise IndexMissingError(f"The vector index is empty or missing. Build it with: {INGEST_COMMAND}")


def _to_scored(result: dict) -> list[ScoredChunk]:
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    scored: list[ScoredChunk] = []
    for rank, chunk_id in enumerate(ids):
        metadata = metadatas[rank] if rank < len(metadatas) else {}
        distance = float(distances[rank]) if rank < len(distances) else 1.0
        scored.append(
            ScoredChunk(
                chunk=Chunk(id=chunk_id, text=documents[rank] or "", metadata=dict(metadata or {})),
                score=1.0 - distance,
                rank=rank + 1,
            )
        )
    return scored


def _chroma_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Chroma accepts only scalar metadata values."""
    flat: dict[str, str | int | float | bool] = {}
    for key, value in chunk.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif value is not None:
            flat[key] = str(value)
    flat.setdefault("source_file", chunk.source_file)
    return flat
