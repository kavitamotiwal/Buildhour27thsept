from __future__ import annotations

import hashlib

import tiktoken

from .config import CONFIG
from .errors import IngestError
from .models import Chunk

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _encode(text: str) -> list[int]:
    try:
        return _ENCODER.encode(text)
    except Exception as exc:
        raise IngestError(f"tokenization failed: {exc}") from exc


def _decode(tokens: list[int]) -> str:
    return _ENCODER.decode(tokens)


def chunk_id(source_file: str, anchor: str, chunk_index: int) -> str:
    raw = f"{source_file}:{anchor}:{chunk_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _token_windows(tokens: list[int], size: int, overlap: int) -> list[list[int]]:
    if size <= 0:
        raise IngestError("CHUNK_SIZE must be positive")
    if overlap < 0 or overlap >= size:
        raise IngestError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
    step = size - overlap
    return [tokens[start : start + size] for start in range(0, len(tokens), step) if tokens[start : start + size]]


def _anchor_of(metadata: dict) -> str:
    page = metadata.get("page")
    if page is not None:
        return f"page:{page}"
    section = metadata.get("section")
    return f"section:{section}" if section else "segment:none"


def chunk_segments(segments: list[tuple[str, dict]], size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """Split each segment into overlapping token-counted windows.

    Windows never span segments, so a chunk always carries one page or section anchor
    (architecture.md section 3.4). chunk_index is per segment, keeping ids stable across re-ingests.
    """
    size = CONFIG.CHUNK_SIZE if size is None else size
    overlap = CONFIG.CHUNK_OVERLAP if overlap is None else overlap

    chunks: list[Chunk] = []
    for text, metadata in segments:
        source_file = str(metadata.get("source_file", ""))
        anchor = _anchor_of(metadata)
        for chunk_index, window in enumerate(_token_windows(_encode(text), size, overlap)):
            body = _decode(window).strip()
            if not body:
                continue
            full_metadata = {**metadata, "chunk_index": chunk_index}
            chunks.append(
                Chunk(
                    id=chunk_id(source_file, anchor, chunk_index),
                    text=body,
                    metadata=full_metadata,
                )
            )
    return chunks


def chunk_file(path, root=None, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """Load one document and return its chunks. Convenience wrapper for CLI and tests."""
    from .loaders import load

    return chunk_segments(load(path, root), size=size, overlap=overlap)
