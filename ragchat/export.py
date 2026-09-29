"""Human-readable dump of the indexed chunks and their embeddings.

The live index is binary (data/chroma/). This writes a text artifact so a chunk, its
metadata, and its vector can be inspected without opening a database.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from .chunker import _encode
from .config import CONFIG
from .ingest import manifest_path, read_manifest
from .models import Chunk
from .vectorstore import VectorStore

RULE = "=" * 78
THIN = "-" * 78


def _location(chunk: Chunk) -> str:
    if "page" in chunk.metadata:
        return f"page {chunk.metadata['page']}"
    return f'section "{chunk.metadata.get("section", "")}"'


def _format_vector(values: list[float], width: int) -> str:
    shown = ", ".join(f"{value:+.4f}" for value in values[:width])
    if len(values) > width:
        shown += f", ... (+{len(values) - width} more)"
    return f"[{shown}]"


def _format_bars(values: list[float], top_n: int = 6) -> list[str]:
    ranked = sorted(enumerate(values), key=lambda pair: -abs(pair[1]))[:top_n]
    scale = max((abs(value) for _, value in ranked), default=0.0) or 1.0
    lines = ["", "      strongest dimensions:"]
    for index, value in ranked:
        bar = "#" * max(1, round(abs(value) / scale * 28))
        lines.append(f"        dim {index:<4} {value:+.4f}  {bar}")
    return lines


def build_report(pairs: list[tuple[Chunk, list[float]]], width: int = 12, full: bool = False) -> str:
    manifest = read_manifest() or {}
    dimension = len(pairs[0][1]) if pairs and pairs[0][1] else 0
    shown = dimension if full else min(width, dimension)

    lines = [
        RULE,
        "RAG CHUNK + EMBEDDING EXPORT",
        RULE,
        "Debug artifact. The live index is binary and lives in data/chroma/;",
        "this file is a readable snapshot and is not read at runtime.",
        "",
        f"generated       : {datetime.now().isoformat(timespec='seconds')}",
        f"embedding model : {manifest.get('embedding_model', CONFIG.EMBEDDING_MODEL)}",
        f"chunk size      : {manifest.get('chunk_size', CONFIG.CHUNK_SIZE)} tokens",
        f"chunk overlap   : {manifest.get('chunk_overlap', CONFIG.CHUNK_OVERLAP)} tokens",
        f"top-k           : {CONFIG.TOP_K}",
        f"threshold       : {CONFIG.SIMILARITY_THRESHOLD}",
        f"source hash     : {manifest.get('source_hash', '(unknown)')}",
        f"chunks          : {len(pairs)}",
        f"vector dim      : {dimension}",
        "vectors         : L2-normalized before storage; cosine similarity = dot product",
        f"showing         : {'all' if full else f'first {shown}'} of {dimension} dimensions",
        "",
        "Query-time scoring is cosine similarity, so 1.0000 means identical direction",
        "and 0.0000 means unrelated. A threshold of "
        f"{CONFIG.SIMILARITY_THRESHOLD} separates in-corpus from out-of-corpus questions.",
        RULE,
    ]

    for number, (chunk, vector) in enumerate(pairs, start=1):
        norm = math.sqrt(sum(value * value for value in vector)) if vector else 0.0
        lines.extend(
            [
                "",
                THIN,
                f"[{number}]  id={chunk.id}",
                f"      source     : {chunk.metadata.get('source_file', '')}",
                f"      location   : {_location(chunk)}",
                f"      chunk_index: {chunk.metadata.get('chunk_index')}",
                f"      tokens     : {len(_encode(chunk.text))}",
                f"      norm       : {norm:.4f}",
                "",
                "      text:",
            ]
        )
        for text_line in chunk.text.splitlines():
            lines.append(f"        {text_line}")
        lines.extend(
            [
                "",
                f"      embedding  : {_format_vector(vector, shown)}",
            ]
        )
        if not full:
            lines.extend(_format_bars(vector))

    lines.extend(["", RULE, f"end of export ({len(pairs)} chunks)", RULE, ""])
    return "\n".join(lines)


def export_chunks(out_path=None, width: int = 12, full: bool = False, store: VectorStore | None = None) -> Path:
    store = store or VectorStore()
    pairs = store.all_chunks_with_vectors()
    if not pairs:
        raise ValueError("The index is empty. Build it first: python -m ragchat.ingest --commit")
    out_path = Path(out_path) if out_path else CONFIG.chroma_path.parent / "chunks.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(pairs, width=width, full=full), encoding="utf-8")
    return out_path
