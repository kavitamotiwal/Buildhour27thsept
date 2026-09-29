from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .chunker import chunk_segments
from .config import CONFIG
from .loaders import discover, load, source_hash
from .vectorstore import VectorStore

MANIFEST_FIELDS = ("embedding_model", "chunk_size", "chunk_overlap", "chunk_count", "source_hash")


def manifest_path(chroma_dir=None) -> Path:
    chroma_dir = Path(chroma_dir) if chroma_dir else CONFIG.chroma_path
    return chroma_dir.parent / "manifest.json"


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def run_ingest(embedder, docs_dir=None, size=None, overlap=None, chroma_dir=None, verbose: bool = True) -> dict:
    """Rebuild the index from scratch and write the manifest.

    The embedder is injected so tests can exercise this path without a network key.
    """
    docs_dir = Path(docs_dir) if docs_dir else CONFIG.docs_path
    size = CONFIG.CHUNK_SIZE if size is None else size
    overlap = CONFIG.CHUNK_OVERLAP if overlap is None else overlap

    paths = discover(docs_dir)
    if not paths:
        raise FileNotFoundError(f"No supported documents found in {docs_dir}")

    chunks = []
    for path in paths:
        segments = load(path, root=docs_dir)
        chunks.extend(chunk_segments(segments, size=size, overlap=overlap))
    if not chunks:
        raise ValueError("corpus produced zero chunks")

    store = VectorStore(path=Path(chroma_dir) if chroma_dir else CONFIG.chroma_path)
    store.reset()

    batch_size = CONFIG.EMBED_BATCH_SIZE
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        store.add(batch, embedder.embed([chunk.text for chunk in batch]))
        if verbose:
            print(f"  embedded {min(start + batch_size, len(chunks))}/{len(chunks)}")

    manifest = {
        "embedding_model": getattr(embedder, "model", CONFIG.EMBEDDING_MODEL),
        "chunk_size": size,
        "chunk_overlap": overlap,
        "chunk_count": len(chunks),
        "source_hash": source_hash(docs_dir),
    }
    write_manifest(manifest, manifest_path(chroma_dir))
    return manifest


def run_dry_run(docs_dir=None, size=None, overlap=None) -> int:
    docs_dir = docs_dir or CONFIG.docs_path
    size = CONFIG.CHUNK_SIZE if size is None else size
    overlap = CONFIG.CHUNK_OVERLAP if overlap is None else overlap

    paths = discover(docs_dir)
    print(f"documents dir : {docs_dir}")
    print(f"chunking      : size={size} tokens, overlap={overlap} tokens")
    print(f"files found   : {len(paths)}\n")

    if not paths:
        print("no supported documents found (expected .pdf, .md, .txt)")
        return 1

    total_segments = 0
    total_chunks = 0
    rows = []
    for path in paths:
        segments = load(path, root=docs_dir)
        chunks = chunk_segments(segments, size=size, overlap=overlap)
        total_segments += len(segments)
        total_chunks += len(chunks)
        rows.append((path, len(segments), len(chunks)))

    width = max(len(p.name) for p, _, _ in rows)
    for path, segments, chunks in rows:
        print(f"{path.name.ljust(width)}  segments={segments:>4}  chunks={chunks:>5}")

    print(f"\ntotal segments: {total_segments}")
    print(f"total chunks  : {total_chunks}")
    print("\ndry run: nothing written. Re-run with --commit to build the index.")
    return 0


def read_manifest(chroma_dir=None) -> dict | None:
    path = manifest_path(chroma_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_warnings(expected_model: str | None = None, docs_dir=None) -> list[str]:
    """Startup warnings when the index and the running config disagree (architecture.md section 5.4)."""
    manifest = read_manifest()
    if manifest is None:
        return ["No data/manifest.json found. The index may not have been built, or was built with --dry-run."]
    warnings: list[str] = []
    model = expected_model or CONFIG.EMBEDDING_MODEL
    if manifest.get("embedding_model") != model:
        warnings.append(
            f"Index was built with embedding model {manifest.get('embedding_model')!r} but the running "
            f"config is {model!r}. Similarity scores are not comparable; rebuild with: python -m ragchat.ingest --commit"
        )
    current = source_hash(docs_dir or CONFIG.docs_path)
    if manifest.get("source_hash") != current:
        warnings.append(
            "The documents have changed since the index was built. Rebuild with: python -m ragchat.ingest --commit"
        )
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragchat.ingest", description="Build the HDFC mutual-fund FAQ index.")
    parser.add_argument("--dry-run", action="store_true", help="load and chunk only; write nothing (default)")
    parser.add_argument("--commit", action="store_true", help="build the index and write data/manifest.json")
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--chroma-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args(argv)

    size = CONFIG.CHUNK_SIZE if args.chunk_size is None else args.chunk_size
    overlap = CONFIG.CHUNK_OVERLAP if args.chunk_overlap is None else args.chunk_overlap

    if args.commit:
        from .embedder import get_embedder
        from .errors import RagChatError

        try:
            embedder = get_embedder()
        except RagChatError as exc:
            print(f"error: {exc}")
            return 1
        try:
            manifest = run_ingest(
                embedder,
                docs_dir=args.docs_dir,
                size=size,
                overlap=overlap,
                chroma_dir=args.chroma_dir,
            )
        except RagChatError as exc:
            print(f"error: {exc}")
            return 1
        print("\nindex built")
        for key in MANIFEST_FIELDS:
            print(f"  {key:<16}: {manifest[key]}")
        return 0

    return run_dry_run(docs_dir=args.docs_dir, size=size, overlap=overlap)


if __name__ == "__main__":
    raise SystemExit(main())
