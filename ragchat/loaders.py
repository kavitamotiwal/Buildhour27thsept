from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from .errors import IngestError

Segment = tuple[str, dict]

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_PDF_SUFFIXES = {".pdf"}
_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_TEXT_SUFFIXES = {".txt", ".text"}
SUPPORTED_SUFFIXES = _PDF_SUFFIXES | _MARKDOWN_SUFFIXES | _TEXT_SUFFIXES


def _source_file(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _load_pdf(path: Path, source_file: str) -> list[Segment]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise IngestError(f"{source_file}: could not read PDF ({exc})") from exc

    segments: list[Segment] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            raise IngestError(f"{source_file}: could not extract page {page_number} ({exc})") from exc
        if not text:
            continue
        segments.append((text, {"source_file": source_file, "page": page_number}))
    return segments


def _load_markdown(path: Path, source_file: str) -> list[Segment]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise IngestError(f"{source_file}: could not read file ({exc})") from exc

    lines = raw.splitlines()
    segments: list[Segment] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            body = f"{current_heading}\n{text}" if current_heading else text
            segments.append((body, {"source_file": source_file, "section": current_heading or "(preamble)"}))

    for line in lines:
        match = _HEADING.match(line)
        if match:
            flush()
            buffer = []
            current_heading = match.group(2).strip()
        else:
            buffer.append(line)
    flush()
    return segments


def _load_text(path: Path, source_file: str) -> list[Segment]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise IngestError(f"{source_file}: could not read file ({exc})") from exc
    if not text:
        return []
    return [(text, {"source_file": source_file, "section": "(full document)"})]


def load(path: Path, root: Path | None = None) -> list[Segment]:
    """Read one document into anchored segments.

    PDF yields one segment per page, Markdown one per heading, plain text one for the file.
    Raises IngestError naming the file rather than returning partial or empty text.
    """
    path = Path(path)
    root = Path(root) if root else path.parent
    source_file = _source_file(path, root)
    suffix = path.suffix.lower()

    if suffix in _PDF_SUFFIXES:
        segments = _load_pdf(path, source_file)
    elif suffix in _MARKDOWN_SUFFIXES:
        segments = _load_markdown(path, source_file)
    elif suffix in _TEXT_SUFFIXES:
        segments = _load_text(path, source_file)
    else:
        raise IngestError(f"{source_file}: unsupported extension {suffix or '(none)'}")

    if not segments:
        raise IngestError(f"{source_file}: no extractable text")
    return segments


def discover(docs_dir: Path) -> list[Path]:
    """All supported documents under docs_dir, sorted for deterministic ingestion."""
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        return []
    return sorted(p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def source_hash(docs_dir: Path) -> str:
    """Content fingerprint of the corpus, independent of mtime or absolute path.

    Part of the manifest (architecture.md section 5.4). Compared at serve time by Phase 6.
    """
    docs_dir = Path(docs_dir)
    digest = hashlib.sha256()
    for path in discover(docs_dir):
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{_source_file(path, docs_dir)}:{file_digest}\n".encode("utf-8"))
    return digest.hexdigest()
