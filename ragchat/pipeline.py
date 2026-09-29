"""The orchestrator. `ask()` is the only entry point the UI needs (architecture.md section 2)."""

from __future__ import annotations

import time

from .config import CONFIG
from .embedder import get_embedder
from .errors import ConfigurationError, IndexMissingError, RagChatError
from .generator import Generator, get_llm
from .ingest import manifest_warnings
from .models import Answer, ScoredChunk, coerce_history
from .prompts import REFUSAL
from .retriever import Retriever
from .vectorstore import VectorStore

UNAVAILABLE = "The system is temporarily unavailable. Please try again."


class Pipeline:
    def __init__(self, embedder=None, store=None, llm=None, k: int | None = None, threshold: float | None = None) -> None:
        self.store = store or VectorStore()
        self.embedder = embedder or get_embedder()
        self.llm = llm
        self.k = CONFIG.TOP_K if k is None else k
        self.threshold = CONFIG.SIMILARITY_THRESHOLD if threshold is None else threshold
        self.retriever = Retriever(self.embedder, self.store, k=self.k)
        self.generator = Generator(llm) if llm is not None else None

    def ask(self, question: str, history: list | None = None) -> Answer:
        """Retrieve, gate, then generate or refuse.

        Runtime failures become an Answer with friendly text. Configuration failures are
        re-raised, because a missing key is a startup problem, not a question-level one.
        """
        history = coerce_history(history)
        started = time.perf_counter()
        try:
            result = self.retriever.retrieve(question, history, k=self.k)
        except ConfigurationError:
            raise
        except RagChatError as exc:
            return Answer(text=str(exc), latency_ms={"total": _elapsed(started)})
        except Exception:
            return Answer(text=UNAVAILABLE, latency_ms={"total": _elapsed(started)})

        retrieve_ms = _elapsed(started)
        base = {
            "embed": result.latency_ms.get("embed", 0),
            "query": result.latency_ms.get("query", 0),
            "retrieve": retrieve_ms,
        }

        if not self.retriever.passes_threshold(result.best_score, threshold=self.threshold):
            return Answer(
                text=REFUSAL,
                retrieved=list(result.hits),
                refused=True,
                latency_ms={**base, "total": _elapsed(started)},
            )

        if self.generator is None:
            raise ConfigurationError(
                "No LLM is configured, so this question cannot be answered. Set LLM_API_KEY in .env."
            )

        generate_started = time.perf_counter()
        text, citations = self.generator.generate(question, [hit.chunk for hit in result.hits], history)
        return Answer(
            text=text,
            citations=citations,
            retrieved=list(result.hits),
            refused=False,
            latency_ms={
                **base,
                "generate": int((time.perf_counter() - generate_started) * 1000),
                "total": _elapsed(started),
            },
        )

    def preflight(self, question: str, history: list | None = None) -> Answer:
        """Retrieve and gate only, never generate. Used to check the demo script offline."""
        history = coerce_history(history)
        started = time.perf_counter()
        result = self.retriever.retrieve(question, history, k=self.k)
        passes = self.retriever.passes_threshold(result.best_score, threshold=self.threshold)
        return Answer(
            text="would answer" if passes else "would refuse",
            retrieved=list(result.hits),
            refused=not passes,
            latency_ms={
                "embed": result.latency_ms.get("embed", 0),
                "query": result.latency_ms.get("query", 0),
                "total": _elapsed(started),
            },
        )


def boot() -> tuple["Pipeline", list[tuple[str, str]], dict]:
    """Build everything a long-lived UI process needs, once.

    Returns (pipeline, startup_issues, config_summary). The LLM is resolved here rather
    than in the UI, so a long-running app can call only Pipeline.ask and never import the
    generator, retriever, or embedder (implementation.md Phase 6).
    """
    issues = startup_checks()
    llm = None
    if not any(level == "error" for level, _ in issues):
        try:
            llm = get_llm()
        except ConfigurationError:
            llm = None
    return Pipeline(llm=llm), issues, describe()


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def startup_checks() -> list[tuple[str, str]]:
    """(level, message) pairs for a startup banner: 'error' blocks use, 'warning' does not."""
    from .errors import ProviderError as _ProviderError

    issues: list[tuple[str, str]] = []

    if not CONFIG.LLM_API_KEY and CONFIG.resolved_llm_backend == "local":
        issues.append(
            ("error", "No LLM configured. Set LLM_API_KEY in .env, or implement ragchat.generator.LocalLLM.")
        )
    else:
        # Constructing the client is the only reliable way to catch a missing key: forcing
        # LLM_BACKEND=hosted with no key would otherwise slip past the check above.
        try:
            get_llm()
        except ConfigurationError as exc:
            issues.append(("error", str(exc)))

    try:
        store = VectorStore()
        count = store.count()
    except Exception as exc:
        issues.append(("error", f"Could not open the vector index at {CONFIG.chroma_path}: {exc}"))
        return issues
    if count == 0:
        issues.append(("error", "The vector index is empty. Build it: python -m ragchat.ingest --commit"))

    try:
        get_embedder()
    except _ProviderError as exc:
        issues.append(("error", str(exc)))

    for warning in manifest_warnings():
        issues.append(("warning", warning))

    return issues


def describe() -> dict[str, object]:
    """Config summary for the UI sidebar. Model names only, never keys."""
    try:
        chunk_count = VectorStore().count()
    except Exception:
        chunk_count = 0
    return {
        "embedding model": CONFIG.EMBEDDING_MODEL,
        "embedding backend": CONFIG.resolved_embedding_backend,
        "llm model": CONFIG.LLM_MODEL,
        "llm backend": CONFIG.resolved_llm_backend,
        "chunk size": f"{CONFIG.CHUNK_SIZE} tokens",
        "top-k": CONFIG.TOP_K,
        "similarity threshold": CONFIG.SIMILARITY_THRESHOLD,
        "history turns": CONFIG.HISTORY_TURNS,
        "retrieval history turns": CONFIG.RETRIEVAL_HISTORY_TURNS,
        "indexed chunks": chunk_count,
    }


__all__ = ["Pipeline", "boot", "startup_checks", "describe", "ScoredChunk", "REFUSAL"]
