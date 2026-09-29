from __future__ import annotations

import time

from .config import CONFIG
from .embedder import Embedder
from .errors import IndexMissingError
from .models import RetrieveResult, coerce_history
from .vectorstore import INGEST_COMMAND, VectorStore


# Words that signal the question refers back to something already said. History is
# prepended ONLY when one of these is present, because prepending unconditionally lets a
# stale topic dominate the embedding: after asking about chunk overlap, the unrelated
# "What is the capital city of France?" scored 0.8160 and sailed past the gate, purely
# because the earlier turn was concatenated in front of it. Degrading to stateless
# retrieval is the safe failure mode; that path is covered by the 20-question set.
REFERRING_TERMS = (
    "they", "them", "their", "theirs", "it", "its", "that", "those", "this", "these",
    "one", "same", "other", "another", "above", "earlier", "previous", "mentioned",
    "both", "either", "also", "instead", "too",
)

# A question opening with a connective is a continuation: "And why?", "But what about X?".
CONTINUATION_STARTERS = ("and", "but", "so", "then", "also", "because", "since", "however", "what about", "how about")


def is_follow_up(question: str) -> bool:
    """True when the question explicitly refers back to earlier turns."""
    lowered = question.lower().strip()
    if lowered.startswith(CONTINUATION_STARTERS):
        return True
    words = {token.strip(".,!?;:'\"()[]") for token in lowered.split()}
    return bool(words & set(REFERRING_TERMS))


def condition(question: str, history: list[dict] | None = None, turns: int | None = None) -> str:
    """Prepend recent turns to the question so a follow-up resolves its referents.

    The concatenation is what gets embedded; there is no separate query-rewrite model call
    (architecture.md section 3.5). History is prepended only for questions that actually
    refer back (see is_follow_up), so unrelated questions stay stateless.

    The window is RETRIEVAL_HISTORY_TURNS, wider than the HISTORY_TURNS the generator sees:
    the embedding needs the older referent ("the metric we mentioned before the split"),
    but the model does not need every turn replayed to read four source blocks. The wider
    window is safe here only because is_follow_up keeps unrelated questions stateless, and
    because 10 turns of history is still bounded -- past that, prefix text starts crowding
    out the question itself.
    """
    question = question.strip()
    if not history or not is_follow_up(question):
        return f"user: {question}"
    turns = CONFIG.RETRIEVAL_HISTORY_TURNS if turns is None else turns
    parts: list[str] = []
    for message in coerce_history(history)[-2 * turns :]:
        parts.append(f"{message['role']}: {message['content']}")
    parts.append(f"user: {question}")
    return "\n".join(parts)


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore | None = None, k: int | None = None) -> None:
        self.embedder = embedder
        self.store = store or VectorStore()
        self.k = CONFIG.TOP_K if k is None else k

    def retrieve(self, question: str, history: list[dict] | None = None, k: int | None = None) -> RetrieveResult:
        if self.store.count() == 0:
            raise IndexMissingError(f"The vector index is empty or missing. Build it with: {INGEST_COMMAND}")

        query = condition(question, history)
        started = time.perf_counter()
        vector = self.embedder.embed([query])[0]
        embed_ms = int((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        hits = self.store.query(vector, k or self.k)
        query_ms = int((time.perf_counter() - started) * 1000)

        best_score = hits[0].score if hits else 0.0
        return RetrieveResult(
            hits=hits,
            best_score=best_score,
            conditioned_query=query,
            latency_ms={"embed": embed_ms, "query": query_ms},
        )

    def passes_threshold(self, best_score: float, threshold: float | None = None) -> bool:
        """The out-of-corpus gate (architecture.md section 3.7).

        An unset threshold means calibration has not run yet, and the gate is open.
        """
        threshold = CONFIG.SIMILARITY_THRESHOLD if threshold is None else threshold
        if threshold is None:
            return True
        return best_score >= threshold
