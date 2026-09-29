from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)

    @property
    def source_file(self) -> str:
        return str(self.metadata.get("source_file", ""))

    @property
    def anchor(self) -> str:
        """Human-readable position: page for PDFs, section heading otherwise."""
        page = self.metadata.get("page")
        if page is not None:
            return f"p.{page}"
        section = self.metadata.get("section")
        return str(section) if section else ""


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


def coerce_history(history: list | None) -> list[dict]:
    """Accept Turn objects or plain dicts; return OpenAI-shaped message dicts."""
    messages: list[dict] = []
    for item in list(history or []):
        turn = item if isinstance(item, Turn) else Turn(**item) if isinstance(item, dict) else None
        if turn is not None and turn.content.strip():
            messages.append({"role": turn.role, "content": turn.content})
    return messages


class RetrieveResult(BaseModel):
    """Outcome of one retrieval pass, including the query that was actually embedded."""

    hits: list[ScoredChunk] = Field(default_factory=list)
    best_score: float = 0.0
    conditioned_query: str = ""
    latency_ms: dict[str, int] = Field(default_factory=dict)


class Answer(BaseModel):
    text: str
    citations: list[Chunk] = Field(default_factory=list)
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    refused: bool = False
    latency_ms: dict[str, int] = Field(default_factory=dict)
