from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from .config import CONFIG
from .errors import ProviderError

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.0
DEFAULT_BASE_URL = "https://api.openai.com/v1"


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input, in order."""
        ...


class HostedEmbedder:
    """OpenAI-compatible /embeddings client, batched and retried."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int | None = None,
        timeout_s: int = 60,
    ) -> None:
        self.model = model or CONFIG.EMBEDDING_MODEL
        self.api_key = api_key if api_key is not None else CONFIG.EMBEDDING_API_KEY
        self.base_url = (base_url if base_url is not None else CONFIG.EMBEDDING_BASE_URL) or DEFAULT_BASE_URL
        self.batch_size = batch_size or CONFIG.EMBED_BATCH_SIZE
        self.timeout_s = timeout_s
        if not self.api_key:
            raise ProviderError("EMBEDDING_API_KEY is not set. Copy .env.example to .env and fill it in.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/embeddings",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return self._parse(body, len(batch))
            except Exception as exc:
                last_error = exc
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_BACKOFF_S * (2**attempt))
        raise ProviderError(f"embedding request failed after {RETRY_ATTEMPTS} attempts: {last_error}")

    @staticmethod
    def _parse(body: dict, expected: int) -> list[list[float]]:
        data = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in data]
        if len(vectors) != expected:
            raise ProviderError(f"provider returned {len(vectors)} vectors for {expected} inputs")
        return vectors


class LocalEmbedder:
    """Offline backend: a real ONNX sentence-embedding model, no API key required.

    Scores from this backend are NOT interchangeable with the hosted backend's. Recalibrate
    SIMILARITY_THRESHOLD whenever the embedding model changes (architecture.md section 5.3).
    """

    def __init__(self, model: str | None = None, batch_size: int | None = None) -> None:
        self.model = model or CONFIG.EMBEDDING_MODEL
        self.batch_size = batch_size or CONFIG.EMBED_BATCH_SIZE
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ProviderError("fastembed is not installed. Run: pip install fastembed") from exc
            self._encoder = TextEmbedding(self.model)
        return self._encoder

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoder = self._get_encoder()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend([vector.tolist() for vector in encoder.embed(batch)])
        return vectors


def get_embedder(backend: str | None = None) -> Embedder:
    backend = backend or CONFIG.resolved_embedding_backend
    if backend == "local":
        return LocalEmbedder()
    if backend == "hosted":
        return HostedEmbedder()
    raise ProviderError(f"unknown EMBEDDING_BACKEND {backend!r}; expected 'auto', 'hosted', or 'local'")
