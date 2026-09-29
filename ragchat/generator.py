from __future__ import annotations

import json
import time
import urllib.request
from typing import Protocol, runtime_checkable

from .config import CONFIG
from .errors import ConfigurationError, ProviderError
from .models import Chunk
from .prompts import FRIENDLY_ERROR, REFUSAL, build_prompt

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.0
DEFAULT_BASE_URL = "https://api.openai.com/v1"
USER_AGENT = "ragchat/0.1"


@runtime_checkable
class LLM(Protocol):
    def complete(self, messages: list[dict]) -> str:
        """Return the assistant's reply to a chat message list."""
        ...


class HostedLLM:
    """OpenAI-compatible /chat/completions client, non-streaming, with retries."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        timeout_s: int | None = None,
    ) -> None:
        self.model = model or CONFIG.LLM_MODEL
        self.api_key = api_key if api_key is not None else CONFIG.LLM_API_KEY
        self.base_url = (base_url if base_url is not None else CONFIG.LLM_BASE_URL) or DEFAULT_BASE_URL
        self.temperature = CONFIG.LLM_TEMPERATURE if temperature is None else temperature
        self.timeout_s = CONFIG.LLM_TIMEOUT_S if timeout_s is None else timeout_s
        if not self.api_key:
            raise ConfigurationError("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")

    def complete(self, messages: list[dict]) -> str:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": self.temperature, "stream": False}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            # Cloudflare in front of Groq answers 403 "error code: 1010" to Python's default
            # Python-urllib/x.y User-Agent, before the request ever reaches the API. Naming
            # the client honestly is enough to get through.
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"chat completion failed: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}") from exc


class LocalLLM:
    """Offline fallback. Not selected in this phase (architecture.md section 5.6)."""

    available = False

    def complete(self, messages: list[dict]) -> str:
        raise ConfigurationError(
            "LocalLLM is not implemented yet. Set LLM_API_KEY, or implement this class to run offline."
        )


def get_llm(backend: str | None = None) -> LLM:
    backend = backend or CONFIG.resolved_llm_backend
    if backend == "local":
        if LocalLLM.available:
            return LocalLLM()
        raise ConfigurationError(
            "No LLM is configured. Set LLM_API_KEY (and LLM_BASE_URL / LLM_MODEL) in .env, "
            "or implement ragchat.generator.LocalLLM to run without a hosted model."
        )
    if backend == "hosted":
        return HostedLLM()
    raise ConfigurationError(f"unknown LLM_BACKEND {backend!r}; expected 'auto', 'hosted', or 'local'")


def unique_citations(chunks: list[Chunk]) -> list[Chunk]:
    """Citations are derived from what was sent to the model, in order, de-duplicated.

    The model's own [n] markers are never parsed to build this list, so a citation can
    never reference a chunk that was not provided (architecture.md section 3.6).
    """
    seen: set[str] = set()
    citations: list[Chunk] = []
    for chunk in chunks:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        citations.append(chunk)
    return citations


class Generator:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def generate(self, question: str, chunks: list[Chunk], history: list[dict] | None = None) -> tuple[str, list[Chunk]]:
        """Return (answer text, citations).

        Runtime failures become a friendly message. Configuration failures are re-raised:
        they are not retryable and must surface at startup, not per question.
        """
        if not chunks:
            return REFUSAL, []
        try:
            text = self._with_retries(build_prompt(question, chunks, history))
        except ConfigurationError:
            raise
        except Exception:
            return FRIENDLY_ERROR, unique_citations(chunks)
        return text, unique_citations(chunks)

    def _with_retries(self, messages: list[dict]) -> str:
        last_error: Exception | None = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                return self.llm.complete(messages)
            except ConfigurationError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_BACKOFF_S * (2**attempt))
        raise ProviderError(f"model failed after {RETRY_ATTEMPTS} attempts: {last_error}")
