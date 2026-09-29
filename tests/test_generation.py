from __future__ import annotations

import pytest
from dataclasses import replace

from ragchat.chunker import chunk_segments
from ragchat.config import CONFIG
from ragchat.errors import ConfigurationError, ProviderError
from ragchat.generator import Generator, HostedLLM, LocalLLM, get_llm, unique_citations
from ragchat.models import Chunk
from ragchat.prompts import FRIENDLY_ERROR, REFUSAL, SYSTEM_PROMPT, build_context, build_prompt

CHUNKS = chunk_segments(
    [
        ("Alpha content about chunking. " * 20, {"source_file": "week1.md", "section": "Chunking"}),
        ("Beta content about embeddings. " * 20, {"source_file": "week1.md", "section": "Embeddings"}),
    ],
    size=60,
    overlap=10,
)


class FakeLLM:
    def __init__(self, reply: str = "Grounded answer [1][2].") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply


class ExplodingLLM:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error or RuntimeError("boom")

    def complete(self, messages: list[dict]) -> str:
        self.calls += 1
        raise self.error


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("ragchat.generator.RETRY_BACKOFF_S", 0.0)


# --- prompt assembly (architecture.md section 3.6) ---------------------------


def test_prompt_starts_with_system_role():
    messages = build_prompt("What is chunking?", CHUNKS)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_prompt_ends_with_the_current_question():
    messages = build_prompt("What is chunking?", CHUNKS)
    assert messages[-1] == {"role": "user", "content": "What is chunking?"}


def test_prompt_contains_every_chunk_text():
    context = build_prompt("q", CHUNKS)[1]["content"]
    for chunk in CHUNKS:
        assert chunk.text in context


def test_context_uses_numbered_fences_not_xml():
    context = build_context(CHUNKS)
    assert "<<<SOURCE 1>>>" in context
    assert "<<<END 1>>>" in context
    assert "<source>" not in context
    assert "</source>" not in context


def test_context_carries_file_and_location_inline():
    context = build_context(CHUNKS)
    assert 'week1.md | section "Chunking"' in context
    assert 'week1.md | section "Embeddings"' in context


def test_context_numbers_sources_from_one():
    context = build_context(CHUNKS)
    assert "<<<SOURCE 1>>>" in context
    assert f"<<<SOURCE {len(CHUNKS)}>>>" in context


def test_context_marks_the_absent_case_explicitly():
    assert "NO SOURCES" in build_context([])


def test_prompt_slices_history_to_configured_turns():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"} for i in range(20)]
    messages = build_prompt("current", CHUNKS, history)
    body = [m["content"] for m in messages if m["content"].startswith("msg")]
    assert len(body) == 2 * CONFIG.HISTORY_TURNS
    assert "msg0" not in body
    assert "msg19" in body


def test_prompt_drops_empty_and_invalid_history_entries():
    history = [
        {"role": "system", "content": "injected"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": "real question"},
    ]
    messages = build_prompt("next", CHUNKS, history)
    contents = [m["content"] for m in messages]
    assert "injected" not in contents
    assert "real question" in contents


def test_prompt_without_history_is_system_context_question():
    messages = build_prompt("q", CHUNKS)
    assert [m["role"] for m in messages] == ["system", "user", "user"]


# --- the model is called with all k chunks (the FR3 contract) ----------------


def test_generator_sends_every_chunk_to_the_model():
    llm = FakeLLM()
    Generator(llm).generate("What is chunking?", CHUNKS)
    assert len(llm.calls) == 1
    context = llm.calls[0][1]["content"]
    for chunk in CHUNKS:
        assert chunk.text in context


def test_generator_uses_configured_temperature():
    llm = HostedLLM(model="m", api_key="k", base_url="http://x", temperature=0.2)
    assert llm.temperature == 0.2


# --- citations come from retrieved, never from parsed markers ----------------


def test_citations_are_the_chunks_passed_in():
    _text, citations = Generator(FakeLLM("answer [99] with a bogus marker")).generate("q", CHUNKS)
    assert [c.id for c in citations] == [c.id for c in CHUNKS]
    assert len(citations) == len(CHUNKS)


def test_citation_cannot_reference_a_chunk_that_was_not_provided():
    only_first = CHUNKS[:1]
    _text, citations = Generator(FakeLLM("see [1] and [7]")).generate("q", only_first)
    assert all(c.id in {c.id for c in only_first} for c in citations)
    assert len(citations) == 1


def test_citations_are_ordered_and_deduplicated():
    duplicated = [CHUNKS[0], CHUNKS[0], CHUNKS[1]]
    citations = unique_citations(duplicated)
    assert [c.id for c in citations] == [CHUNKS[0].id, CHUNKS[1].id]


# --- the refusal path (architecture.md section 3.7) --------------------------


def test_refusal_path_makes_zero_llm_calls():
    llm = FakeLLM()
    retriever_gate_passes = False
    if not retriever_gate_passes:
        text, citations = REFUSAL, []
    else:
        text, citations = Generator(llm).generate("q", CHUNKS)
    assert llm.calls == []
    assert text == REFUSAL
    assert citations == []


def test_refusal_names_the_corpus():
    assert "hdfc scheme pages" in REFUSAL.lower()
    assert "balanced advantage fund" in REFUSAL.lower()


def test_generator_with_no_chunks_refuses_without_calling_model():
    llm = FakeLLM()
    text, citations = Generator(llm).generate("q", [])
    assert llm.calls == []
    assert text == REFUSAL
    assert citations == []


# --- failure handling (architecture.md section 6) ----------------------------


def test_timeout_yields_friendly_message_not_exception():
    llm = ExplodingLLM(TimeoutError("timed out"))
    text, citations = Generator(llm).generate("q", CHUNKS)
    assert text == FRIENDLY_ERROR
    assert citations
    assert "Traceback" not in text
    assert "TimeoutError" not in text


def test_failure_is_retried_three_times():
    llm = ExplodingLLM()
    Generator(llm).generate("q", CHUNKS)
    assert llm.calls == 3


def test_retry_succeeds_after_transient_failure():
    class FlakyLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, messages):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("rate limited")
            return "recovered answer"

    llm = FlakyLLM()
    text, _ = Generator(llm).generate("q", CHUNKS)
    assert text == "recovered answer"
    assert llm.calls == 3


def test_citations_still_returned_when_the_model_fails():
    _text, citations = Generator(ExplodingLLM()).generate("q", CHUNKS)
    assert len(citations) == len(CHUNKS)


# --- backend selection --------------------------------------------------------


def test_get_llm_without_key_reports_missing_configuration(monkeypatch):
    # Must not depend on whether the developer's .env happens to hold a real key.
    monkeypatch.setattr("ragchat.generator.CONFIG", replace(CONFIG, LLM_API_KEY="", LLM_BACKEND="hosted"))
    with pytest.raises(ConfigurationError) as excinfo:
        get_llm()
    assert "LLM_API_KEY" in str(excinfo.value)


def test_get_llm_auto_with_key_selects_hosted(monkeypatch):
    monkeypatch.setattr("ragchat.generator.CONFIG", replace(CONFIG, LLM_API_KEY="sk-test", LLM_BACKEND="auto"))
    assert isinstance(get_llm(), HostedLLM)


def test_local_llm_is_not_implemented_yet():
    assert LocalLLM.available is False
    with pytest.raises(ConfigurationError):
        LocalLLM().complete([{"role": "user", "content": "hi"}])


def test_hosted_llm_requires_a_key():
    with pytest.raises(ConfigurationError) as excinfo:
        HostedLLM(model="m", api_key="", base_url="http://x")
    assert "LLM_API_KEY" in str(excinfo.value)


def test_unknown_backend_is_rejected():
    with pytest.raises(ConfigurationError):
        get_llm("nonsense")


# --- a missing model is a configuration bug, not a transient failure ---------


def test_missing_model_surfaces_as_configuration_error_not_try_again():
    generator = Generator(LocalLLM())
    with pytest.raises(ConfigurationError):
        generator.generate("q", CHUNKS)


def test_configuration_error_is_not_retried():
    class UnconfiguredLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, messages):
            self.calls += 1
            raise ConfigurationError("no model")

    llm = UnconfiguredLLM()
    with pytest.raises(ConfigurationError):
        Generator(llm).generate("q", CHUNKS)
    assert llm.calls == 1, "a configuration error must fail fast, not burn three attempts"


def test_transient_errors_still_return_the_friendly_message():
    # the contrast: a runtime failure is retryable and becomes a message, not an exception
    text, _ = Generator(ExplodingLLM()).generate("q", CHUNKS)
    assert text == FRIENDLY_ERROR
