from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ragchat.config import CONFIG
from ragchat.errors import ConfigurationError
from ragchat.ingest import run_ingest
from ragchat.models import Turn
from ragchat.pipeline import UNAVAILABLE, Pipeline, describe, startup_checks
from ragchat.vectorstore import VectorStore

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "scripts" / "demo_questions.jsonl"
IN_CORPUS = "Why do consecutive chunks overlap?"
OUT_OF_CORPUS = "What is the capital city of France?"


class ScriptedLLM:
    def __init__(self, reply: str = "Chunks overlap by 50 tokens so no sentence is cut in half.") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply


class ExplodingLLM:
    def complete(self, messages: list[dict]) -> str:
        raise RuntimeError("upstream is down")


class SlowLLM:
    """Takes measurable time, so the latency timer has something to report."""

    def complete(self, messages: list[dict]) -> str:
        time.sleep(0.05)
        return "slow but correct"


@pytest.fixture(scope="module")
def embedder():
    from ragchat.embedder import LocalEmbedder

    return LocalEmbedder()


@pytest.fixture(scope="module")
def index(tmp_path_factory, embedder):
    chroma_dir = tmp_path_factory.mktemp("pipeline") / "chroma"
    run_ingest(embedder, docs_dir=CONFIG.docs_path, chroma_dir=chroma_dir, verbose=False)
    return chroma_dir


@pytest.fixture
def pipeline(index, embedder):
    return Pipeline(embedder=embedder, store=VectorStore(path=index), llm=ScriptedLLM())


# --- the happy path -----------------------------------------------------------


def test_ask_answers_an_in_corpus_question(pipeline):
    answer = pipeline.ask(IN_CORPUS)
    assert not answer.refused
    assert "50 tokens" in answer.text


def test_ask_returns_citations_for_an_in_corpus_question(pipeline):
    answer = pipeline.ask(IN_CORPUS)
    assert answer.citations
    assert all(citation.metadata["source_file"] for citation in answer.citations)


def test_ask_keeps_the_retrieved_chunks_for_debugging(pipeline):
    answer = pipeline.ask(IN_CORPUS)
    assert answer.retrieved
    assert answer.retrieved[0].score >= answer.retrieved[-1].score


# --- the gate -----------------------------------------------------------------


def test_ask_refuses_an_out_of_corpus_question_without_calling_the_model(pipeline):
    llm = pipeline.generator.llm
    answer = pipeline.ask(OUT_OF_CORPUS)
    assert answer.refused
    assert answer.citations == []
    assert llm.calls == [], "a refused question must never reach the model"


def test_ask_returns_the_retrieved_chunks_even_when_it_refuses(pipeline):
    answer = pipeline.ask(OUT_OF_CORPUS)
    assert answer.retrieved, "refusing silently hides retrieval quality during a demo"


def test_threshold_override_changes_the_gate(index, embedder):
    permissive = Pipeline(embedder=embedder, store=VectorStore(path=index), llm=ScriptedLLM(), threshold=0.1)
    assert not permissive.ask(OUT_OF_CORPUS).refused, "a low threshold should let anything through"


# --- history ------------------------------------------------------------------


def test_history_is_coerced_from_turn_objects(pipeline):
    history = [Turn(role="user", content="What is chunking?"), Turn(role="assistant", content="Splitting a document.")]
    answer = pipeline.ask("Why do they overlap?", history)
    assert not answer.refused


def test_history_is_coerced_from_plain_dicts(pipeline):
    answer = pipeline.ask(IN_CORPUS, [{"role": "user", "content": "earlier question"}])
    assert not answer.refused


def test_turn_objects_reach_the_model(pipeline):
    pipeline.ask("And the overlap?", [Turn(role="user", content="What is chunking?")])
    sent = pipeline.generator.llm.calls[-1]
    assert any(message["content"] == "What is chunking?" for message in sent)


def test_empty_history_is_fine(pipeline):
    assert not pipeline.ask(IN_CORPUS, []).refused


# --- the error boundary -------------------------------------------------------


def test_provider_failure_becomes_friendly_text_not_a_traceback(index, embedder):
    answer = Pipeline(embedder=embedder, store=VectorStore(path=index), llm=ExplodingLLM()).ask(IN_CORPUS)
    assert answer.text
    assert "Traceback" not in answer.text
    assert answer.citations, "citations still render, so the audience can see what was retrieved"


def test_unexpected_retrieval_failure_becomes_friendly_text(pipeline, monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError("something impossible")

    monkeypatch.setattr(pipeline.retriever, "retrieve", boom)
    answer = pipeline.ask(IN_CORPUS)
    assert answer.text == UNAVAILABLE


def test_empty_index_tells_the_user_how_to_build_it(embedder, tmp_path):
    answer = Pipeline(embedder=embedder, store=VectorStore(path=tmp_path / "empty"), llm=ScriptedLLM()).ask("hi")
    assert "--commit" in answer.text
    assert "ingest" in answer.text


def test_missing_model_raises_instead_of_returning_try_again(index, embedder):
    with pytest.raises(ConfigurationError) as excinfo:
        Pipeline(embedder=embedder, store=VectorStore(path=index), llm=None).ask(IN_CORPUS)
    assert "LLM_API_KEY" in str(excinfo.value)


def test_a_refused_question_never_needs_a_model(index, embedder):
    answer = Pipeline(embedder=embedder, store=VectorStore(path=index), llm=None).ask(OUT_OF_CORPUS)
    assert answer.refused


# --- latency ------------------------------------------------------------------


def test_latency_is_reported_for_each_stage(pipeline):
    latency = pipeline.ask(IN_CORPUS).latency_ms
    assert set(latency) == {"embed", "query", "retrieve", "generate", "total"}
    assert all(value >= 0 for value in latency.values())


def test_generate_time_is_measured_separately_from_retrieval(index, embedder):
    pipeline = Pipeline(embedder=embedder, store=VectorStore(path=index), llm=SlowLLM())
    latency = pipeline.ask(IN_CORPUS).latency_ms
    assert 40 <= latency["generate"] <= 2000, "the generation stage must be timed, not reported as zero"


def test_total_latency_covers_the_whole_turn(pipeline):
    latency = pipeline.ask(IN_CORPUS).latency_ms
    assert latency["total"] >= latency["retrieve"]


# --- pre-flight ---------------------------------------------------------------


def test_preflight_never_generates(pipeline):
    llm = pipeline.generator.llm
    answer = pipeline.preflight(IN_CORPUS)
    assert answer.text == "would answer"
    assert answer.refused is False
    assert llm.calls == []


def test_preflight_reports_the_gate_for_an_out_of_corpus_question(pipeline):
    answer = pipeline.preflight(OUT_OF_CORPUS)
    assert answer.text == "would refuse"
    assert answer.refused is True


def test_preflight_runs_without_any_model_configured(index, embedder):
    assert Pipeline(embedder=embedder, store=VectorStore(path=index), llm=None).preflight(IN_CORPUS).text == "would answer"


# --- startup and describe -----------------------------------------------------


def test_startup_checks_flags_a_missing_llm(monkeypatch):
    # CONFIG is a frozen dataclass, so swap the whole object the pipeline module sees.
    monkeypatch.setattr("ragchat.pipeline.CONFIG", replace(CONFIG, LLM_API_KEY="", LLM_BACKEND="auto"))
    errors = [message for level, message in startup_checks() if level == "error"]
    assert any("LLM_API_KEY" in message for message in errors)


def test_startup_checks_returns_level_message_pairs():
    for level, message in startup_checks():
        assert level in {"error", "warning"}
        assert message.strip()


def test_describe_never_leaks_a_key():
    summary = describe()
    text = json.dumps(summary, default=str)
    assert "api_key" not in text.lower()
    assert "sk-" not in text


def test_describe_reports_the_live_index_size():
    assert describe()["indexed chunks"] >= 1


def test_describe_surfaces_the_calibrated_threshold():
    assert describe()["similarity threshold"] == pytest.approx(0.6739)


# --- the demo script ----------------------------------------------------------


def test_demo_script_drives_the_full_pipeline(pipeline):
    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    for item in questions:
        answer = pipeline.preflight(item["question"])
        if item.get("in_corpus", True):
            assert not answer.refused, f"in-corpus question was wrongly refused: {item['question']}"
        else:
            assert answer.refused, f"out-of-corpus question was wrongly answered: {item['question']}"
