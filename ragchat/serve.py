"""Streamlit UI. The only pipeline entry point used here is Pipeline.ask
(implementation.md Phase 6). Deliberately imports nothing from the retriever, generator,
or embedder, so the UI cannot bypass the gate.
"""

from __future__ import annotations

import streamlit as st

from ragchat.models import Answer, Turn
from ragchat.pipeline import boot

CORPUS = """These are the course notes for this module. The bot answers only from them and
says so when a question falls outside them."""

GROUNDING = "This bot answers only from the indexed course notes. It is designed to refuse rather than guess."

st.set_page_config(page_title="Course Notes RAG Bot", page_icon="📚", layout="centered")

MIN_QUESTION_LENGTH = 2


@st.cache_resource(show_spinner="Loading the index and the model...")
def get_boot():
    """Built once per process. Reset must NOT clear this, or the index reloads."""
    return boot()


def looks_like_a_question(text: str) -> bool:
    """Reject empty or garbled input before it costs an embedding call."""
    stripped = text.strip()
    if len(stripped) < MIN_QUESTION_LENGTH:
        return False
    return any(character.isalnum() for character in stripped)


def thread() -> list[dict]:
    st.session_state.setdefault("messages", [])
    return st.session_state["messages"]


def history() -> list[Turn]:
    return [Turn(role=message["role"], content=message["content"]) for message in thread()]


def render_sources(answer: Answer) -> None:
    """Collapsed expander with the raw chunks behind the answer."""
    if not answer.retrieved:
        return
    top_score = answer.retrieved[0].score
    with st.expander(f"Sources ({len(answer.retrieved)}) · top score {top_score:.2f}"):
        for hit in answer.retrieved:
            chunk = hit.chunk
            st.markdown(
                f"**[{hit.rank}] {chunk.source_file} · {chunk.anchor}**  ·  score `{hit.score:.4f}`"
            )
            st.caption(chunk.text)
            st.divider()


def render_assistant_turn(answer: Answer) -> None:
    st.markdown(answer.text)

    if answer.citations:
        row = "  ·  ".join(f"{chunk.source_file} · {chunk.anchor}" for chunk in answer.citations)
        st.caption(row)

    if answer.refused and answer.retrieved:
        st.caption(f"Best match scored {answer.retrieved[0].score:.4f}, below the refusal threshold.")

    render_sources(answer)

    total = answer.latency_ms.get("total", 0)
    if total:
        st.caption(f"{total} ms")


def render_banner(issues: list[tuple[str, str]]) -> None:
    """Startup problems shown without blocking the app."""
    for level, message in issues:
        if level == "error":
            st.error(message, icon="⚠️")
        else:
            st.warning(message, icon="⚠️")


def render_sidebar(config: dict) -> None:
    with st.sidebar:
        st.header("About")
        st.markdown(CORPUS)

        st.header("Configuration")
        for key, value in config.items():
            st.markdown(f"**{key}**: {value}")

        st.header("Thread")
        if st.button("Reset conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


def main() -> None:
    pipeline, issues, config = get_boot()

    st.title("📚 Course Notes RAG Bot")
    render_banner(issues)

    for message in thread():
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_turn(message["answer"])
            else:
                st.markdown(message["content"])

    typed = st.chat_input("Ask a question about the course notes")
    if typed is not None:
        if not looks_like_a_question(typed):
            st.warning("Type a real question to continue.")
        else:
            # Capture history BEFORE recording the new question, so the prompt does not
            # contain it twice.
            prior = history()
            question = typed.strip()
            thread().append({"role": "user", "content": question})

            try:
                with st.spinner("Retrieving and answering…"):
                    answer = pipeline.ask(question, prior)
            except Exception:
                # No stack trace may be reachable from the UI.
                st.error("Something went wrong answering that. Try rephrasing, or reset the thread.")
            else:
                thread().append({"role": "assistant", "content": answer.text, "answer": answer})
            st.rerun()

    render_sidebar(config)
    st.divider()
    st.caption(GROUNDING)


main()
