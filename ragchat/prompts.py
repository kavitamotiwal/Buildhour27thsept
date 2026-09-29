from __future__ import annotations

from .config import CONFIG
from .models import Chunk

SYSTEM_PROMPT = (
    "You are a course-notes assistant. You answer questions about one fixed set of course "
    "material, and you answer only from the numbered sources supplied in the user message.\n"
    "\n"
    "Rules:\n"
    "- Use only facts stated in those sources. Do not use outside knowledge and do not guess.\n"
    "- If the sources do not contain the answer, say so plainly. Do not speculate or fill the gap.\n"
    "- Refer to the sources you used by number, for example [1], so the reader can trace the claim.\n"
    "- Be concise and factual. Match the register of the source material."
)

REFUSAL = (
    "I can only answer from the course notes I was given, and that question isn't covered in them. "
    "Try asking about the material in those notes."
)

FRIENDLY_ERROR = "The model didn't respond just now. Please try that again."


def _label(chunk: Chunk) -> str:
    if "page" in chunk.metadata:
        location = f"page {chunk.metadata['page']}"
    else:
        location = f'section "{chunk.metadata.get("section", "")}"'
    return f"{chunk.metadata.get('source_file', '')} | {location}"


def build_context(chunks: list[Chunk]) -> str:
    """Numbered, fenced source block.

    Fences are numbered rather than XML-ish so a retrieved passage cannot open or close a
    fence whose name it does not know (architecture.md section 3.6).
    """
    if not chunks:
        return "<<<NO SOURCES PROVIDED>>>"
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"<<<SOURCE {number}>>>\n"
            f"source: {_label(chunk)}\n"
            f"{chunk.text}\n"
            f"<<<END {number}>>>"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[Chunk], history: list[dict] | None = None) -> list[dict]:
    """Ordered chat messages: system, context block, recent history, current question."""
    turns = CONFIG.HISTORY_TURNS
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Here are the sources:\n\n{build_context(chunks)}"},
    ]
    for message in list(history or [])[-2 * turns :]:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        if content and role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages
