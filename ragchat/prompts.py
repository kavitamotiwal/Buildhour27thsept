from __future__ import annotations

from .config import CONFIG
from .models import Chunk

SYSTEM_PROMPT = (
    "You are a facts-only assistant for HDFC Mutual Fund schemes. You answer only from the "
    "source blocks supplied in the user message, which come from official public scheme pages.\n"
    "\n"
    "Rules:\n"
    "- Answer using ONLY facts stated in the supplied sources. Do not use outside knowledge and do not guess.\n"
    "- If the sources do not contain the answer, say so plainly. Do not speculate or fill the gap.\n"
    "- Answer in AT MOST 3 SHORT SENTENCES.\n"
    "- Include EXACTLY ONE source link — the single most relevant source URL, printed as a plain URL on its own line.\n"
    "- Never give investment advice, return predictions, or scheme comparisons."
)

REFUSAL = (
    "I can only answer from the official HDFC scheme pages I was given (Large Cap, Flexi Cap, "
    "ELSS Tax Saver, Small Cap, and Balanced Advantage Fund). That question isn't covered in "
    "them. Try asking about expense ratio, exit load, minimum SIP, riskometer, benchmark, or "
    "the ELSS lock-in."
)

ADVICE_REFUSAL = (
    "I can only share facts from official scheme pages, not investment advice. "
    "For guidance on choosing funds, see the AMFI investor education page: "
    "https://www.amfiindia.com/investor-corner"
)

PII_RESPONSE = (
    "I don't ask for or store personal information (PAN, Aadhaar, account numbers, OTPs, "
    "emails, or phone numbers). Please rephrase your question using only public scheme facts."
)

FRIENDLY_ERROR = "The model didn't respond just now. Please try that again."


def _label(chunk: Chunk) -> str:
    if chunk.source_url:
        return chunk.source_url
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
