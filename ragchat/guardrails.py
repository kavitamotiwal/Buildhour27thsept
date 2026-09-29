"""Pre-generation guardrails (PRD sections 6.3, 8.2 step 3).

Every question is classified before retrieval costs an embedding call: PII-bearing inputs
are answered with a safe generic response and never persisted, opinion/advice questions are
refused with an educational link, questions about schemes outside the fixed 5-page list are
refused outright, and only the rest reach the retrieval gate.
"""

from __future__ import annotations

import re
from typing import Literal

QuestionKind = Literal["factual", "advice", "pii"]

_ANY_PAN = r"[A-Z]{5}[0-9]{4}[A-Z]"
_ANY_AADHAAR = r"[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}"
_ANY_EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
_ANY_PHONE = r"[6-9][0-9]{9}"

PII_PATTERNS = (
    re.compile(rf"\b{_ANY_PAN}\b"),
    re.compile(rf"\b{_ANY_AADHAAR}\b"),
    re.compile(rf"\b{_ANY_EMAIL}\b"),
    re.compile(rf"\b{_ANY_PHONE}\b"),
)

# Token-level, boundary-safe. Notes:
# - "account number" / "bank account" cover balances, folio and beneficiary phrasing.
# - Bare "pan" is NOT a keyword: it matches inside words like "expand" or "Hispanic",
#   so it is caught only in its full PAN-alphanumeric form by the regex above.
PII_KEYWORDS = re.compile(
    r"\b(?:pan number|pan no|aadhaar|aadhar|otp|account number|bank account|folio number)\b",
    re.IGNORECASE,
)

# Deliberately phrase-level, not single words: "in this fund" must not trip a bare "invest"
# substring by itself. Factual lookup questions ("What is the minimum SIP?") never contain
# any of these full phrases.
ADVICE_MARKERS = (
    "should i", "should we", "should you", "is it a good idea", "recommend", "advise",
    "advice", "better than", "is better", "worth", "good time", "best fund", "top fund",
    "buy or sell", "buy or not", "sell or hold", "which fund", "switch", "predict",
    "expected return", "returns will", "will give", "opinion", "compare", "comparison",
    "portfolio of", "should i invest", "should i buy", "should i sell", "good to invest",
    "returns next year", "returns in the future",
)

# The corpus is a fixed 5-scheme list (PRD section 6.1). The embedding alone cannot tell
# "HDFC Short Term Fund" (absent) from "HDFC Small Cap Fund" (present) — both embed near
# our expense-ratio chunks. So an explicit scope guard runs before retrieval: a question
# must name at least one in-scope scheme (or HDFC MF generically) and no other AMC/brand.
SIGNATURE_PHRASES = (
    "large cap", "flexi cap", "elss tax saver", "elss", "small cap", "balanced advantage",
    "hdfc mutual fund", "hdfc fund", "hdfc scheme",
)

# Other AMC brands. Word-boundary regex: bare "sbi" would otherwise hit inside words like
# "subsidence".
_OTHER_AMC_RE = re.compile(r"\b(?:sbi|icici|nippon|axis|kotak|quant|tata|mirae|sundaram|dsp|uti|pgim)\b")

# HDFC schemes NOT in the fixed list. Plain substrings are safe here: each is a distinctive
# "hdfc <name>" fragment and cannot appear inside an in-scope scheme name.
_OTHER_HDFC_SCHEMES = (
    "hdfc short term", "hdfc short duration", "hdfc mid cap", "hdfc midcap",
    "hdfc mid & small", "hdfc large & mid cap", "hdfc large and mid cap", "hdfc focused",
    "hdfc value", "hdfc equity savings", "hdfc arbitrage", "hdfc conservative hybrid",
    "hdfc aggressive hybrid", "hdfc multi cap", "hdfc banking", "hdfc technology",
    "hdfc corporate bond", "hdfc credit risk", "hdfc money market", "hdfc liquid fund",
    "hdfc overnight", "hdfc defence", "hdfc innovation", "hdfc consumption",
    "hdfc manufacturing", "hdfc infrastructure", "hdfc business cycle", "hdfc pharma",
    "hdfc transportation", "hdfc children", "hdfc housing", "hdfc financial services",
)


def classify_scope(question: str) -> bool:
    """True when the question is about the 5 in-scope HDFC schemes (PRD section 6.1).

    A question naming any other AMC, or any HDFC scheme outside the fixed list, is refused
    even when its wording otherwise resembles an in-corpus question ("What is the expense
    ratio of HDFC Short Term Fund?" embeds at 0.82 against our expense-ratio chunks).
    """
    lowered = question.lower()
    if _OTHER_AMC_RE.search(lowered):
        return False
    if any(name in lowered for name in _OTHER_HDFC_SCHEMES):
        return False
    return any(phrase in lowered for phrase in SIGNATURE_PHRASES)


def classify_question(question: str) -> QuestionKind:
    """Classify a question as factual, advice-seeking, or PII-bearing.

    PII wins first: it must never be echoed, stored, or passed to a model. Advice is a
    "just refuse" decision too, but only after PII has been ruled out.
    """
    if any(pattern.search(question) for pattern in PII_PATTERNS):
        return "pii"
    if PII_KEYWORDS.search(question):
        return "pii"
    lowered = question.lower()
    if any(marker in lowered for marker in ADVICE_MARKERS):
        return "advice"
    return "factual"


__all__ = ["classify_question", "classify_scope", "QuestionKind"]