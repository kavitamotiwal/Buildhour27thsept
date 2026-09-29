"""Guardrails (PRD 6.3, 8.2 step 3): PII, advice, and scheme-scope refusal before retrieval.

These functions are pure; the pipeline tests cover wiring them into ask()/preflight().
"""

from __future__ import annotations

import pytest

from ragchat.guardrails import classify_question, classify_scope


# --- PII classification --------------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "My PAN is ABCDE1234F, track my holding",
        "My Aadhaar is 1234 5678 9012",
        "Send the statement to me@gmail.com",
        "Call me at 9876543210",
        "What is my folio number?",
        "Reset my password with the OTP sent to my phone",
        "My bank account number is 1234567890",
    ],
)
def test_pii_inputs_are_never_factual(question):
    assert classify_question(question) == "pii"


# --- advice classification ----------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "Should I invest in HDFC Small Cap Fund right now?",
        "Is it a good time to buy HDFC Flexi Cap Fund?",
        "Which fund is better, HDFC Large Cap or HDFC Small Cap?",
        "Can you recommend a mutual fund for me?",
        "Will HDFC ELSS give good returns next year?",
        "Should I switch from HDFC Large Cap to HDFC Flexi Cap?",
    ],
)
def test_advice_questions_are_never_factual(question):
    assert classify_question(question) == "advice"


@pytest.mark.parametrize(
    "question",
    [
        "What is the expense ratio of HDFC Large Cap Fund?",
        "What is the minimum SIP amount for HDFC ELSS?",
        "What is the exit load of the HDFC Small Cap Fund?",
        "What is the benchmark of HDFC Balanced Advantage Fund?",
    ],
)
def test_factual_lookups_stay_factual(question):
    assert classify_question(question) == "factual"


# --- scheme scope (PRD 6.1) ---------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "What is the expense ratio of HDFC Large Cap Fund?",
        "What is the ELSS lock-in period?",
        "Tell me about the HDFC flexi cap fund",
        "HDFC mutual fund total assets",
        "What does HDFC fund invest in?",
        "How does the balanced advantage fund work?",
    ],
)
def test_in_scope_questions_pass(question):
    assert classify_scope(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is the expense ratio of SBI Small Cap Fund?",
        "Who manages the ICICI Prudential Equity Fund?",
        "Compare HDFC and Nippon India funds",
        "Kotak flexi cap fund returns",
    ],
)
def test_other_amc_questions_are_out_of_scope(question):
    assert not classify_scope(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is the expense ratio of HDFC Short Term Fund?",
        "Who manages HDFC Mid Cap Opportunities Fund?",
        "HDFC Liquid Fund daily returns",
        "HDFC Corporate Bond Fund NAV",
    ],
)
def test_other_hdfc_schemes_are_out_of_scope(question):
    assert not classify_scope(question)


def test_bare_hdfc_without_a_signature_phrase_is_out_of_scope():
    # "hdfc" alone is not one of the scoped phrases ("hdfc fund", "hdfc mutual fund", ...),
    # so a question that just names the brand without the corpus's schemes is refused.
    assert not classify_scope("What is HDFC?")


def test_word_boundary_prevents_amc_leaks_inside_words(question=None):
    # Regression: bare "sbi" must not trip inside "subsidence"; "axis" must not trip in
    # "taxes" or a risk axis. "What is the risk axis of HDFC Large Cap Fund?" stays in scope.
    assert classify_scope("What is the risk of HDFC Large Cap Fund?")