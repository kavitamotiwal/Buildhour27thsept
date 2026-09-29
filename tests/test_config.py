"""Config resolution, with an emphasis on SIMILARITY_THRESHOLD failing closed.

A fresh clone has no .env. Anything the app depends on for a safety property has to survive
that, so the refusal gate's threshold is pinned here rather than only in the integration tests.
"""

from __future__ import annotations

import pytest

from ragchat.config import CONFIG, DEFAULTS, DEFAULT_SIMILARITY_THRESHOLD, load_config


def test_threshold_default_is_the_calibrated_demo_value():
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.6694
    assert DEFAULTS["SIMILARITY_THRESHOLD"] == DEFAULT_SIMILARITY_THRESHOLD


def test_threshold_falls_back_to_the_default_when_the_env_var_is_absent(monkeypatch):
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)
    assert load_config().SIMILARITY_THRESHOLD == DEFAULT_SIMILARITY_THRESHOLD


def test_threshold_falls_back_to_the_default_when_the_env_var_is_empty(monkeypatch):
    # A half-finished `cp .env.example .env` leaves keys present but blank, which must not read
    # as "unset" in a way that reopens the gate.
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "")
    assert load_config().SIMILARITY_THRESHOLD == DEFAULT_SIMILARITY_THRESHOLD


def test_threshold_stays_overridable(monkeypatch):
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.9")
    assert load_config().SIMILARITY_THRESHOLD == 0.9


def test_threshold_is_never_none_so_the_gate_is_always_armed():
    # The old default was None, which the retriever read as "no gate". The type now says float,
    # so a regression back to None should fail here rather than silently reopen the gate.
    assert CONFIG.SIMILARITY_THRESHOLD is not None
    assert isinstance(CONFIG.SIMILARITY_THRESHOLD, float)


def test_config_print_masks_secrets():
    for name, value in CONFIG.as_dict().items():
        if any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            assert value in ("***", ""), f"{name} leaked a raw value: {value!r}"
