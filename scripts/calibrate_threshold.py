"""Threshold calibration per architecture.md section 5.3.

Delegates to `python -m ragchat.cli calibrate`, which owns the scoring and reporting.
Exits non-zero when the in-corpus and out-of-corpus score populations overlap, because
overlap means retrieval quality is the problem, not the threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragchat.cli import main  # noqa: E402

QUESTIONS = Path(__file__).resolve().parent / "demo_questions.jsonl"


if __name__ == "__main__":
    raise SystemExit(main(["calibrate", "--questions", str(QUESTIONS)]))
