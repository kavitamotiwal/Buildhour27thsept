# Implementation Plan: HDFC Mutual Fund FAQ RAG Chatbot (Class Demo)

Status: **all phases implemented.** Spec: `PRD.md`. Design: `architecture.md`.

## 0. How To Use This File

- Phase gates are "working and demonstrated", not "code touched".
- Every new behavior ships with a test that pins it.

## Phase 0 — Scaffold, Config, Data Contracts

**Tasks:** package layout (`ragchat/`), `config.py` env loader, pydantic models
(`Chunk`, `ScoredChunk`, `RetrieveResult`, `Answer`, `Turn`), `python -m ragchat.cli`.
**Working rules:** never amend/reset/rebase/force-push; never print `.env`; `.env` stays
gitignored; commit only requested changes.

## Phase 1 — Loaders and Chunker (FR1a)

**Tasks:** `loaders.py` (PDF page segments, markdown-per-heading segments, text whole-file)
+ front matter parser (`source_url`, `fetch_date`, `scheme`, `category` on every segment);
`chunker.py` token windows, never spanning segments, stable ids.
**Result:** 5 scheme markdown files in `documents/`, one heading per fact field.

## Phase 2 — Embedder, Vector Store, Ingest CLI (FR1b)

**Tasks:** `embedder.py` (fastembed local MiniLM or OpenAI-compatible hosted),
`vectorstore.py` (persistent Chroma, cosine, normalization, reset-on-ingest),
`ingest.py` (dry-run + `--commit`, writes `data/manifest.json`).
**Verification:** `python -m ragchat.ingest --dry-run`, then `--commit`; manifest matches.

## Phase 3 — Retriever and Threshold Calibration (FR2/FR5)

**Tasks:** `retriever.py` (history-conditioned embedding, top-k, gate);
`scripts/demo_questions.jsonl` (in/out-of-corpus labels + expected source);
`scripts/calibrate_threshold.py` recommends `SIMILARITY_THRESHOLD`.
**Result:** calibrated default in `config.DEFAULT_SIMILARITY_THRESHOLD`, mirror set in
`scripts/demo_questions.jsonl`.

## Phase 4 — Prompts and Generator (FR2, FR3, FR4)

**Tasks:** `prompts.py` (facts-only system prompt, context fences, REFUSAL, ADVICE_REFUSAL,
PII_RESPONSE); `generator.py` HostedLLM (retries, honest User-Agent), `unique_citations`
(pass-through, never parsed markers), `last_updated_footer` (newest `fetch_date`).
**Constraints:** ≤3 sentences; exactly one source URL; footer appended deterministically.

## Phase 5 — Pipeline Orchestrator

**Tasks:** `pipeline.py` — `ask()` = guardrails → retrieve → gate → generate; `preflight()`
for offline checks; `boot()`/`startup_checks()`/`describe()` for the UI.
**Guardrails:** `guardrails.py` classifies PII → `PII_RESPONSE`, advice/comparison →
`ADVICE_REFUSAL` (AMFI link) — both before any model call.

## Phase 6 — Streamlit UI (FR8)

**Tasks:** `ragchat/serve.py` — "HDFC Mutual Fund Assistant", welcome line, 3 example
questions, "Facts-only. No investment advice." disclaimer, chat with citations rendered as
source links plus a retrieved-passage expander.
**Rule:** the UI imports only `Pipeline.ask`/`boot` — it cannot bypass the gate.

## Phase 7 — Demo Hardening (FR8, NFRs)

**Tasks:** refusal/guardrail golden tests, threshold re-run on the final corpus, Render
deploy (streamlit + `$PORT`), a prepared sample-Q&A sheet for the demo.

## Appendix A — Phase-to-Requirement Traceability

| Phase | Requirements |
|---|---|
| 0–3 | FR1 (facts-only retrieval), FR5 (out-of-corpus gate) |
| 4 | FR2 (one source link), FR3 (≤3 sentences), FR4 (last-updated footer) |
| 5 | FR5/FR6 (advice + comparison refusal), FR7 (PII) |
| 6 | FR8 (welcome, 3 examples, disclaimer) |

## Appendix B — Time Budget

Phases 0–7 are a single day's sprint; each phase is independently demonstrable.

## Appendix C — Definition of Done (whole project)

- `python -m ragchat.ingest --commit` builds the index from `documents/`.
- `scripts/calibrate_threshold.py` exits 0 with separated score populations.
- `pytest` green.
- CLI: retrieval, gate refusal, opinion refusal, PII refusal, and generated answers all
  demonstrable.
- UI: 3 example questions, disclaimers, source links, refusal paths.
- Deployed to Render; README documents setup, corpus, and limitations.