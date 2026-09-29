# HDFC Mutual Fund Assistant

A facts-only, cited-retrieval chatbot over five official HDFC Mutual Fund scheme pages
(Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, Balanced Advantage). It answers **only**
from those pages, keeps every answer to **≤3 sentences**, shows **exactly one source link**,
and appends **"Last updated from sources: <date>"**. Opinion, advice, return-comparison, and
PII-bearing questions are refused before any retrieval.

Hand-rolled RAG pipeline on Chroma — no LangChain, no LlamaIndex, no external vector service.
Spec: `PRD.md`. Design: `architecture.md`.

## Why it refuses

There are two independent refusal layers:

1. **Guardrails (pre-retrieval).** `ragchat/guardrails.py` classifies every question.
   PII (PAN, Aadhaar, OTP, account details) gets a safe generic response and is never sent to
   a model or written to disk. Advice/opinion/return-comparison questions ("should I buy X?",
   "which fund is better?") get a polite facts-only refusal with an AMFI education link.
2. **Similarity gate (post-retrieval).** Questions are embedded and compared against
   `SIMILARITY_THRESHOLD`. Below it, the answer is refused **without ever reaching the LLM** —
   no prompt, no tokens, no chance of a plausible-sounding guess.

## Privacy

**Your retrieved chunks are sent to the LLM provider you configure** (`LLM_BASE_URL` +
`LLM_API_KEY`). Only the top-k chunks on the retrieved scheme pages ever enter the prompt.
The embedding backend is fully local (fastembed + MiniLM). Run with no `LLM_API_KEY` and the
app refuses cleanly, naming the missing variable.

Personal information is out of scope by construction: the guardrail refuses it, nothing is
logged. User inputs survive only transiently in the browser session.

## Source list (the corpus)

Each file in `documents/` is a cleaned copy of one public scheme page, with `---` front matter
carrying `scheme`, `category`, `source_url`, and `fetch_date`:

| Scheme | Category | Source URL |
|---|---|---|
| HDFC Large Cap Fund – Direct Growth | Equity Large Cap | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| HDFC Flexi Cap Fund – Direct Growth | Equity Flexi Cap | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| HDFC ELSS Tax Saver Fund – Direct Plan Growth | Equity ELSS | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| HDFC Small Cap Fund – Direct Growth | Equity Small Cap | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| HDFC Balanced Advantage Fund – Direct Growth | Hybrid Dynamic Asset Allocation | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

## Prerequisites

- Python 3.11+ (developed on 3.14)
- ~25 MB for the MiniLM embedding model, downloaded once (cache note below)
- An LLM API key, for answers. Retrieval alone needs none.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
```

Then edit `.env`:

| Variable | Needed for | Notes |
|---|---|---|
| `EMBEDDING_BACKEND` | everything | `auto` picks local when no key is set |
| `EMBEDDING_MODEL` | nothing | blank → local `sentence-transformers/all-MiniLM-L6-v2` |
| `LLM_API_KEY` | answering | blank means "refuse everything" |
| `LLM_BASE_URL` | non-OpenAI providers | **blank means OpenAI.** See below |
| `LLM_MODEL` | answering | exactly as the provider spells it |

`LLM_BASE_URL` is the easiest thing to get wrong. Leave it blank and your key goes to OpenAI,
which returns a 401 for any other provider's key.

```
OpenAI        (blank)              -> gpt-4o-mini
Groq          https://api.groq.com/openai/v1    (openai/gpt-oss-120b works well)
OpenRouter    https://openrouter.ai/api/v1
Ollama        http://localhost:11434/v1
```

## Build the index

```bash
python -m ragchat.ingest --commit
python -m ragchat.ingest --dry-run     # counts only, writes nothing
```

Ingest is all-or-nothing: one unparseable file aborts the build and names the file.

## Run

```bash
streamlit run ragchat/serve.py         # web UI on http://localhost:8501
```

Headless equivalents, useful for demos and debugging:

```bash
python -m ragchat.cli ask "What is the expense ratio of HDFC Small Cap Fund?"
python -m ragchat.cli retrieve "What is the exit load of HDFC Large Cap Fund?"   # scored chunks, no LLM
python -m ragchat.cli chat                                    # terminal chat
python -m ragchat.cli run-demo                                # the labeled question script
python -m ragchat.cli run-demo --check                        # gate only, no LLM needed
python -m ragchat.cli calibrate                                # re-derive the threshold
```

`retrieve` and `run-demo --check` never call the LLM, so you can rehearse the retrieval half of
the demo with no key and no network.

## How retrieval works

1. Guardrails classify the question (PII / advice / factual) — non-factual stops here.
2. The question is conditioned on recent conversation turns when it refers back.
3. The query string is embedded once. No query-rewrite LLM call.
4. Cosine top-k against Chroma.
5. `best_score` ≥ `SIMILARITY_THRESHOLD` → generate; below → refusal, no LLM call.
6. The chunks become a numbered, fenced source block; the model answers in ≤3 sentences with
   one source URL; the generator appends the deterministic "Last updated" footer.

### Two history windows

| Window | Default | Used for |
|---|---|---|
| `RETRIEVAL_HISTORY_TURNS` | 10 | the embedding's query |
| `HISTORY_TURNS` | 3 | the model prompt |

They differ deliberately. The embedding needs the older referent to resolve a follow-up; the
model only needs recent turns to read four source blocks.

### The threshold

`SIMILARITY_THRESHOLD` is measured, not guessed. Against this 5-page corpus with MiniLM, the
calibrated value lives in `ragchat/config.py` (`DEFAULT_SIMILARITY_THRESHOLD`). That number is
also the built-in fallback, so an unconfigured install refuses rather than answering out of
corpus.

Re-run `python -m ragchat.cli calibrate` after changing `EMBEDDING_MODEL`, `CHUNK_SIZE`,
`CHUNK_OVERLAP`, or the corpus. If the in-corpus and out-of-corpus score ranges overlap,
**fix the chunking/data** — do not nudge the number until the demo passes.

## Tests

```bash
pytest -q
```

No API key is required: the suite uses the local embedding backend and fake LLMs. The *first*
run does need network, to fetch the MiniLM model — see below.

### Where the embedding model is cached

`fastembed` caches into a temp directory (`%LOCALAPPDATA%\Temp\fastembed_cache` on Windows,
`/tmp` elsewhere) rather than a durable cache. Windows clears temp periodically, so expect the
first question of a session to be slow while it re-downloads. To make it durable, point
`TextEmbedding` at a real directory in `ragchat/embedder.py`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "The vector index is empty or missing" | never ingested | `python -m ragchat.ingest --commit` |
| "No LLM configured. Set LLM_API_KEY" | no key | add `LLM_API_KEY` to `.env` |
| 401 from the provider | `LLM_BASE_URL` blank, non-OpenAI key | set the base URL explicitly |
| Every factual question refused | threshold too high, or index stale | `python -m ragchat.cli calibrate`, check `retrieve` scores |
| Advice questions get answered | guardrail markers missed a phrasing | rephrase the test; extend `guardrails.py` |
| PII gets echoed | should never happen — bug | report; the guardrail runs before the model |
| First question slow, later ones fast | one-off local model load | expected |
| Slow again after a few days | model cached in temp and cleared | expected; re-downloads once |
| Streamlit won't start | module not found | `pip install -r requirements.txt` |

## Demo script

```bash
python -m ragchat.cli run-demo          # full run
python -m ragchat.cli run-demo --check  # retrieval + gate + guardrails only
```

`scripts/demo_questions.jsonl` holds the labeled set: per-scheme factual lookups (expense
ratio, exit load, minimum SIP, riskometer, benchmark, ELSS lock-in), out-of-corpus non-scheme
questions, opinion/advice questions, and PII probes. The refusals are the point — especially
the near-miss out-of-corpus ones that *sound* plausible and get declined anyway.

## Layout

```
ragchat/     config, loaders, chunker, embedder, vectorstore, retriever, guardrails,
             prompts, generator, pipeline (the orchestrator), serve (UI), cli
documents/   5 cleaned HDFC scheme pages (.md with front matter)
scripts/     demo_questions.jsonl, calibrate_threshold.py
data/        generated: chroma index, manifest, export. All gitignored.
tests/       ~201 tests
```

`data/` is entirely generated. Delete it and re-run ingest to rebuild. `ragchat/pipeline.py`
is the single orchestrator; the UI calls `Pipeline.ask` and nothing else, so the UI cannot
bypass the guardrails or the threshold gate.