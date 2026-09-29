# Course Notes RAG Bot

A retrieval-augmented chatbot over a fixed set of course notes. It answers **only** from an
indexed corpus, cites the exact chunks it used, and refuses outright when a question falls
outside the material.

Hand-rolled pipeline on Chroma — no LangChain, no LlamaIndex, no external vector service.

## Why it refuses

The interesting part of this project is not that it answers, it's that it *declines*. A bot
that answers everything is a bot you cannot trust. So retrieval is gated on a calibrated
similarity threshold, and questions that fall below it are refused **without ever reaching the
LLM** — no prompt, no tokens, no chance of a plausible-sounding guess.

The 20-question demo set is 15 in-corpus and 5 out-of-corpus. Those 5 are the demonstration.

## Privacy

**Your document text is sent to a third-party model provider.** When a question is answered, the
retrieved chunks (not your whole library — only the top-k) are placed in the prompt and sent to
the LLM endpoint you configured. The embedding backend can be fully local; the LLM generally is
not.

Run it with no `LLM_API_KEY` and it will refuse everything with a message naming the missing
variable, rather than falling back to sending data somewhere unexpected.

## Prerequisites

- Python 3.11+ (developed on 3.14)
- ~65 MB for the embedding model, downloaded once (see the cache note below)
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
| `EMBEDDING_API_KEY` | hosted embeddings only | leave blank to embed locally, no network |
| `LLM_API_KEY` | answering | blank means "refuse everything" |
| `LLM_BASE_URL` | non-OpenAI providers | **blank means OpenAI.** See below |
| `LLM_MODEL` | answering | exactly as the provider spells it |

`LLM_BASE_URL` is the easiest thing to get wrong. Leave it blank and your key goes to OpenAI,
which returns a 401 for any other provider's key.

```
OpenAI        (blank)              -> gpt-4o-mini
Groq          https://api.groq.com/openai/v1
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
python -m ragchat.cli ask "what is a vector embedding"
python -m ragchat.cli retrieve "what is a vector embedding"   # scored chunks, no LLM
python -m ragchat.cli chat                                    # terminal chat
python -m ragchat.cli run-demo                                # the 20-question script
python -m ragchat.cli run-demo --check                        # gate only, no LLM needed
python -m ragchat.cli calibrate                                # re-derive the threshold
```

`retrieve` and `run-demo --check` never call the LLM, so you can rehearse the retrieval half of
the demo with no key and no network.

## How retrieval works

1. The question is conditioned on recent conversation turns, concatenated into one string.
2. That string is embedded once. No query-rewrite LLM call.
3. Cosine top-k against Chroma.
4. `best_score` is compared to `SIMILARITY_THRESHOLD`. Below it: refusal, no LLM call.
5. Above it: the chunks become a numbered, fenced source block in the prompt.

History is prepended **only** when the question actually refers back — "why does *that* work?".
An unrelated question stays stateless. This guard is load-bearing: with history prepended
unconditionally, a stale topic dragged an unrelated question from 0.44 to 0.82, straight past
the gate.

### Two history windows

| Window | Default | Used for |
|---|---|---|
| `RETRIEVAL_HISTORY_TURNS` | 10 | the embedding's query |
| `HISTORY_TURNS` | 3 | the model prompt |

They differ deliberately. The embedding needs the older referent to resolve a follow-up; the
model only needs recent turns to read four source blocks.

### The threshold

`SIMILARITY_THRESHOLD` is measured, not guessed. Against the bundled 9-chunk corpus with
`BAAI/bge-small-en-v1.5`: in-corpus scores land in `[0.748, 0.835]`, out-of-corpus in
`[0.442, 0.600]`. The threshold `0.6739` is the midpoint.

That `0.6739` is also the built-in fallback (`DEFAULT_SIMILARITY_THRESHOLD` in
`ragchat/config.py`). `SIMILARITY_THRESHOLD` is optional and merely overrides it; when the
variable is absent, the application uses the default and the refusal gate stays closed, so an
unconfigured install refuses rather than answering out of corpus.

Re-run `python -m ragchat.cli calibrate` after changing `EMBEDDING_MODEL`, `CHUNK_SIZE`,
`CHUNK_OVERLAP`, or the corpus. If the two score ranges overlap, **fix the chunking** — do not
nudge the number until the demo passes.

## Tests

```bash
pytest -q
```

175 pass, 1 xfailed. No API key is required: the suite uses the local embedding backend and
fake LLMs. The *first* run does need network, to fetch the embedding model — see below.

### Where the embedding model is cached

`fastembed` caches into a temp directory (`%LOCALAPPDATA%\Temp\fastembed_cache` on Windows,
`/tmp` elsewhere) rather than a durable cache. Windows clears temp periodically, so expect to
re-download the ~65 MB model occasionally, and expect the first question of a session to be slow
while it does. To make it durable, point `TextEmbedding` at a real directory in
`ragchat/embedder.py:100`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "The vector index is empty or missing" | never ingested | `python -m ragchat.ingest --commit` |
| "No LLM configured. Set LLM_API_KEY" | no key | add `LLM_API_KEY` to `.env` |
| 401 from the provider | `LLM_BASE_URL` blank, non-OpenAI key | set the base URL explicitly |
| Every question refused | threshold too high, or index not built | `python -m ragchat.cli calibrate`, check `retrieve` scores |
| Every question answered, nothing refused | unlikely with default config: an unset `SIMILARITY_THRESHOLD` falls back to the built-in `0.6739` and the gate stays fail-closed | if in-corpus questions are being refused, run `python -m ragchat.cli calibrate` and inspect `retrieve` scores — the cause is more likely threshold or retrieval quality than an open gate |
| First question slow, later ones fast | one-off local model load | expected; steady state is well under the 5s target |
| Slow again after a few days | model was cached in temp and got cleared | expected; re-downloads once. See the cache note above |
| Slow on every question | hosted embeddings, or a large top-k | switch to local embeddings, or lower `TOP_K` |
| Streamlit won't start | module not found | `pip install -r requirements.txt` |

## Demo script

```bash
python -m ragchat.cli run-demo          # full run
python -m ragchat.cli run-demo --check  # retrieval + gate only
```

`scripts/demo_questions.jsonl` holds 20 questions: 10 lookups, 2 definitions, 3 cross-section
relations, and 5 out-of-corpus (2 plain misses, 3 near-misses that sound plausible but are
genuinely absent — e.g. a Docker install question on notes about vector stores).

The near-misses are the ones to lead with. "What is the capital of France?" is an easy refusal
and proves little. A question that *should* retrieve, scores 0.60, and gets declined anyway is
what shows the gate is doing real work.

## Layout

```
ragchat/     config, loaders, chunker, embedder, vectorstore, retriever,
             prompts, generator, pipeline (the orchestrator), serve (UI), cli
documents/   the corpus: .md, .txt, .pdf
scripts/     demo_questions.jsonl, calibrate_threshold.py, make_sample_pdf.py
data/        generated: chroma index, manifest, export. All gitignored.
tests/       167 tests
```

`data/` is entirely generated. Delete it and re-run ingest to rebuild.

`ragchat/pipeline.py` is the single orchestrator. The UI calls `Pipeline.ask` and nothing else,
so the UI cannot bypass the threshold gate.
