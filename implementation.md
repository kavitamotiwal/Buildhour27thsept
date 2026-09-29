# Implementation Plan: Course RAG Chatbot (Class Demo)

**Guides:** `PRD.md` (scope) → `architecture.md` (design) → this file (execution)
**How to use:** Each phase is a self-contained unit with a copy-pasteable prompt for Cursor.
Complete and verify one phase before starting the next. **Do not start a phase before the
previous phase's verification passes** — the ordering exists because retrieval problems are
cheapest to fix before an LLM is in the loop (PRD M2).

---

## 0. How To Use This File

### 0.1 Working rules for the coding agent

Paste this at the top of every Cursor session. It keeps the agent inside the designed scope.

```
CONTEXT
- Read PRD.md and architecture.md first. This task implements one phase of a RAG chatbot.
- architecture.md is authoritative for design. If a task here and architecture.md disagree,
  architecture.md wins — flag the conflict instead of guessing.

HARD CONSTRAINTS
- Stay inside the phase you are given. Do not implement later phases, do not "improve" the
  design, do not add dependencies, do not add auth, upload, streaming, reranking, or hybrid
  search. Those are PRD non-goals.
- Hand-rolled pipeline, no LangChain/LlamaIndex (architecture.md §5.1).
- Chroma via its Python client, local and persistent. No external vector service.
- Every tunable in config.py, read from env with a default. No magic numbers inline.
- Never print, log, or return API keys. No secrets in code or committed files.
- Type hints on all public functions. No bare except Exception outside the boundaries
  described in architecture.md §6.
- No comments that restate the code. Comment only the non-obvious *why*.

WORKING STYLE
- Implement the smallest thing that satisfies the phase's acceptance criteria.
- Print real values (scores, counts, file names) so verification is possible.
- If a task is ambiguous, choose the simplest reading and note the assumption in your summary.

BEFORE YOU FINISH
- Run the phase's verification commands and paste real output.
- List every file you created or changed, and every assumption you made.
```

### 0.2 Phase gates

| Phase | Deliverable | Must pass before moving on |
|-------|-------------|---------------------------|
| 0 | Repo scaffold, config, data contracts | `python -m ragchat.config` prints resolved config |
| 1 | Loaders + chunker (FR1a) | Unit tests green, no network needed |
| 2 | Embedder + vector store + ingest CLI (FR1b) | Index built, count matches, re-ingest is identical |
| 3 | Retriever + threshold (FR2, FR5) | 15 in-corpus hit, 5 out-of-corpus miss, threshold calibrated |
| 4 | Prompts + generator (FR3, FR4) | Golden Q&A answers with correct citations |
| 5 | Pipeline orchestrator | 20-question demo script runs clean end to end |
| 6 | Streamlit UI (FR7) | Chat, citations, sources expander, reset all work |
| 7 | Demo hardening (FR8, NFRs) | Cold setup < 15 min, offline path rehearsed |

Phases 0–3 need **no LLM**. That is deliberate: get retrieval demonstrably right before
spending money or time on prompts.

---

## Phase 0 — Scaffold, Config, Data Contracts

**Maps to:** PRD M1 (setup), architecture.md §3.1, §3.2, §3.9 (config surface), §6 (startup checks)
**Needs:** nothing
**Needs network:** no

### Tasks

1. Create `requirements.txt` with pinned minor versions: `pydantic`, `python-dotenv`, `chromadb`,
   `pypdf`, `tiktoken`, plus per-phase additions (`pytest` now; `streamlit` in Phase 6).
2. Create the package skeleton — empty modules with docstring-free stubs for every file listed
   in architecture.md §3.1, plus `tests/`, `documents/`, `data/` (with `data/chroma`
   gitignored), `scripts/`, `.env.example`, and a `.gitignore` covering `.env`, `data/chroma/`,
   `__pycache__/`, `.pytest_cache/`.
3. `config.py` — dataclass with every key from architecture.md §3.10, env-overridable, defaults
   as **[ASSUMPTION]** values: `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`, `TOP_K=4`,
   `LLM_TEMPERATURE=0.2`, `HISTORY_TURNS=3`, `EMBED_BATCH_SIZE=32`, `LLM_TIMEOUT_S=30`,
   `EMBEDDING_MODEL`, `LLM_MODEL`, `CHROMA_DIR`, `DOCS_DIR`, `SIMILARITY_THRESHOLD`.
4. `models.py` — `Chunk`, `ScoredChunk`, `Answer` exactly per architecture.md §3.2, including
   `latency_ms`.
5. `errors.py` — `RagChatError`, `IngestError`, `ProviderError`, `IndexMissingError`
   (architecture.md §3.3 references these names).
6. Add `if __name__ == "__main__":` to `config.py` printing the resolved config, with any
   secret value replaced by `***`.

### Acceptance criteria

- `pip install -r requirements.txt` succeeds.
- `python -m ragchat.config` prints every key, no key leaks.
- `from ragchat.models import Chunk` works; the three models round-trip.

### Verification

```bash
pip install -r requirements.txt
python -m ragchat.config
python -c "from ragchat.models import Chunk, ScoredChunk, Answer; print('models ok')"
pytest -q
```

### Cursor prompt

```
Implement Phase 0 only. Read PRD.md and architecture.md first; architecture.md is
authoritative.

Create the repository scaffold for the RAG chatbot:
- requirements.txt (pydantic, python-dotenv, chromadb, pypdf, tiktoken, pytest — pinned
  minor versions). No LangChain.
- Directory layout exactly as architecture.md §3.1: ragchat/ with config.py, models.py,
  errors.py, loaders.py, chunker.py, embedder.py, vectorstore.py, retriever.py, prompts.py,
  generator.py, pipeline.py, serve.py, ingest.py, cli.py. Plus tests/, documents/, data/,
  scripts/.
- Do NOT implement the other modules. Create them as empty files with an __init__.py.
- .gitignore: .env, data/chroma/, __pycache__/, .pytest_cache/
- .env.example listing every env var name with a placeholder value, no real keys.

config.py: a frozen dataclass holding every key in architecture.md §3.10 with the documented
default, overridable from environment via python-dotenv. Must include a human-readable
repr/print that masks any secret. Add a __main__ block that prints the resolved config.

models.py: Chunk, ScoredChunk, Answer per architecture.md §3.2. Answer must always carry a
populated `retrieved` list, including on refusal.

errors.py: the four exception classes named in architecture.md §3.3, all deriving from one
base RagChatError.

Type-hint everything. No comments restating code. Then run:
  pip install -r requirements.txt
  python -m ragchat.config
  python -c "from ragchat.models import Chunk, ScoredChunk, Answer; print('models ok')"
and paste the real output. List every file you created and any assumptions made.
```

---

## Phase 1 — Loaders and Chunker (FR1a)

**Maps to:** architecture.md §3.4, §3.3
**Depends on:** Phase 0
**Needs network:** no

### Tasks

1. `loaders.py` — `load(path) -> list[tuple[str, dict]]`. PDF via `pypdf` (one segment per
   page), `.md` (split on headings, keep heading text as the section anchor), `.txt`
   (whole file as one segment). Metadata per segment: `source_file`, and `page` or `section`.
2. `chunker.py` — token-counted split using `tiktoken`, `CHUNK_SIZE` tokens with
   `CHUNK_OVERLAP` overlap, respecting segment boundaries so a chunk never spans two pages or
   sections.
3. Stable chunk IDs: `sha1(f"{source_file}:{page_or_section}:{chunk_index}")` (architecture.md
   §3.4).
4. `ingest.py` skeleton with a `--dry-run` flag that loads + chunks and prints per-file
   segment and chunk counts, writing nothing.
5. `tests/test_chunker.py` — chunk size never exceeds `CHUNK_SIZE`; consecutive chunks overlap
   by roughly `CHUNK_OVERLAP`; no chunk crosses a page/section boundary; IDs are stable across
   two runs; metadata is always populated.
6. Add 2–3 small sample documents to `documents/` (one .md, one .txt, and a short generated
   PDF if convenient) so later phases have something real to run against.

### Acceptance criteria

- `--dry-run` prints real per-file counts.
- Chunker tests pass with no API key and no network.
- A chunk's `text` is never empty.

### Verification

```bash
python -m ragchat.ingest --dry-run
pytest tests/test_chunker.py -q
```

### Cursor prompt

```
Implement Phase 1 only (FR1a: loaders and chunker). Read architecture.md §3.1, §3.3, §3.4
first. Phase 0 is done.

- loaders.py: load(path) -> list[(text, metadata)].
  PDF (pypdf): one segment per page, metadata has source_file and page.
  Markdown: split on headings, keep heading text as the section anchor in metadata.
  Text: whole file as one segment.
  Unsupported extension -> IngestError naming the file.
  Empty/unparseable text -> IngestError naming the file. Never return empty text silently
  (architecture.md §6).
- chunker.py: token-counted with tiktoken. CHUNK_SIZE tokens, CHUNK_OVERLAP overlap.
  Chunks must NOT span a page or section boundary — split within each segment, then merge
  the results. chunk.id = sha1("source_file:page_or_section:chunk_index"). metadata always
  carries source_file, page or section, chunk_index.
- ingest.py: argparse with --dry-run (default behaviour) and later a --commit flag. Dry run
  loads and chunks, prints per-file segment and chunk counts plus total, writes nothing.
- tests/test_chunker.py: assert no chunk exceeds CHUNK_SIZE tokens; consecutive chunks
  overlap by about CHUNK_OVERLAP; no chunk crosses a page/section boundary; two runs produce
  identical ids; no empty chunk text.
- Add 2-3 small sample documents under documents/ (one .md, one .txt, one short PDF) with
  real course-like content, enough to test retrieval later. Keep them small.

Then run `python -m ragchat.ingest --dry-run` and `pytest tests/test_chunker.py -q` and paste
the real output.
```

---

## Phase 2 — Embedder, Vector Store, Ingest CLI (FR1b)

**Maps to:** architecture.md §3.4, §3.5, §5.2, §5.4
**Depends on:** Phase 1
**Needs network:** yes (embedding calls), one API key

### Tasks

1. `embedder.py` — `Embedder` protocol with `embed(texts: list[str]) -> list[list[float]]`;
   hosted implementation now, local implementation stubbed behind the same protocol
   (architecture.md §5.6). Batch internally at `EMBED_BATCH_SIZE`. Retry ×3 with backoff, then
   raise `ProviderError`.
2. `vectorstore.py` — thin wrapper: `add(chunks_with_vectors)`, `query(vec, k) -> list[ScoredChunk]`,
   `count()`, `reset()`. Cosine similarity. `reset()` before rebuild so re-ingest can never
   leave stale chunks (architecture.md §3.4, §5.2).
3. `ingest.py --commit` — the real path: sorted deterministic file order, batched embedding,
   rebuild collection, write `data/manifest.json` with embedding model, chunk size, overlap,
   chunk count, and a hash of the source files.
4. A `scripts/fingerprint.py` (or a `config.py` helper) exposing the hash so serve can verify
   it in Phase 6.
5. `tests/test_manifest.py` — manifest is written, contains all five fields, and two ingests
   of an unchanged corpus produce an identical chunk count and manifest hash.

### Acceptance criteria

- `--commit` builds the index; printed chunk count matches `manifest.json`.
- Re-running ingest produces a byte-identical manifest and identical chunk IDs (PRD M1 exit
  condition).
- A fresh `.venv` + key is the only setup needed.

### Verification

```bash
python -m ragchat.ingest --commit
python -m ragchat.ingest --commit
python -c "import json;print(json.load(open('data/manifest.json')))"
pytest tests/test_manifest.py -q
```

### Cursor prompt

```
Implement Phase 2 only (FR1b: embedder, vector store, ingest commit path). Read
architecture.md §3.4, §3.5, §5.2, §5.4 first. Phases 0-1 are done.

- embedder.py: an `Embedder` Protocol with embed(texts: list[str]) -> list[list[float]].
  One hosted implementation that batches at config.EMBED_BATCH_SIZE and retries 3x with
  exponential backoff before raising ProviderError. Add a clearly-marked local fallback class
  implementing the same protocol but raising NotImplementedError for now — do not pick a local
  model yet (architecture.md §5.6). Never log the API key.
- vectorstore.py: thin wrapper over persistent local Chroma at config.CHROMA_DIR, collection
  "course_notes". Methods: add(pairs of (Chunk, vector)), query(vector, k) -> list[ScoredChunk]
  with cosine similarity and rank, count(), reset(). Normalize on write so scores are
  comparable across runs.
- ingest.py: extend the dry-run scaffold with --commit. Iterate documents/ in sorted order for
  determinism. For each file: load -> chunk -> batched embed -> add. Call reset() first so a
  re-ingest can never leave stale chunks behind. Print per-file and total chunk counts.
- On success write data/manifest.json containing: embedding_model, chunk_size,
  chunk_overlap, chunk_count, and source_hash (sha256 over sorted "relpath:size:mtime-free
  content hash" of every document). Write it atomically.
- Expose the source-hash function from config.py or a small scripts/fingerprint.py so Phase 6
  can compare it.
- tests/test_manifest.py: manifest contains all five fields; ingesting an unchanged corpus
  twice gives an identical manifest and identical chunk count.

Do not add caching, query expansion, reranking, or hybrid search.

Then run the two --commit runs, print the manifest, and run
`pytest tests/test_manifest.py -q`. Paste the real output.
```

---

## Phase 3 — Retriever and Threshold Calibration (FR2, FR5)

**Maps to:** architecture.md §3.5, §3.7, §5.3
**Depends on:** Phase 2
**Needs network:** embeddings only, no LLM

**This is the phase that determines whether the project works. Allocate real time here.**

### Tasks

1. `retriever.py` — condition the query on the last `HISTORY_TURNS` turns, embed it once,
   return top-`TOP_K` chunks, and expose `best_score`.
2. `cli.py` — `python -m ragchat.cli retrieve "<question>"` printing rank, score, file, page,
   and an excerpt of each hit. This is the PRD M2 deliverable and the tool for tuning.
3. `scripts/calibrate_threshold.py` — run the positive set and negative set, print both score
   distributions, and recommend a threshold that maximizes the margin. Per architecture.md
   §5.3, if the distributions overlap, **stop and fix chunking** rather than nudging the
   threshold.
4. Author `scripts/demo_questions.jsonl` now, 20 lines: 15 in-corpus covering direct lookup,
   definitions, and "how does X relate to Y"; 5 out-of-corpus including at least two
   near-misses that sound plausible but are genuinely absent. Every in-corpus line carries the
   expected `source_file` (and page where known) so it can double as a test fixture.
5. `tests/test_retrieval.py` — all 15 in-corpus questions return a chunk from the expected
   file; all 5 out-of-corpus questions return `best_score < SIMILARITY_THRESHOLD`.

### Acceptance criteria

- `cli.py retrieve` returns sensible chunks for all 15 in-corpus questions.
- 5/5 out-of-corpus questions fall below the calibrated threshold.
- The threshold is a measured number, written into `.env` with a comment recording how it was
  derived.
- If separation fails, chunk size/overlap are retuned and the calibration rerun.

### Verification

```bash
python -m ragchat.cli retrieve "what is a vector embedding"
python scripts/calibrate_threshold.py
pytest tests/test_retrieval.py -q
```

### Cursor prompt

```
Implement Phase 3 only (retrieval + threshold calibration). Read architecture.md §3.5, §3.7,
§5.3 first. Phases 0-2 are done and the index is built.

- retriever.py: retrieve(question, history) conditions the query on the last
  config.HISTORY_TURNS turns (concatenated as one string, no LLM call — architecture.md §3.5),
  embeds it once, returns top config.TOP_K hits plus best_score. Empty index raises
  IndexMissingError whose message includes the command `python -m ragchat.ingest --commit`.
- cli.py: `python -m ragchat.cli retrieve "<question>" [--turns N]` prints a table of rank,
  similarity score, source file, page/section, and a ~200 char excerpt, then the best score.
  No LLM anywhere in this phase.
- scripts/demo_questions.jsonl: author 20 lines. 15 in-corpus (direct factual lookup,
  definitions, "how does X relate to Y" across two sections) and 5 out-of-corpus, at least two
  of which are near-misses that sound plausible but are NOT in the documents. In-corpus lines
  include expected_source_file (and page when known).
- scripts/calibrate_threshold.py: run all 20 questions, print each best_score labelled
  positive/negative, print both ranges, and recommend a threshold maximizing the margin. If the
  ranges overlap, print a loud warning that chunking (CHUNK_SIZE / CHUNK_OVERLAP) must be fixed
  before the threshold can be trusted, and exit non-zero. Do not auto-pick a threshold that
  misclassifies either class.
- tests/test_retrieval.py: assert each in-corpus question's top hit comes from
  expected_source_file, and each out-of-corpus question's best_score is below
  SIMILARITY_THRESHOLD.

Then run a couple of `cli.py retrieve` queries, run calibrate_threshold.py, and paste the real
output including both score ranges. Report the recommended threshold and whether the classes
separate. If they do not separate, say so plainly instead of lowering the threshold.
```

---

## Phase 4 — Prompts and Generator (FR3, FR4)

**Maps to:** architecture.md §3.6, §5.5
**Depends on:** Phase 3
**Needs network:** yes, LLM key

### Tasks

1. `prompts.py` — system prompt establishing the role and the single hard rule (answer only
   from the provided context, say when it is insufficient), the grounded user template with
   numbered `<<<SOURCE n>>>` / `<<<END n>>>` delimiters, the fixed refusal string naming the
   corpus, and `HISTORY_TURNS` slicing.
2. `generator.py` — LLM client behind a protocol, `temperature = LLM_TEMPERATURE`, non-streaming,
   `LLM_TIMEOUT_S` timeout, retry ×3 with backoff. On exhaustion return a friendly string, never
   a partial answer.
3. **Citation assembly from `retrieved`, not from parsing the model's markers** (architecture.md
   §3.6). A citation must never reference a chunk that was not provided.
4. `tests/test_generation.py` — prompt assembly is unit-testable with a fake LLM: context
   contains all k chunks; the refusal path makes **zero** LLM calls; a timeout produces the
   friendly message, not an exception.

### Acceptance criteria

- Golden questions (3–5 from the demo set) answered with a citation pointing at the right
  file/page.
- The refusal path is verified to skip the LLM entirely.
- A deliberate out-of-context question produces a refusal, not an answer.

### Verification

```bash
pytest tests/test_generation.py -q
python -m ragchat.cli ask "what is a vector embedding"
python -m ragchat.cli ask "what is the capital of France"
```

### Cursor prompt

```
Implement Phase 4 only (prompting + generation). Read architecture.md §3.6, §5.5 first.
Phases 0-3 are done; retrieval and the threshold are calibrated and tested.

- prompts.py:
  - SYSTEM_PROMPT: assistant role for a course-notes Q&A bot; the one hard rule is to answer
    ONLY from the supplied context and to say plainly when the context is insufficient. Keep
    it short — it is a demo, not a research paper.
  - Numbered source delimiters `<<<SOURCE 1>>>` ... `<<<END 1>>>` (per architecture.md §3.6 —
    NOT XML-ish tags, and do not explain why in a comment; the reasoning is in the doc).
  - Each source block includes its file and page/section inline so the model can reference it.
  - REFUSAL: a fixed string that names the corpus, e.g. it can only answer from the course
    notes it was given. Returned without any LLM call.
  - build_prompt(question, history, chunks) that slices to config.HISTORY_TURNS and returns the
    ordered message list.
- generator.py: an LLM client behind a Protocol. temperature = config.LLM_TEMPERATURE,
  non-streaming, timeout config.LLM_TIMEOUT_S, retry 3x with backoff. On final failure return a
  friendly user-facing string — never a partial answer, never raise to the UI.
  generate(question, history, chunks) -> (text, citations) where citations are derived from the
  chunks passed in, in order, de-duplicated. Do NOT parse `[1]` markers out of the model's prose
  to build the citation list.
- cli.py: add `ask "<question>"` printing the answer, its citations (file + page), and the
  top score.
- tests/test_generation.py with a fake LLM: assert the assembled prompt contains all k chunk
  texts; assert the refusal path makes zero LLM calls; assert a simulated timeout yields the
  friendly message rather than an exception.

Then run `pytest tests/test_generation.py -q` and paste the real output.
```

---

## Phase 5 — Pipeline Orchestrator

**Maps to:** architecture.md §2 (application), §3.3, §4, §6
**Depends on:** Phase 4
**Needs network:** yes
**Status:** orchestrator done and verified. The generation half of the 20-question run is
blocked on an `LLM_API_KEY`; `run-demo --check` verifies retrieval and the gate without one.

### Tasks

1. `pipeline.py` — `ask(question, history) -> Answer` as the single orchestrator: retriever →
   threshold gate → generator (or refusal). Populate `latency_ms` for embed/retrieve/generate.
2. Error boundary: every internal failure becomes an `Answer` with a friendly `text`; nothing
   propagates to the UI as a trace (architecture.md §6).
3. Startup checks: missing key → named message naming the env var; empty index →
   "run `python -m ragchat.ingest --commit`"; model/manifest mismatch → warn loudly but
   continue.
4. Run the full 20-question demo script end to end and record the result per question.
   `run-demo` does this; `run-demo --check` stops after the gate so the script can be
   rehearsed with no model configured.

### Acceptance criteria

- All 20 demo questions produce an `Answer`; no exceptions escape.
- 5 out-of-corpus questions are refused with no LLM call.
- `latency_ms` is populated and total is under ~5s for typical questions (NFR).

### Verification

```bash
python -m ragchat.cli run-demo --check
python -m ragchat.cli run-demo
pytest -q
```

### Observed (2026-09-29, no LLM configured)

`run-demo --check` against the 9-chunk sample corpus:

```
questions   : 20
answered    : 15
refused     : 5  (expected 5)
gate correct: 5/5 out-of-corpus questions refused
in-corpus wrongly refused: 0
slowest turn: 2165ms  What is retrieval-augmented generation?
```

In-corpus scores span 0.748-0.835; the five out-of-corpus questions score 0.442-0.600,
so the 0.6739 gate separates the two populations with a 0.148 margin. The slowest turn is
the first one, which pays the one-off `fastembed` model load (~2s); steady-state turns are
well under the 5s NFR.

### Cursor prompt

```
Implement Phase 5 only (the orchestrator). Read architecture.md §2, §3.3, §4, §6 first.
Phases 0-4 are done.

- pipeline.py: ask(question, history) -> Answer is the ONLY entry point the UI will use.
  Flow: condition + embed the question, cosine top-k, threshold gate, then either generate or
  return the fixed refusal. Populate Answer.retrieved in BOTH cases — including refusal, where
  the weak hits are the most useful thing to show. Populate latency_ms for embed, retrieve, and
  generate. Set refused=True on the refusal path.
  Catch internal failures here and convert them into an Answer with a friendly message. No
  exception should ever escape to a caller as a trace.
- Startup checks, surfaced as human-readable results rather than crashes: missing API key ->
  name the env var; empty/missing index -> tell the user to run `python -m ragchat.ingest
  --commit`; embedding model or source_hash mismatch against data/manifest.json -> warn
  loudly but continue.
- cli.py: add `run-demo` that reads scripts/demo_questions.jsonl and prints a per-question
  table: question, refused yes/no, top score, top source file + page, latency. Finish with a
  summary: how many answered, how many refused, and the slowest turn.

Then run `python -m ragchat.cli run-demo`, `pytest -q`, and paste the real output including the
summary table.
```

---

## Phase 6 — Streamlit UI (FR7)

**Maps to:** architecture.md §3.8, §3.9, §3.1
**Depends on:** Phase 5
**Needs network:** yes

### Tasks

1. `serve.py` — chat thread, input box, send, loading state, reset button.
2. Assistant turn renders: answer text → citations row (`file · p.N`) → collapsed
   "Sources (k) · top score" expander with raw chunk text, score, file/page.
3. Sidebar: corpus description, resolved configuration (model names only, no keys), chunk
   count, reset.
4. Footer: one line naming what the bot is grounded in.
5. `st.session_state` holds the message list; no external store; reset clears it only.
6. Empty/garbled input rejected before hitting the pipeline.
7. Startup warnings from Phase 5 rendered as a non-blocking banner.

### Acceptance criteria

- One command starts the app; a question returns an answer with a citations row.
- The sources expander shows the raw chunks that produced the answer.
- Reset clears the thread without rebuilding the index.
- No stack trace is reachable from the UI.

### Verification

```bash
streamlit run ragchat/serve.py
```

Then manually: ask a factual question, expand sources, ask an out-of-corpus question, ask a
follow-up pronoun question, hit reset. Screenshot each for the demo.

### Cursor prompt

```
Implement Phase 6 only (the Streamlit UI). Read architecture.md §3.8, §3.9 first. Phases 0-5
are done and `python -m ragchat.cli run-demo` is clean.

- serve.py: calls ONLY pipeline.ask() — do not import retriever, generator, or embedder.
- Layout in the order the demo audience needs it:
  1. Main: chat thread. Each assistant turn shows the answer, then a citations row
     ("file · p.N"), then a COLLAPSED expander labelled like "Sources (4) · top score 0.81"
     containing each chunk's raw text, score, file, and page.
  2. Sidebar: what the corpus is, the resolved config (model NAMES only — never a key), the
     corpus chunk count, and a Reset button that clears the message list and nothing else.
  3. Footer: one line naming what the bot is grounded in.
- Conversation state lives in st.session_state. Guard against empty/whitespace input before
  calling the pipeline. Show a loading state during the call.
- Render Phase 5's startup warnings as a non-blocking banner.
- No stack trace may be reachable from the UI. Add a top-level error handler that shows a short
  friendly message.

Then run `streamlit run ragchat/serve.py`, confirm it boots, and paste the startup output. Do
not add streaming, auth, or upload.
```

---

## Phase 7 — Demo Hardening (FR8, NFRs)

**Maps to:** architecture.md §6, §5.4, §5.6, §7
**Depends on:** Phase 6
**Needs network:** yes, plus one deliberately-offline rehearsal

### Tasks

1. `README.md` — install, `.env` setup, ingest, run, the demo script, an honest note that the
   course material is sent to the model provider, and a troubleshooting table (index missing,
   key missing, everything refused, everything answered).
2. Cold-setup rehearsal on a clean machine or fresh venv, timed (NFR: under 15 minutes).
3. Rehearse the refusal as a *feature* — the 5 out-of-corpus questions are the strongest part
   of the demo, not an embarrassment (PRD G2).
4. Local model fallback behind the `Embedder` protocol (architecture.md §5.6) so demo-day
   network failure has a degraded path, plus a recorded transcript as the last resort
   (PRD §10).
5. Add a pytest end-to-end smoke test: start the app's pipeline, ask one question, assert a
   citation renders.
6. Record the 20-question transcript into the README as a fallback artifact.
7. Review every file for secrets, and confirm `.env` is gitignored.

### Acceptance criteria

- Cold setup to running app under 15 minutes.
- The full 20-question script runs without a crash.
- A network failure produces a degraded response, not a stack trace.
- No secret appears anywhere in the repo or in the UI.

### Verification

```bash
# in a fresh venv
python -m venv .venv && . .venv/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in keys
python -m ragchat.ingest --commit
streamlit run ragchat/serve.py
# then: time the whole thing, run the 20-question script, unplug the network and try one question
pytest -q
```

### Cursor prompt

```
Implement Phase 7 only (demo hardening). Read architecture.md §6, §5.4, §5.6, §7 first. All
phases 0-6 are done.

- README.md: prerequisites, install, .env setup, ingest command, run command, the 20-question
  demo script with an explanation of why the out-of-corpus questions are a feature, a plain
  statement that course material is sent to the third-party model provider, and a
  troubleshooting table (index missing / key missing / every question refused / every question
  answered / slow).
- Finish the local Embedder fallback behind the same protocol (architecture.md §5.6) and a local
  LLM fallback behind the generator protocol, selected by env var. They must degrade quality
  gracefully, not crash.
- tests/test_smoke.py: end-to-end through pipeline.ask — ask one question, assert text is
  non-empty, assert at least one citation is returned, assert latency_ms is populated.
- Add a pytest that asserts every secret-looking env var is absent from the tracked files and
  that .env is in .gitignore.

Do NOT write credentials anywhere. Do not commit data/chroma/.

Then run `pytest -q` and paste the real output. List every file added or changed.
```

---

## Appendix A — Phase-to-Requirement Traceability

| PRD requirement | Phase | Verified by |
|-----------------|-------|-------------|
| FR1 Corpus ingestion | 1, 2 | `test_chunker.py`, `test_manifest.py` |
| FR2 Retrieval (top-k, threshold) | 3 | `test_retrieval.py`, `cli.py retrieve` |
| FR3 Answer generation | 4 | `test_generation.py` |
| FR4 Citation display | 4, 6 | `test_generation.py`, UI check |
| FR5 Out-of-corpus handling | 3, 4 | refusal path makes zero LLM calls |
| FR6 Conversation context | 3 (retrieval), 4 (prompt), 6 (state) | `test_retrieval.py`, UI follow-up |
| FR7 Web UI | 6 | manual UI check |
| FR8 Demo mode | 7 | 20-question script |
| NFR latency | 5 | `cli.py run-demo` summary |
| NFR portability | 7 | cold-setup rehearsal |
| NFR robustness | 5, 7 | `test_smoke.py`, error boundary |
| NFR privacy | 7 | secret scan, README disclosure |
| NFR explainability | 6 | sidebar config panel |
| Goal G1 grounded answers | 4, 6 | citations visible in UI |
| Goal G2 honest refusal | 3, 4 | 5/5 out-of-corpus refused |
| Goal G3 retrieval visible | 6 | sources expander |
| Goal G4 demo reliability | 7 | 20-question clean run |

## Appendix B — Time Budget

Rough, for a class team of 2–3. Adjust once open questions are answered.

| Phase | Effort | Parallelizable? |
|-------|--------|-----------------|
| 0 Scaffold | 1–2 h | No — everything depends on it |
| 1 Loaders + chunker | 2–3 h | Partly; tests can be written alongside |
| 2 Embedder + store + ingest | 3–4 h | No |
| 3 Retriever + calibration | **6–10 h** | Chunk-size tuning can be split across people |
| 4 Prompts + generator | 3–5 h | Prompt iteration can be split across people |
| 5 Pipeline | 2–3 h | No |
| 6 UI | 4–6 h | Yes, with Phase 4/5 |
| 7 Hardening | 3–5 h | No — needs everything |
| **Total** | **~24–38 h** | |

Phase 3 is the long pole and the one most likely to be underestimated. If time is short, cut
Phase 6 polish (Phase 7's transcript fallback covers a rough UI) before cutting Phase 3.

## Appendix C — Definition of Done (whole project)

- [ ] All tests green, including the out-of-corpus threshold test
- [ ] 20-question demo script runs clean, refusals included
- [ ] Cold setup to running app under 15 minutes
- [ ] No secrets in the repo, no stack trace reachable from the UI
- [ ] Sidebar shows the resolved config; README explains the demo
- [ ] Demo runs with no network and degrades gracefully
- [ ] Every **[ASSUMPTION]** in PRD.md §13 resolved to a real decision, and this file's
      defaults updated to match
- [ ] Open questions in PRD.md §12 answered, particularly corpus size, model budget, and
      whether LangChain is permitted
