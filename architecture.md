# Architecture: Course RAG Chatbot (Class Demo)

**Companion to:** `PRD.md`
**Status:** Draft — for team review
**Scope:** Everything here implements FR1–FR8 and the NFRs in `PRD.md` §6–§7. Nothing beyond
that scope is designed. Choices inherited as **[ASSUMPTION]** from the PRD stay marked as such.

---

## 1. Design Goals

Ordered by priority. When two goals conflict, the higher one wins.

1. **Retrieval correctness before generation quality.** If the chunks are wrong, no prompt
   saves it. The pipeline is built so retrieval is testable with no LLM in the loop (PRD M2).
2. **Every answer traceable to a chunk.** No chunk, no answer. The UI can always reconstruct
   *why* a given answer was produced (G1, G3, US2, US3).
3. **Deterministic failure over probabilistic success.** When retrieval is weak we refuse; we do
   not ask the LLM to guess (G2, FR5).
4. **Minimal moving parts on demo day.** No servers to start, no database to provision, one
   command to launch (G4, NFR setup/latency).
5. **No hidden state.** Model, chunker, k, and threshold are config, printed at startup (NFR
   explainability).

## 2. System Context

```
┌───────────────────────┐
│  Browser (Streamlit)  │
└───────────┬───────────┘
            │ HTTP (localhost)
┌───────────▼───────────────────────────────────────────┐
│  Application  (ragchat/serve.py)                      │
│  - conversation state per session                     │
│  - orchestration of retrieve → gate → generate        │
│  - error → friendly message mapping                   │
└──┬──────────────┬─────────────────────┬───────────────┘
   │              │                     │
┌──▼───────────┐ ┌▼────────────────┐  ┌▼──────────────────┐
│ VectorStore  │ │ Embedder        │  │ LLM client        │
│ (Chroma,     │ │ (1 interface,   │  │ (chat, temp~0.2)  │
│  local dir)  │ │  2 backends)    │  │                   │
└──▲───────────┘ └▲────────────────┘  └▲──────────────────┘
   │              │                     │
   │        ┌─────┴─────────────────────┴─────┐
┌──┴────────┴─────────────────────────────────┴──┐
│  Offline ingestion (ragchat/ingest.py)         │
│  documents/ → loaders → chunker → embedder →   │
│              → vector store + manifest.json     │
└───────────────────┬────────────────────────────┘
                    │ one-time, pre-demo
┌───────────────────▼────────────────────────────┐
│  documents/  (PDF, Markdown, .txt)             │
└────────────────────────────────────────────────┘
```

Two things never run in the same process: ingestion writes the index, serving only reads it.
This is what makes PRD M1/M2 independent of the app.

## 3. Component Design

### 3.1 Module layout

```
ragchat/
  config.py          # env + defaults; single source of truth for tunables
  models.py          # Chunk, ScoredChunk, Answer, Turn  (pydantic)
  loaders.py         # file → (text, metadata) with page/section anchors
  chunker.py         # text → Chunk  (token-counted, overlapping)
  embedder.py        # Embedder protocol + hosted & local implementations
  vectorstore.py     # thin wrapper: add / query / stats
  retriever.py       # embed query → top-k → threshold decision
  prompts.py         # system prompt, grounded user template, refusal string
  generator.py       # LLM call, citation assembly, retries
  pipeline.py        # ask(question, history) → Answer   ← the orchestrator
  serve.py           # Streamlit UI
  ingest.py          # CLI entry point for ingestion
  cli.py             # CLI for testing retrieval/generation headlessly
data/
  chroma/            # persisted index (gitignored)
  manifest.json      # ingest fingerprint (see §5.4)
documents/           # the corpus (committed)
tests/
scripts/
  demo_questions.jsonl   # PRD FR8
```

`pipeline.ask()` is the only entry point the UI needs. Everything else is importable and
independently runnable — that is what makes M2 possible.

### 3.2 Data contracts

```python
# models.py
class Chunk(BaseModel):
    id: str                 # sha1(source_file:page:chunk_index) — stable across re-ingests
    text: str
    metadata: dict          # source_file, page|section, chunk_index

class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float            # similarity, higher is better
    rank: int

class Answer(BaseModel):
    text: str
    citations: list[Chunk]          # ordered, de-duplicated
    retrieved: list[ScoredChunk]    # always populated, even on refusal
    refused: bool
    latency_ms: dict[str, int]      # embed / retrieve / generate
```

`Answer` deliberately keeps `retrieved` populated on refusal. The UI shows the weak matches
that triggered the refusal — that is the most persuasive part of the G2 demo.

### 3.3 Component contracts

| Component | Interface | Failure behavior |
|-----------|-----------|------------------|
| `loaders` | `load(path) -> list[(text, meta)]` | Raise `IngestError` naming the file; ingestion stops (a half-built index is worse than none) |
| `chunker` | `chunk(text, meta) -> list[Chunk]` | Raise on tokenization failure; never silently drop text |
| `embedder` | `embed(texts: list[str]) -> list[list[float]]` | Retry ×3 with backoff, then raise `ProviderError` |
| `vectorstore` | `add(chunks)`, `query(vec, k) -> list[ScoredChunk]`, `count()` | Raise; surfaced as a startup banner, not a crash loop |
| `retriever` | `retrieve(question, history, k) -> RetrieveResult` | Empty index → raise `IndexMissingError` with the ingest command in the message |
| `generator` | `generate(question, history, chunks) -> (text, citations)` | Retry ×3; on exhaustion return a message, never a partial answer |
| `pipeline` | `ask(question, history) -> Answer` | Catches everything above; maps to `Answer` with a friendly `text` |

### 3.4 Ingestion pipeline (FR1)

```
for each file in documents/ (sorted, for determinism):
    segments = loaders.load(file)                  # page/section anchored
    for segment:
        for chunk in chunker.chunk(segment.text, meta):   # 500 tokens, 50 overlap
            embedder.embed([chunk.text])           # batched, ~32 per call
            vectorstore.add([chunk])
write data/manifest.json
```

Design notes:

- **Deterministic ordering and stable chunk IDs** (sha1 of source + position) mean re-running
  ingestion over an unchanged corpus produces an identical index, so a demo-day re-ingest
  cannot silently change answers.
- **Batched embedding calls** — one call per 32 chunks, not per chunk, for cost and speed.
- **Page/section anchors survive chunking.** This is what makes FR4's citations real rather
  than decorative.
- **Overwrite semantics** on re-ingest: the collection is rebuilt, not appended to. A stale
  index is the most likely silent-demo-killer.
- Chunking is token-counted, not character-counted, so chunk sizes are stable across models.

### 3.5 Retrieval (FR2)

```
query_vec = embedder.embed([conditioned_question])[0]
hits      = vectorstore.query(query_vec, k=4)          # cosine similarity
best      = hits[0].score if hits else 0.0
if best < THRESHOLD:  -> refused, no LLM call
else:                 -> hits to generator
```

- **Query conditioning (FR6):** the retrieval query is `current_question` prepended with recent
  turns of history as a single string. This is enough to resolve "that" /
  "it" without a separate query-rewrite LLM call — one fewer network hop on the latency budget.
  **[ASSUMPTION]** last-10-turns conditioning, per PRD FR6. See "Two history windows" below.
- **Two history windows.** Retrieval conditions on `RETRIEVAL_HISTORY_TURNS` (10) while the
  generator prompt is capped at `HISTORY_TURNS` (3). They differ on purpose: the embedding
  needs the older referent to resolve a follow-up ("the metric from before the split"), whereas
  the model only needs recent turns to read four source blocks. Widening the retrieval window
  raises the risk noted in `ragchat/retriever.py` of a stale topic diluting the query, which is
  why history is prepended *only* when the question actually refers back.
- **One embedding per question.** Retrieval and the LLM are strictly sequential; no parallel
  fan-out.
- **Similarity metric:** cosine. Chroma's default; stored normalized so scores are comparable
  across runs.
- **Threshold is calibrated, not guessed** — see §5.3. This is the single most important
  tunable in the system.
- No reranking, no hybrid keyword search, no query expansion (PRD FR2, v1 non-goal).

### 3.6 Generation (FR3) and citations (FR4)

Prompt assembly, in order:

1. **System prompt** — role, the one hard rule (answer only from `<context>`), and an explicit
   instruction to say when context is insufficient. Chunk excerpts are delimited by
   `<<<SOURCE n>>>` / `<<<END n>>>` markers rather than XML-ish tags, because delimiter
   injection from document text is a real risk and numbered fences are easier to cite.
2. **Context block** — top-k chunks with their `[1]`, `[2]`… numbers, each with
   `source_file` and `page`/`section` inline.
3. **Recent history** — last *k* assistant/user pairs.
4. **Current question.**

Then:

- Chat call, `temperature ≈ 0.2` **[ASSUMPTION]**, non-streaming in v1 (keeps the citation
  assembly and error handling simple).
- **Citation assembly:** chunks are returned to the UI regardless of what the model wrote. We
  do not parse `[1]` markers out of the prose to build the list — the LLM's own marker
  handling is untrustworthy, and a half-parsed citation is worse than none. `citations` is
  derived from `retrieved`, so a citation can never reference a chunk that was not actually
  provided.
- Timeout on the LLM call, then retry with backoff; after exhaustion the user gets
  "the model didn't respond, try again" rather than a stack trace.

### 3.7 Out-of-corpus gate (FR5)

```
score >= THRESHOLD  -> generate
score <  THRESHOLD  -> Answer(text=REFUSAL, refused=True, retrieved=hits)
```

The refusal is a fixed string that names the corpus. It is returned **without calling the
LLM**, which makes it deterministic, instant, and impossible to talk past. The retrieved
(weak) chunks still travel with the response for display.

Rationale for a hard gate rather than letting the LLM decide: the LLM's "I don't know" is
sampled, so it will sometimes confidently answer a question the corpus doesn't cover. A score
threshold is a property of the retriever, and therefore testable and tunable.

### 3.8 Session and conversation state (FR6, FR7)

- Streamlit's own `st.session_state` holds the message list; no external session store.
- Reset button clears it and nothing else (no index rebuild).
- History passed to the *prompt* is capped at the last `HISTORY_TURNS` (3) turns, at prompt-build
  time, so long conversations cannot grow the prompt unbounded. The retriever's window is
  separate and wider; see A3.5.
- The message list stored for display and the list passed to the model are the same object,
  sliced — no second source of truth for "what has been said".

### 3.9 UI (FR7)

Three regions, in the order the demo audience needs them:

1. **Chat thread** — messages, loading state, and for each assistant turn: answer text, a
   citations row (`file · p.N`), and a collapsed expander "Sources (k) · top score 0.81"
   containing the raw chunk text, score, and file/page.
2. **Sidebar** — corpus description/footer, the resolved configuration (models, chunk size, k,
   threshold, corpus chunk count), and Reset.
3. **Footer** — one line naming what the bot is grounded in, so the audience always knows the
   scope of the corpus.

The sources panel is not a nice-to-have. It is the G3 / US3 / US6 evidence, and it is the panel
the audience will look at.

### 3.10 Configuration

All tunables in `config.py`, overridable by env, printed at startup:

| Key | Default **[ASSUMPTION]** | Role |
|-----|------------------------|------|
| `EMBEDDING_MODEL` | hosted embedding model | Must match what built the index |
| `LLM_MODEL` | hosted chat model | |
| `LLM_TEMPERATURE` | `0.2` | FR3 |
| `CHUNK_SIZE` | `500` | FR1, in tokens |
| `CHUNK_OVERLAP` | `50` | FR1 |
| `TOP_K` | `4` | FR2 |
| `SIMILARITY_THRESHOLD` | calibrated, §5.3 | FR2 / FR5 |
| `HISTORY_TURNS` | `3` | FR6, prompt window |
| `RETRIEVAL_HISTORY_TURNS` | `10` | FR6, retrieval window |
| `EMBED_BATCH_SIZE` | `32` | Ingest throughput |
| `LLM_TIMEOUT_S` | `30` | NFR robustness |
| `CHROMA_DIR` | `data/chroma` | |

`.env` holds keys, is gitignored, and a missing key fails at startup with a named message
rather than mid-question. Ingest and serve refuse to run if the *pair* of
`EMBEDDING_MODEL` + corpus fingerprint in `manifest.json` doesn't match the running config —
this catches the "re-ingested with a different model" class of silent nonsense.

## 4. Request Flow

Happy path:

```
1. streamlit   user submits question
2. pipeline    ask(question, history)
3. retriever   condition question on last 10 turns
4. embedder    embed conditioned question          [network]
5. vectorstore cosine search, top-4                [local, <100ms]
6. gate        best_score >= THRESHOLD?
7a. generator  build prompt → chat completion      [network, dominant cost]
7b. generator  skip; return refusal
8. pipeline    assemble Answer (text, citations, retrieved, latencies)
9. streamlit   render answer + citations row + sources expander
```

Refusal path diverges at step 6 and never reaches step 7a — visibly faster, which is worth
pointing out during the demo.

## 5. Key Decisions

### 5.1 Hand-rolled pipeline, not a framework

**Decision:** ~300 lines of explicit orchestration over LangChain/LlamaIndex.

The graph above has five steps. A framework would add an abstraction layer whose internals we
cannot show on a projector, plus a dependency that can change under us before the demo. We
need every line of the prompt and the threshold visible to explain the system. The cost is
that we own chunking and prompting ourselves — acceptable at this size.

**Revisit if:** PRD Open Question 3 answers "LangChain is fine", or the corpus grows past
what we can chunk well ourselves.

### 5.2 Chroma, local, rebuilt on ingest

**Decision:** persistent local Chroma collection, wiped and rebuilt on every ingest.

A file-based store that needs no server, no container, and no schema migrations satisfies the
"runs from a single command" requirement (G4). The rebuild-on-ingest rule trades ingestion
time — seconds at corpus scale — for the guarantee that we never serve an index that
disagrees with the documents.

**Rejected:** hosted vector DB (another credential and a failure mode on demo day); FAISS
(we'd be writing persistence and ID management ourselves for no benefit here).

### 5.3 Threshold calibration

**Decision:** treat `SIMILARITY_THRESHOLD` as a measured value, not a default.

Procedure, run once as a script (`scripts/calibrate_threshold.py`) before M4:

1. Take the 15 in-corpus demo questions as positives.
2. Take the 5 out-of-corpus questions plus a set of near-miss questions (plausible-sounding,
   genuinely absent) as negatives.
3. Record `best_score` for every question.
4. Pick the threshold that separates them, maximizing margin, and confirm no in-corpus
   question falls below it.

If the two score populations overlap, retrieval quality is the problem — go back to chunk size
and overlap, not to the threshold. Tuning the threshold to paper over bad retrieval is the trap
this step exists to avoid.

### 5.4 Corpus fingerprint

`manifest.json` records embedding model, chunk size, overlap, chunk count, and a hash of the
source files. Serve checks it at startup and warns loudly on mismatch. Cheap insurance against
the most confusing possible demo failure: answers that changed and nobody knows why.

### 5.5 Two-sided prompt defense

Grounding is enforced twice, on purpose: the system prompt forbids outside knowledge, *and* the
threshold gate means insufficient context usually never reaches the LLM. The prompt is the
backstop, not the primary control. Documented this way so the demo can be honest about the
limitation: prompt-level grounding is best-effort, and the threshold is the real mechanism.

### 5.6 Local model fallback

Both `Embedder` and the LLM client sit behind protocols with a hosted implementation and a
local one. A local fallback is the mitigation for the highest-likelihood risk in PRD §10
(demo-day network/API failure). It is a fallback, not the default: retrieval quality drops, and
we'd rather rehearse the good path.

## 6. Error Handling

| Failure | Detection | Behavior | User sees |
|---------|-----------|----------|------------|
| Missing API key | Startup | Fail fast | Named message + which env var to set |
| Index missing/empty | Startup | Fail fast | "Run `python -m ragchat.ingest`" |
| Model/manifest mismatch | Startup | Warn, continue | Yellow sidebar warning |
| Provider error or rate limit | Exception | Retry ×3, backoff, then give up that turn | "Model unavailable, try again" — no stack trace |
| LLM timeout | `LLM_TIMEOUT_S` | Retry, then abandon | "The model took too long" |
| No chunks above threshold | `best_score` | Refuse, no LLM call | "I can only answer from the course notes I was given" + weak sources shown |
| Empty/garbled input | UI guard | Reject before pipeline | Input hint, no API call |
| Chunk text empty after parsing | Ingest | Raise, abort ingest | Names the file — a broken index is never published |

No exception ever reaches the Streamlit frontend as a trace. `pipeline.ask()` is the single
boundary that converts internal failure into a user-facing string.

## 7. Security and Privacy

- **No secrets in the repo.** Keys in `.env`, gitignored, loaded via `python-dotenv`. Never
  echoed to logs or printed in the UI's config panel (the panel shows model *names* only).
- **The corpus leaves the machine** as part of embedding and prompt requests. Documented
  plainly in the README — it is a real consideration with course material, not a formality.
- **No telemetry, no analytics, no outbound calls** other than to the configured model
  provider. Nothing is written to disk during a question beyond Streamlit's own state.
- **Prompt-injection exposure is known and bounded.** Ingested course documents are
  untrusted text. Our exposure is answer quality, not system access: the LLM has no tools, no
  file access, and no ability to act, so a malicious chunk can at worst produce a bad answer
  or a misleading citation. The delimiters in §3.6 narrow this; the model has no capabilities
  worth stealing. This is stated explicitly because "RAG is vulnerable to prompt injection" is
  the obvious question from a technical audience, and "we scoped it deliberately" beats
  dodging.
- **No auth is intentional** (PRD non-goal): localhost-only, single-user demo. Binding to
  anything other than `localhost` would change that, so it is a deliberate refusal.

## 8. Trade-offs Accepted

| Trade-off | Accepted cost | Mitigated by |
|-----------|--------------|--------------|
| Hand-rolled over framework | We own chunking/prompt quality bugs | Small surface, fully visible, testable at each step |
| Similarity threshold is coarse | Near-miss questions may be wrongly refused or admitted | Calibrated on the demo set (§5.3); refusal is the safe direction to err |
| Last-10-turns retrieval conditioning only | Very distant follow-ups lose referents | Demo script uses adjacent follow-ups; prompt still capped at 3 turns |
| Non-streaming generation | ~1–2s of blank screen before the answer | Loading state; under the 5s NFR |
| Refusal skips the LLM entirely | Refusals can't be phrased in context | Intentional — deterministic refusal is more convincing than a sampled one |
| Corpus sent to a third-party provider | Course material leaves the machine | Documented in README; local fallback exists |
| No runtime document upload | Corpus is frozen at ingest | PRD non-goal |

## 9. Testing Strategy

Testing follows the pipeline, and the first tier needs no API key.

| Tier | Scope | Needs network? | Purpose |
|------|-------|----------------|---------|
| Unit | `chunker` respects size/overlap and metadata; stable chunk IDs; loaders anchor pages | No | Catches ingestion bugs at the source |
| Retrieval | Run the 15 in-corpus + 5 out-of-corpus questions; assert a chunk is returned and that scores separate | Embeddings only | M2/M4 gate — proves the core assumption |
| Threshold | Assert every out-of-corpus question falls below `SIMILARITY_THRESHOLD` | Embeddings only | Makes FR5 a regression test, not a vibe |
| Pipeline | Golden Q&A pairs: question → expected citation file/page | Yes | Guards against regressions from prompt changes |
| Smoke | Start the app, ask one question, assert a citation renders | Yes | The actual demo path, exercised once before presenting |

The retrieval and threshold tiers are the ones that matter — they are where real failures live,
and both run without the LLM, so they are cheap to iterate on.

## 10. Milestone Mapping

| PRD milestone | Architecture work | Exit condition |
|---------------|-------------------|----------------|
| M1 Corpus + ingestion | §2 ingestion path, §3.1 loaders/chunker, §3.4, §5.4 fingerprint | Re-running ingest yields an identical index; counts in manifest match |
| M2 Retrieval (no LLM) | §3.1 `retriever`, §3.5, `cli.py` | All 15 in-corpus questions return a sensible chunk; CLI prints scores |
| M3 Generation + citations | §3.6, §3.2 `Answer` | Golden questions answered with correct file/page citations |
| M4 Threshold | §5.3 calibration, §3.7 | All 5 out-of-corpus questions refused; the tier-3 test suite is green |
| M5 Streamlit UI | §3.8, §3.9, `serve.py` | Chat, citations row, sources expander, reset all working |
| M6 Demo hardening | §6, §5.6, `scripts/demo_questions.jsonl` | Full 20-question script runs clean; offline path rehearsed; setup <15 min |

## 11. Known Limitations

Stated plainly so the demo is honest rather than caught out:

- Retrieval is single-shot cosine with no reranking; a question spanning two sections gets
  mixed or partial context.
- The threshold is a single global number, so it is simultaneously tuned for one corpus and one
  question distribution. A new corpus means recalibration.
- Grounding is best-effort at the prompt level (§5.5). A sufficiently adversarial chunk could
  still mislead.
- Citations are the retrieved chunks, not spans the model quoted — so a citation can point to
  a chunk that supports the answer only partially.
- Follow-ups resolve over ~10 turns of retrieval context (3 for the model), not arbitrary
  conversation length.
- No support for images, tables-of-content cross-references, or documents whose meaning lives
  in a figure or diagram — a real limitation for some STEM material.
- Ingestion is all-or-nothing; a single bad file aborts the build.

## 12. Open Technical Questions

Carried from PRD §12, with the architectural consequence of each:

1. **Corpus size and format** — drives batch sizing and whether a single ingest pass is
   comfortable. If it's large, ingestion needs progress output and resumability.
2. **Model and budget** — the local fallback in §5.6 depends on this answer. Without a hosted
   key, the fallback becomes the default and §5.3 calibration needs to be redone against local
   score distributions.
3. **LangChain acceptable?** — §5.1 assumes no. If yes, §3.5's conditioning and §3.6's prompt
   assembly get replaced, but the threshold gate (§3.7) and the fingerprint (§5.4) should
   survive regardless; they are our own logic, not framework features.
4. **Demo-day connectivity** — decides whether the offline path is a real rehearsal item or
   a footnote.
5. **Deadline** — §11's limitations are the natural backlog if there is slack.
6. **Shared vs. per-student index** — a shared prebuilt index means one ingestion whose
   `data/chroma` we probably should *not* commit (binary, model-specific); per-student means
   ingestion must be fast and well-documented. The architecture supports both; the README
   instructions differ.
7. **Grading rubric** — if retrieval quality is the graded axis, §5.3 and the tier-2/3 tests
   deserve more time than the UI does, and the source order flips.
