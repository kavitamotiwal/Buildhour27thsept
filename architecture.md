# Architecture: HDFC Mutual Fund FAQ RAG Chatbot (Class Demo)

Facts-only assistant over 5 official HDFC mutual-fund scheme pages. Every factual answer is
≤3 sentences, carries exactly one source link, and ends with a "Last updated from sources"
footer. Advice-seeking and PII-bearing questions are refused before any retrieval. Spec:

- Product/acceptance: `PRD.md` (FR1–FR8, section 13 success criteria)
- Build plan: `implementation.md`
- Code: `ragchat/`

## 1. Design Goals

1. **Facts-only, cited.** Answers come only from the indexed scheme pages, and the source URL
   is visible in every answer (FR1, FR2).
2. **Short, dated.** ≤3 sentences plus a deterministic "Last updated from sources: <date>"
   footer (FR3, FR4).
3. **Fail closed.** Anything out-of-corpus, advice-seeking, or PII-bearing is refused safely;
   there is no "confident guess" path (FR5, FR6, FR7).
4. **Demoable end-to-end.** A full load → chunk → embed → store → retrieve → generate
   pipeline you can watch, with a terminal and a Streamlit UI (FR8).

## 2. System Context

- **Corpus:** 5 public scheme pages (Groww) as front-mattered markdown in `documents/`
  (Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, Balanced Advantage).
- **Embedding:** fastembed, local ONNX `sentence-transformers/all-MiniLM-L6-v2` (PRD §9).
- **Vector DB:** persistent local Chroma (`data/chroma`); collection `hdfc_mf_faq`.
- **LLM:** any OpenAI-compatible `/chat/completions` API (`LLM_BASE_URL`, `LLM_API_KEY`,
  `LLM_MODEL`); e.g. Groq `openai/gpt-oss-120b`.
- **UIs:** `ragchat.cli` (terminal) and `ragchat/serve.py` (Streamlit).

```
[5 Groww scheme pages -> documents/*.md (front matter: URL, fetch date)]
        |  ingestion: load -> section/chunk -> embed -> Chroma
        v
[ChromaDB: id, text, vector, metadata {scheme, category, source_url, fetch_date, section}]

User question
   |  guardrails (PII? advice? -> refuse, no model call)
   |  embed -> Chroma similarity -> top-k
   |  gate (best_score >= SIMILARITY_THRESHOLD)
   v
LLM generates answer (<=3 sentences, one source URL) + "Last updated" footer + citations
```

## 3. Component Design

### 3.1 Module layout

- `ragchat/loaders.py` – files → anchored segments (one per PDF page, markdown heading,
  or whole text file); parses `---` front matter into shared metadata.
- `ragchat/chunker.py` – segments → token-counted overlapping windows, never spanning a
  segment, ids stable across re-ingests.
- `ragchat/embedder.py` – fastembed / hosted OpenAI-compatible embeddings.
- `ragchat/vectorstore.py` – Chroma wrapper: reset/add/query/all_chunks; cosine; normalize.
- `ragchat/retriever.py` – query conditioning + similarity + refusal gate.
- `ragchat/guardrails.py` – pre-retrieval classification (pii / advice / factual).
- `ragchat/prompts.py` – system prompt, refusal texts, context-block builder.
- `ragchat/generator.py` – LLM client with retries; deterministic "Last updated" footer.
- `ragchat/pipeline.py` – orchestrator; `Pipeline.ask()` is the only UI entry point.
- `ragchat/config.py` – env config + calibrated threshold default.
- `ragchat/ingest.py` – rebuild index + `data/manifest.json`.

### 3.2 Data contracts

`models.py`:

- `Chunk` – `id`, `text`, `metadata` dict. Convenience properties: `source_file`,
  `anchor` (page or section), `source_url`, `fetch_date`.
- `ScoredChunk` – `chunk`, `score` (1 − cosine distance), `rank`.
- `RetrieveResult` – hits + best_score + conditioned query + latencies.
- `Answer` – `text`, `citations`, `retrieved`, `refused`, `latency_ms`.

`loaders.Segment = (text, metadata)`. Front matter keys (`scheme`, `category`,
`source_url`, `fetch_date`) are merged into every segment of that file, so each chunk
carries the public link and fetch date needed for FR2/FR4 without joining later.

### 3.3 Component contracts

- `Embedder.embed(texts) -> list[list[float]]` (batched).
- `VectorStore.add(chunks, vectors)` stores normalized vectors; `query(vector, k)`;
  `reset()` clears before ingest; `count()`; `all_chunks()`.
- `Retriever.retrieve(question, history, k) -> RetrieveResult`; `passes_threshold(score)`.
- `Generator.generate(question, chunks, history) -> (text, citations)`.
- `Pipeline.ask(question, history) -> Answer`.

### 3.4 Ingestion pipeline (FR1)

`python -m ragchat.ingest --commit` (or `--dry-run` to inspect):

1. `discover()` all supported files under `documents/` (`.md/.txt/.pdf`).
2. `load()` per file → segments. Markdown is split on headings (field-oriented: one segment
   per "Expense Ratio", "Exit Load", … — the PRD's section-aware preference); front matter
   is merged into every segment.
3. `chunk_segments()` → token windows of `CHUNK_SIZE` with `CHUNK_OVERLAP`, per segment.
4. Reset the collection, embed in `EMBED_BATCH_SIZE` batches, `add()` normalized vectors.
5. Write `data/manifest.json` (embedding model, chunk size/overlap, count, source hash).

### 3.5 Retrieval (FR1)

`Retriever.retrieve` embeds the query (conditioned on recent history only when the question
actually refers back — see the `REFERRING_TERMS` docstring) and returns the top-k nearest
chunks by cosine similarity.

### 3.6 Generation (FR2, FR3, FR4) and citations

`Generator` builds the prompt from `prompts.build_prompt`: system prompt (facts-only, ≤3
sentences, exactly one source URL), a numbered fenced context block, recent history, then
the question. Fences are numbered (`<<<SOURCE 1>>>`) and password-free by construction: a
retrieved passage cannot open or close a fence it does not know.

- **One source link** is enforced as an instruction to the model (the single most relevant
  source URL). The UI also lists the actual citations as links, so the link is never lost.
- **"Last updated from sources"** is appended *deterministically* after generation from the
  newest `fetch_date` in the supplied chunks (`generator.last_updated_footer`), so FR4 holds
  even if the model omits it.
- **Citations** are derived from the chunks actually sent to the model, in order,
  de-duplicated (`unique_citations`). The model's own markers are never parsed, so no chunk
  outside the context can ever be cited.

### 3.7 Out-of-corpus gate (FR5)

`SIMILARITY_THRESHOLD` (calibrated per §5.3). Below it, `Pipeline.ask` returns `REFUSAL`
with the retrieved hits attached and makes no model call. A missing threshold falls back to
`DEFAULT_SIMILARITY_THRESHOLD` — an unset value never means "gate open".

### 3.8 Guardrails (FR5, FR6, FR7)

`guardrails.classify_question` runs **before** retrieval:

- **PII** (PAN, Aadhaar, email, phone, or keywords like OTP/account number) → `PII_RESPONSE`,
  refused; the text is never sent to the model and never written to disk (it survives only
  transiently in the browser page session).
- **Advice/opinion/comparison** ("should I buy…", "which fund is better…", return
  predictions) → `ADVICE_REFUSAL` with an educational link (AMFI).
- **Factual** → proceeds to retrieval and the gate.

### 3.9 Session and conversation state (FR2/FR4 continuity)

Threads are flat message lists of `Turn`s. Retrieval conditions follow-ups on up to
`RETRIEVAL_HISTORY_TURNS` of history; the generator replays only `HISTORY_TURNS`. The UI
keeps the thread in Streamlit session state.

### 3.10 UI (FR8)

`ragchat/serve.py`: welcome line, **3 example questions**, a visible "Facts-only. No
investment advice." disclaimer, then a chat that renders the answer, its source URL links,
retrieved-passage expander, and the refusal footer when gated.

### 3.11 Configuration

Env vars (see `.env.example`): `EMBEDDING_BACKEND/MODEL`, `LLM_BACKEND/BASE_URL/MODEL/KEY`,
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `SIMILARITY_THRESHOLD`, `HISTORY_TURNS`,
`RETRIEVAL_HISTORY_TURNS`, `EMBED_BATCH_SIZE`. Reading `*.env` via `python-dotenv`.

## 4. Request Flow

```
UI/CLI question
  -> classify_question            (pii/advice -> refuse, stop)
  -> retriever.retrieve           (embed + top-k)
  -> gate: best_score < threshold? -> REFUSAL (no model call)
  -> generator.generate           (prompt, LLM, footer, citations)
  -> Answer(text, citations, retrieved, refused)
```

## 5. Key Decisions

### 5.1 Hand-rolled pipeline, not a framework

Loaders/chunker/retriever/generator are explicit, testable modules. The demo's value is the
pipeline itself.

### 5.2 Chroma, local, rebuilt on ingest

Persistent local Chroma under `data/chroma`; `ingest --commit` resets and rebuilds it, so
index and corpus can never silently drift. Metadata (URL, date, scheme) is stored per chunk.

### 5.3 Threshold calibration

`scripts/calibrate_threshold.py` embeds a labeled question set (`scripts/demo_questions.jsonl`,
in-corpus vs out-of-corpus), reports score distributions, fails loudly if the classes
overlap, and recommends `(max_neg + min_pos) / 2`. The result is written into
`ragchat/config.py` (`DEFAULT_SIMILARITY_THRESHOLD`) and `.env.example`. Recalibrate after
changing the corpus, `EMBEDDING_MODEL`, `CHUNK_SIZE`, or `CHUNK_OVERLAP`.

### 5.4 Corpus fingerprint

`source_hash()` fingerprints the corpus independent of mtime/absolute path. It is stored in
the manifest and compared at startup, so a rebuilt-from-stale-code index is detected and
nudges you to re-ingest.

### 5.5 Two-sided prompt defense

Numbered fences + "answer only from the sources" + refusal-when-uncovered. The gate (§3.7)
keeps out-of-corpus text out of the context; the prompt keeps the model from inventing
inside it.

### 5.6 Local model fallback

`LocalLLM` is a stub (`available = False`). Without `LLM_API_KEY` the app refuses cleanly
and names the missing var at startup; it never silently answers without a model.

## 6. Error Handling

- Loader/chunker/embed/store failures raise typed errors; CLI exits non-zero with a message.
- `Pipeline.ask` converts runtime failures to a friendly `Answer`; configuration errors
  (missing key) are raised at startup, not per question.
- `HostedLLM.complete` retries transient failures (3×, exponential backoff) and raises
  `ProviderError` otherwise.

## 7. Security and Privacy

- PII never reaches the model or disk (FR7).
- `.env` is gitignored; generation never receives keys; `config.masked()` redacts secrets.
- Answers are scoped to the corpus; refusal is the default for anything else.

## 8. Trade-offs Accepted

- The corpus is small and fact-oriented; questions about schemes outside the 5 are refused.
- Facts can go stale (Groww pages change) → hence the "Last updated" footer, never an
  accuracy promise.
- Advice/comparison questions are refused wholesale rather than partially answered.
- "Exactly one source link" is a model instruction; the UI re-renders the real citations as
  links so the link requirement is still visually met.

## 9. Testing Strategy

- Unit + integration: chunking, loading, front matter, store, retrieval, gate, guardrails,
  generation (stubbed LLM), pipeline, manifest, UI-free CLI paths.
- Golden behavioral tests: the calibrated demo question set must keep in-corpus questions
  answering and out-of-corpus ones refusing.

## 10. Milestone Mapping

Phase-by-phase mapping lives in `implementation.md` (Phase 0–7) with traceability to
PRD FR1–FR8.

## 11. Known Limitations

- No rate limiting, auth, or monitoring (explicitly out of scope).
- The 5 pages are the whole corpus; no AMC factsheets are ingested yet.
- `all-MiniLM-L6-v2` is a lightweight embedding; tuning question wording matters.
- Statement-download Q&A ("how do I download a capital-gains statement") is in the PRD but
  the corpus has no such text yet, so those questions refuse until sources are added.

## 12. Open Technical Questions

- Whether to add AMC/AMFI factsheet sources to lower refusal rates on statement-download
  and tax questions.
- Whether the ELSS lock-in text should eventually cite the SID/KIM rather than the Groww badge.