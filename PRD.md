# PRD: Course RAG Chatbot (Class Demo)

**Status:** Draft — for team review
**Owner:** RAG Demo Team
**Target:** End of class project demo

> **Note on inputs:** `Problemstatement.txt` was empty when this PRD was drafted. The scope below
> is an assumption-based proposal built around a RAG chatbot demo. Items marked
> **[ASSUMPTION]** need confirmation before we lock the plan.

---

## 1. Overview

Build a retrieval-augmented generation (RAG) chatbot that answers questions about a fixed body
of course material, grounding every response in the source documents. The goal is a working,
explainable demo: a user asks a question, the system retrieves the relevant course notes and
returns an answer with citations back to those notes.

This is a **demo, not a production system.** It should be impressive in a 5-minute walkthrough
and honest about its limits.

## 2. Goals

| # | Goal | Success signal |
|---|------|----------------|
| G1 | Answer questions grounded in the course corpus | Answers cite the source document/section they came from |
| G2 | Say "I don't know" when the corpus lacks the answer | Out-of-corpus questions return a refusal, not a hallucination |
| G3 | Show the retrieval step in the UI | User can see which chunks were retrieved and their similarity scores |
| G4 | Demo-ready reliability | Runs from a single local command; no crashes across a 20-question demo script |

## 3. Non-Goals

- Multi-user accounts, auth, or sessions beyond a single conversation thread
- Writing back to the corpus, or document upload at runtime (corpus is fixed and pre-loaded)
- Fine-tuning or training any model
- Voice, images, or multi-modal input
- Production deployment, scaling, or cost optimization
- Broad subject coverage — one course, one document set, one demo

## 4. Users

| Persona | Need |
|---------|-------|
| Primary: instructor/TA | Ask a question mid-lecture, get a fast grounded answer with a source pointer |
| Secondary: classmates | Review material after class, ask "how does X relate to Y" |
| Tertiary: demo audience | Watch the system work and understand *why* it works (retrieval visibility) |

## 5. User Stories

- **US1** As a student, I ask a question in plain language so I can get an answer without
  reading 60 pages of notes.
- **US2** As a student, I see which document and page my answer came from, so I can verify it
  and go read the real material.
- **US3** As a student, I see the passages the system retrieved, so I can judge whether the
  answer is trustworthy.
- **US4** As a student, I ask something outside the course material and get an honest
  "not in my sources" instead of a confident wrong answer.
- **US5** As a student, I can ask a follow-up in the same conversation and it understands
  what I mean by "that" / "it".
- **US6** As the demo audience, I get an answer within a few seconds so the demo keeps moving.

## 6. Functional Requirements

### FR1 — Corpus Ingestion (offline, one-time)
- Load a fixed set of course documents (PDF / Markdown / plain text). **[ASSUMPTION]** PDF
  notes plus a few Markdown docs.
- Split each document into overlapping chunks with a fixed size and overlap. **[ASSUMPTION]**
  ~500 tokens with ~50 token overlap.
- Attach metadata to every chunk: `source_file`, `page` or `section`, `chunk_index`.
- Compute an embedding for each chunk and write chunks + embeddings + metadata to a local
  vector store.
- Runs as a separate script, not on every app start.

### FR2 — Retrieval
- Embed the user's question using the **same** embedding model as the corpus. **[ASSUMPTION]**
  a hosted embedding API for quality, with a documented local alternative if no key is available.
- Return the top *k* most similar chunks. **[ASSUMPTION]** k = 4.
- Support a score threshold: if the best match is below the threshold, treat the question as
  out-of-corpus and skip generation.
- Retrieval is single-shot. No reranking, query rewriting, or hybrid keyword search in v1.

### FR3 — Answer Generation
- Build a prompt from the retrieved chunks and the user's question, with explicit instructions
  to answer **only** from the provided context and to say when the context is insufficient.
- Use a chat LLM with a short, low-temperature setting for factual accuracy. **[ASSUMPTION]**
  temperature ~0.1–0.3.
- Return the generated answer alongside the list of citations.

### FR4 — Citation Display
- Each answer shows which chunks were used, with `source_file` + page/section.
- Citations are clickable/visible links back to the source material where the format allows.

### FR5 — Out-of-Corpus Handling
- If retrieval scores fall below threshold, return a fixed refusal message naming the corpus
  ("I can only answer from the course notes I was given") — no LLM call. **[ASSUMPTION]**
  rationale: deterministic, and demonstrably prevents hallucination.
- Non-blocking: a bad question never crashes the app.

### FR6 — Conversation Context
- The current turn's question is rewritten/conditioned on prior turns so follow-ups resolve.
  **[ASSUMPTION] v1 scope: the LLM sees the last *k* messages of history and the top chunks
  are retrieved for the raw question plus history.**
- Full session memory / summarization is out of scope.

### FR7 — Web UI
- Chat interface: message thread, input box, send control, loading state.
- For each assistant message: the answer text, then a citations row, and a collapsible
  "sources" panel showing the raw retrieved chunk text.
- Reset/clear conversation button.
- A small footer noting the corpus being used, so the demo audience knows the scope.

### FR8 — Demo Mode
- A pre-written script of ~20 questions (15 in-corpus, 5 out-of-corpus) stored in the repo,
  covering: direct factual lookup, "how does X relate to Y", definitions, and questions that
  genuinely aren't in the notes.
- One command to start the app; instructions in the README.

## 7. Non-Functional Requirements

| Area | Requirement |
|------|-------------|
| Latency | End-to-end answer in under ~5s for a typical question; retrieval under 1s **[ASSUMPTION]** |
| Setup | `pip install -r requirements.txt` + one env var for the API key; ingestion via one script |
| Privacy | Course material is sent to the model provider only as part of a request; no telemetry, no data retention by us |
| Robustness | Malformed/empty input, API errors, and timeouts surface a friendly message instead of a stack trace |
| Portability | Runs on macOS/Windows/Linux with Python 3.10+ **[ASSUMPTION]** |
| Code quality | Typed core modules, no secrets in the repo, `.env` gitignored |
| Explainability | Model, embedding model, chunk size, k, and threshold all configurable and printed at startup |

## 8. Technical Approach

```
Documents ──(ingest script)──> chunk + embed ──> local vector store
                                                        │
User question ──> embed ──> similarity search ──> top-k chunks
                                                        │
                          prior turns ─────────────────┤
                                                        v
                                              LLM (grounded prompt)
                                                        │
                                        answer + citations + raw chunks
```

**Stack [ASSUMPTION — confirm with team]:**

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.10+ | Ecosystem for vector/embedding libs, fastest to build for a demo |
| Interface | Streamlit | Days-to-hours to a decent chat UI; enough for a live demo |
| Orchestration | LangChain, or ~300 lines of hand-rolled glue | Hand-rolled is more impressive to explain and removes a dependency risk |
| Vector store | Chroma (local, zero-config) | No server to run before the demo |
| Embeddings | Hosted embedding API, local model fallback | Quality vs. offline-ability tradeoff |
| LLM | Hosted chat model | Quality; demo needs it to sound competent |
| Secrets | `.env`, gitignored | Never commit keys |

**Key design decision to present in the demo:** retrieval is grounded, cited, and thresholded, so
the failure mode is "I don't know" rather than a confident fabrication.

## 9. Success Criteria (Demo Acceptance)

The demo passes if, live, we can show:

1. A factual question answered correctly with visible citations. (G1)
2. The retrieved chunks shown in the UI, matching the answer. (G3)
3. At least one out-of-corpus question correctly refused. (G2)
4. One follow-up question that resolves a pronoun. (US5)
5. Total setup-to-running-app time under ~15 minutes on a clean machine.

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-------------|
| Requires network + paid API key on demo day | High | Demo fails | Cache a recorded fallback transcript; rehearse the offline path; cheap/free model option |
| Hallucination on a question we thought was covered | Medium | Credibility loss | Score threshold + strict grounded prompt + the prepared out-of-corpus questions as a safety demo |
| Bad chunking makes retrieval feel broken | Medium | Demo feels weak | Tune chunk size/overlap against the demo question set before presenting |
| Scope creep into "real product" | High | Missed deadline | Non-goals list is binding; new features need a swap, not an addition |
| Provider rate limit mid-demo | Low | Stutter | Retry with backoff; queue concurrent requests |

## 11. Milestones

| # | Milestone | Deliverable |
|---|-----------|-------------|
| M1 | Corpus + ingestion | Ingestion script, chunks embedded, vector store populated |
| M2 | Retrieval working (no LLM) | CLI that prints top-k chunks for a query — validates the core assumption |
| M3 | Answer generation + citations | CLI answers questions with source references |
| M4 | Out-of-corpus threshold | Refusal path verified against a 5-question set |
| M5 | Streamlit UI | Chat + sources panel + reset |
| M6 | Demo hardening | Demo script, README, timing check, offline fallback |

M2 is deliberately LLM-free: if retrieval feels wrong, we fix it before any prompt work starts.

## 12. Open Questions

1. What is the actual source corpus, and how large is it? (currently **[ASSUMPTION]**)
2. Which embedding and chat model — and is there a budget for API usage?
3. Is LangChain acceptable, or should this be dependency-light and hand-rolled?
4. Is the demo presented on a machine with reliable internet?
5. What is the hard deadline?
6. Should ingestion be per-student (each person ingests a subset) or a single shared prebuilt index?
7. Any grading rubric we should design the demo to hit?

## 13. Appendix — Assumptions to Confirm

Everything tagged **[ASSUMPTION]** is a placeholder chosen to be reasonable, not a decision:

- Corpus format and size; PDF notes + Markdown
- Chunk size 500 / overlap 50; top-k = 4
- Stack: Python + Streamlit + Chroma
- LangChain allowed as a dependency
- Hosted embeddings/LLM with a local fallback
- Similarity threshold exists and out-of-corpus skips generation
- Follow-up support limited to last-k message history
- ~20-question demo script, 15 in-corpus / 5 out-of-corpus
- No auth, no runtime upload, no production deployment
