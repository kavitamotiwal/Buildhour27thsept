# PRD: Mutual Fund FAQ RAG Chatbot (Class Demo)

## 1. Summary
A small, facts-only RAG (Retrieval-Augmented Generation) chatbot that answers factual questions about a fixed set of HDFC mutual fund schemes — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements — using only official public scheme pages as its knowledge source. Every answer must cite exactly one source link. The bot must refuse opinion/advice questions (e.g., "should I buy X?").

This is a class demo project: the priority is a working, explainable end-to-end RAG pipeline (ingestion → retrieval → generation), not production robustness.

## 2. Problem Statement
Retail investors and support/content teams repeatedly ask the same factual questions about mutual fund schemes (fees, lock-ins, minimums, risk labels). Answering these manually is repetitive and error-prone, and people sometimes conflate factual lookups with investment advice. We need a small assistant that:
- Answers only from official public pages (no blogs, no hallucinated numbers).
- Always shows a source link per answer.
- Politely declines advice-seeking questions.

## 3. Goals
- Demonstrate a complete RAG pipeline: **Loading → Chunking → Embedding → Vector Store → Retrieval → Answer Generation**.
- Answer factual queries about 5 HDFC schemes with a cited source link in every response.
- Refuse opinionated/portfolio questions gracefully, pointing to an educational resource instead.
- Ship a minimal, demoable UI in the time available for a class project.

## 4. Non-Goals
- No investment advice, recommendations, or "buy/sell" guidance.
- No performance computation or cross-scheme return comparisons (only what the source states, with a link to the factsheet).
- No PII collection or storage (PAN, Aadhaar, account numbers, OTP, email, phone).
- No production-grade auth, multi-user accounts, or scaling — this is a class demo.
- No use of third-party blogs or unofficial sources for facts.

## 5. Target Users
- **Retail users** comparing mutual fund schemes who want quick factual lookups.
- **Support/content teams** who field repetitive MF questions and want a first-pass factual assistant.

## 6. Scope

### 6.1 AMC & Schemes (fixed corpus)
**AMC: HDFC Mutual Fund**

| Category | Scheme | Source URL |
|---|---|---|
| Large Cap | HDFC Large Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| Flexi Cap | HDFC Flexi Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| ELSS | HDFC ELSS Tax Saver Fund – Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| Small Cap | HDFC Small Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

These 5 public pages are the entire retrieval corpus for the demo (supplemented, if time allows, by official AMC/SEBI/AMFI pages such as factsheets, KIM/SID, fee/charges pages, riskometer notes, and statement/tax-doc guides — all must be public, official sources; no third-party blogs).

### 6.2 In-scope query types
- Expense ratio of a named scheme
- Exit load of a named scheme
- Minimum SIP / lumpsum amount
- ELSS lock-in period
- Riskometer level / benchmark index
- How to download a capital-gains / account statement

### 6.3 Out-of-scope query types (must be refused)
- "Should I buy/sell/switch X?"
- Performance comparisons or return predictions ("which fund will give best returns?")
- Any request to accept/store PAN, Aadhaar, account numbers, OTPs, personal email, or phone numbers

## 7. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | System answers factual queries using only retrieved content from the 5 scheme pages (+ any added official sources). |
| FR2 | Every factual answer includes exactly one source link. |
| FR3 | Every factual answer is ≤3 sentences. |
| FR4 | Every answer appends: "Last updated from sources: [date(s)]." |
| FR5 | System detects opinion/advice-seeking questions and responds with a polite facts-only refusal plus a relevant educational link (not a scheme recommendation). |
| FR6 | System never computes or compares returns; if asked, it points to the official factsheet link instead. |
| FR7 | System does not accept or store PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers — such inputs are not persisted and trigger a safe, generic response. |
| FR8 | UI shows a welcome line, 3 example questions, and a visible disclaimer: "Facts-only. No investment advice." |

## 8. System Architecture (RAG Pipeline)

The pipeline follows two stages, per the class rubric: **Data Ingestion** and **Data Retrieval**.

### 8.1 Data Ingestion
1. **Loading** — Fetch/scrape the 5 public scheme pages (HTML → cleaned text). Store raw + cleaned text per scheme with metadata (scheme name, category, source URL, fetch date).
2. **Chunking** — Chunking strategy to be decided based on the actual structure of the scraped data (e.g., section-aware chunking around fields like "Expense Ratio", "Exit Load", "Minimum SIP", "Riskometer", "Benchmark" rather than fixed-size blind splitting, since these pages are short and field-oriented). Each chunk retains metadata: scheme name, source URL, section label.
3. **Embedding** — Model: `sentence-transformers/all-MiniLM-L6-v2`. Each chunk is embedded into a vector.
4. **Vector Store** — ChromaDB collection storing chunk vectors + metadata (scheme, source URL, section, fetch date).

### 8.2 Data Retrieval
1. User query → embedded with the same `all-MiniLM-L6-v2` model.
2. Similarity search against ChromaDB to retrieve top-k relevant chunks.
3. **Guardrail check** — classify the query as factual vs. opinion/advice/PII before generation:
   - If opinion/advice → return the standard refusal + educational link, skip retrieval-based generation.
   - If PII-bearing → return the standard safe response, do not log/store the PII.
4. **Answer generation** — LLM composes an answer strictly from retrieved chunks, ≤3 sentences, with the single most relevant source URL attached, plus the "Last updated from sources" footer.

### 8.3 Architecture Diagram (textual)
```
[5 Public Scheme Pages]
        │  (Loading)
        ▼
[Cleaned Text + Metadata]
        │  (Chunking — field/section-aware)
        ▼
[Chunks]
        │  (Embedding: all-MiniLM-L6-v2)
        ▼
[ChromaDB Vector Store]

User Query → Embed → ChromaDB similarity search → Top-k chunks
        │
        ▼
  Guardrail (factual? opinion? PII?)
        │
   ┌────┴─────┐
   ▼          ▼
Refuse    Generate answer (≤3 sentences + 1 source link + "Last updated" footer)
```

## 9. Tech Stack
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB:** ChromaDB
- **Chunking:** Determined from the actual scraped data structure (section/field-aware over fixed-size)
- **LLM for generation:** TBD by team (any chat-capable model with tool/RAG support)
- **UI:** Minimal (chat interface, welcome line, 3 example questions, disclaimer banner)

## 10. Key Constraints
- **Public sources only.** No app back-end screenshots; no third-party blogs as sources.
- **No PII.** Do not accept/store PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers.
- **No performance claims.** No return computation/comparison; link to the official factsheet instead.
- **Clarity & transparency.** Answers ≤3 sentences; always append "Last updated from sources: ".

## 11. Deliverables
1. Working prototype link (app/notebook), or a ≤3-minute demo video if hosting isn't possible.
2. Source list (CSV/MD) of the 5 URLs used.
3. README with setup steps, scope (AMC + schemes), and known limitations.
4. Sample Q&A file (5–10 queries with the assistant's answers + links).
5. Disclaimer snippet used in the UI (facts-only, no advice).

## 12. Sample Q&A Format (for deliverable #4)
```
Q: What is the expense ratio of HDFC Flexi Cap Fund?
A: [Answer from retrieved chunk, ≤3 sentences]
Source: https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth
Last updated from sources: [date]

Q: Should I invest in HDFC Small Cap Fund right now?
A: I can only share facts from official scheme pages, not investment advice.
   For guidance on choosing funds, see [educational link].
```

## 13. Success Criteria (for the demo)
- All 5 schemes have working factual retrieval for: expense ratio, exit load, minimum SIP, riskometer, benchmark, (ELSS lock-in for the ELSS scheme).
- 100% of factual answers include exactly one source link and the "Last updated" footer.
- At least 3 opinion-style test questions are correctly refused with an educational link.
- No PII is echoed, stored, or logged when test PII inputs are submitted.

## 14. Risks & Known Limitations
- Source pages (Groww) may change layout/values over time — factsheet numbers can go stale; hence the "Last updated from sources" footer instead of implying real-time accuracy.
- Small corpus (5 pages) limits generalization; the bot should not answer questions about schemes outside this list.
- Chunking strategy needs validation once actual scraped content is seen — page structure may not be perfectly uniform across the 5 schemes.
- This is a demo-grade system: no rate limiting, auth, or production monitoring is in scope.
