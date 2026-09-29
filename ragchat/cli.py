from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CONFIG, DEFAULT_SIMILARITY_THRESHOLD
from .errors import RagChatError

EXCERPT_CHARS = 200


def _load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _format_hit(hit) -> str:
    chunk = hit.chunk
    location = f"p.{chunk.metadata['page']}" if "page" in chunk.metadata else chunk.metadata.get("section", "")
    excerpt = " ".join(chunk.text.split())[:EXCERPT_CHARS]
    return f"  [{hit.rank}] score={hit.score:.4f}  {chunk.metadata.get('source_file', '')}  {location}\n      {excerpt}..."


def cmd_retrieve(question: str, history: list[dict] | None, k: int, backend: str | None) -> int:
    from .embedder import get_embedder
    from .retriever import Retriever

    retriever = Retriever(get_embedder(backend))
    result = retriever.retrieve(question, history, k=k)

    print(f"query   : {question}")
    print(f"embedded: {result.conditioned_query.replace(chr(10), ' | ')}")
    print(f"model   : {CONFIG.EMBEDDING_MODEL}  (backend={CONFIG.resolved_embedding_backend if backend is None else backend})")
    print(f"hits    : {len(result.hits)}\n")
    for hit in result.hits:
        print(_format_hit(hit))

    threshold = CONFIG.SIMILARITY_THRESHOLD
    if threshold is None:
        # Config carries a calibrated number in practice. If it ever does not, the retriever
        # falls back to DEFAULT_SIMILARITY_THRESHOLD and the gate stays CLOSED at it, so this
        # is a degraded-config notice rather than an open gate.
        verdict = f"no threshold in config, gate closed at the built-in {DEFAULT_SIMILARITY_THRESHOLD}"
    elif retriever.passes_threshold(result.best_score):
        verdict = "ABOVE threshold"
    else:
        verdict = "BELOW threshold -> would refuse"
    print(f"\nbest_score: {result.best_score:.4f}   {verdict}")
    return 0


def cmd_calibrate(questions_path: Path, backend: str | None) -> int:
    from .embedder import get_embedder
    from .retriever import Retriever

    retriever = Retriever(get_embedder(backend))
    rows = []
    for item in _load_questions(questions_path):
        result = retriever.retrieve(item["question"])
        top = result.hits[0].chunk if result.hits else None
        rows.append(
            {
                "question": item["question"],
                "in_corpus": bool(item.get("in_corpus")),
                "best_score": result.best_score,
                "top_file": top.metadata.get("source_file", "") if top else "",
                "top_anchor": (
                    f"p.{top.metadata['page']}" if top and "page" in top.metadata
                    else (top.metadata.get("section", "") if top else "")
                ),
            }
        )

    positives = [r for r in rows if r["in_corpus"]]
    negatives = [r for r in rows if not r["in_corpus"]]

    print(f"{'score':>7}  {'label':<7} {'top source':<26} {'question'}")
    print("-" * 96)
    for row in sorted(rows, key=lambda r: -r["best_score"]):
        label = "IN" if row["in_corpus"] else "OUT"
        print(f"{row['best_score']:>7.4f}  {label:<7} {row['top_file']:<26} {row['question'][:52]}")

    pos_scores = [r["best_score"] for r in positives]
    neg_scores = [r["best_score"] for r in negatives]
    print("\n--- distributions ---")
    print(f"in-corpus  n={len(pos_scores):<3} min={min(pos_scores):.4f} max={max(pos_scores):.4f}")
    print(f"out-of-corpus n={len(neg_scores):<3} min={min(neg_scores):.4f} max={max(neg_scores):.4f}")

    print("\n--- expected-source check (in-corpus only) ---")
    expected = {item["question"]: item for item in _load_questions(questions_path)}
    mismatches = 0
    for row in positives:
        want = expected[row["question"]]
        ok = row["top_file"] == want.get("expected_source_file")
        if want.get("expected_page") is not None and ok:
            ok = row["top_anchor"] == f"p.{want['expected_page']}"
        if not ok:
            mismatches += 1
            print(f"  MISS  wanted {want.get('expected_source_file')} {want.get('expected_page', want.get('expected_section', ''))}")
            print(f"        got    {row['top_file']} {row['top_anchor']}")
            print(f"        {row['question']}")
    if mismatches == 0:
        print(f"  all {len(positives)} in-corpus questions retrieved the expected source")

    print("\n--- threshold ---")
    if max(neg_scores) >= min(pos_scores):
        overlap_from, overlap_to = max(neg_scores), min(pos_scores)
        print(f"  NO THRESHOLD SEPARATES THE CLASSES. out-of-corpus max {overlap_from:.4f} >= in-corpus min {overlap_to:.4f}")
        print("  This is a retrieval-quality problem, not a threshold problem (architecture.md section 5.3).")
        print("  Retune CHUNK_SIZE / CHUNK_OVERLAP, then rerun. Do not lower the threshold to force separation.")
        return 1

    recommended = (max(neg_scores) + min(pos_scores)) / 2
    print(f"  recommended: SIMILARITY_THRESHOLD={recommended:.4f}")
    print(f"  margin: {min(pos_scores) - recommended:.4f} below the weakest in-corpus question,"
          f" {recommended - max(neg_scores):.4f} above the strongest out-of-corpus question")
    return 0


def cmd_ask(question: str, k: int, backend: str | None, threshold: float | None, show_prompt: bool) -> int:
    from .embedder import get_embedder
    from .prompts import REFUSAL, build_prompt
    from .retriever import Retriever

    retriever = Retriever(get_embedder(backend))
    result = retriever.retrieve(question, k=k)
    gate = CONFIG.SIMILARITY_THRESHOLD if threshold is None else threshold

    if not retriever.passes_threshold(result.best_score, threshold=gate):
        print(f"question   : {question}")
        print(f"top score  : {result.best_score:.4f}  (below {gate})")
        print(f"\nREFUSED, no model call made.\n{REFUSAL}")
        print("\nweakest matches (shown so you can see why it refused):")
        for hit in result.hits:
            print(f"  [{hit.rank}] {hit.score:.4f}  {hit.chunk.metadata.get('source_file', '')}  {hit.chunk.anchor}")
        return 0

    chunks = [hit.chunk for hit in result.hits]
    if show_prompt:
        print(f"question   : {question}")
        print(f"top score  : {result.best_score:.4f}  (above {gate})")
        print(f"model      : {CONFIG.LLM_MODEL}  (backend={CONFIG.resolved_llm_backend})  [NOT CALLED]\n")
        for number, message in enumerate(build_prompt(question, chunks), start=1):
            print(f"{'=' * 78}\nMESSAGE {number}  role={message['role']}\n{'=' * 78}")
            print(message["content"])
            print()
        return 0

    from .generator import Generator, get_llm

    text, citations = Generator(get_llm()).generate(question, chunks)
    print(f"question   : {question}")
    print(f"top score  : {result.best_score:.4f}  (above {gate})")
    print(f"model      : {CONFIG.LLM_MODEL}  (backend={CONFIG.resolved_llm_backend})\n")
    print("--- answer ---")
    print(text)
    print(f"\n--- citations ({len(citations)}) ---")
    for number, chunk in enumerate(citations, start=1):
        print(f"  [{number}] {chunk.metadata.get('source_file', '')}  {chunk.anchor}")
    return 0


def cmd_run_demo(questions_path: Path, backend: str | None, check_only: bool) -> int:
    from .embedder import get_embedder
    from .pipeline import Pipeline, describe, startup_checks

    print("--- startup checks ---")
    issues = startup_checks()
    for level, message in issues:
        print(f"  [{level.upper()}] {message}")
    if not issues:
        print("  ok")
    print()

    for key, value in describe().items():
        print(f"  {key:<22}: {value}")
    print()

    embedder = get_embedder(backend)
    llm = None
    if not check_only:
        from .generator import get_llm

        llm = get_llm()
    pipeline = Pipeline(embedder=embedder, llm=llm)

    questions = _load_questions(questions_path)
    print(f"--- {len(questions)} demo questions ({'pre-flight: retrieval + gate only' if check_only else 'full run'}) ---\n")
    header = f"{'#':>2}  {'kind':<14} {'act':<6} {'score':>6}  {'top source':<26} answer / reason"
    print(header)
    print("-" * len(header))

    answered = refused = 0
    failures: list[str] = []
    slowest = (0, "")
    correct_refusals = expected_refusals = 0
    misrouted: list[str] = []

    for number, item in enumerate(questions, start=1):
        expect_refusal = not item.get("in_corpus", True)
        if expect_refusal:
            expected_refusals += 1
        try:
            answer = pipeline.preflight(item["question"]) if check_only else pipeline.ask(item["question"])
        except RagChatError as exc:
            failures.append(f"{item['question']}: {exc}")
            print(f"{number:>2}  {item.get('kind', '?'):<14} {'ERR':<6} {'-':>6}  {'-':<26} {exc}")
            continue

        if answer.refused and expect_refusal:
            correct_refusals += 1
        if answer.refused != expect_refusal and answer.refused is True and expect_refusal is False:
            misrouted.append(f"#{number} {item['question']}")

        top = answer.retrieved[0].chunk if answer.retrieved else None
        top_file = top.metadata.get("source_file", "") if top else ""
        score = answer.retrieved[0].score if answer.retrieved else 0.0
        latency = answer.latency_ms.get("total", 0)
        if latency > slowest[0]:
            slowest = (latency, item["question"])

        if answer.refused:
            refused += 1
            note = "REFUSED (expected)" if expect_refusal else "REFUSED (UNEXPECTED)"
        else:
            answered += 1
            if expect_refusal:
                note = "answered (UNEXPECTED - should have been refused)"
            elif check_only:
                note = "would answer"
            else:
                note = f"answered, {len(answer.citations)} citations"

        print(f"{number:>2}  {item.get('kind', '?'):<14} {'OK':<6} {score:>6.3f}  {top_file:<26} {note}")

    print("\n--- summary ---")
    print(f"questions   : {len(questions)}")
    print(f"answered    : {answered}")
    print(f"refused     : {refused}  (expected {expected_refusals})")
    print(f"gate correct: {correct_refusals}/{expected_refusals} out-of-corpus questions refused")
    print(f"in-corpus wrongly refused: {len(misrouted)}")
    for item in misrouted:
        print(f"    {item}")
    print(f"slowest turn: {slowest[0]}ms  {slowest[1][:56]}")
    if failures:
        print(f"failures    : {len(failures)}")
        for failure in failures:
            print(f"    {failure}")
        return 1
    if check_only:
        print("\npre-flight only: no answers were generated. Re-run without --check once an LLM is configured.")
    return 0


def cmd_chat(backend: str | None) -> int:
    """Interactive REPL. This is the pre-UI way to prove the bot actually answers."""
    from .embedder import get_embedder
    from .generator import get_llm
    from .models import Turn
    from .pipeline import Pipeline, describe, startup_checks

    print("--- startup checks ---")
    issues = startup_checks()
    for level, message in issues:
        print(f"  [{level.upper()}] {message}")
    if not issues:
        print("  ok")
    print()

    for key, value in describe().items():
        print(f"  {key:<22}: {value}")
    print()

    llm = None
    if any(level == "error" and "LLM" in message for level, message in issues):
        print("  No model available, so this session is PRE-FLIGHT: retrieval and the gate only.")
        print("  Out-of-corpus questions will still refuse correctly. Nothing will be generated.")
        print()
    else:
        llm = get_llm()

    pipeline = Pipeline(embedder=get_embedder(backend), llm=llm)
    history: list[Turn] = []

    print("Commands: /reset clears the thread   /sources reprints the index   /exit to quit")
    print("-" * 72)

    while True:
        try:
            raw = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not raw:
            continue
        if raw in {"/exit", "/quit"}:
            print("bye")
            return 0
        if raw == "/reset":
            history.clear()
            print("(thread cleared)")
            continue
        if raw == "/sources":
            for index, hit in enumerate(pipeline.store.all_chunks(), start=1):
                print(f"  [{index}] {hit.metadata.get('source_file', '?')}  {hit.metadata.get('section', '')}".rstrip())
            continue

        answer = pipeline.ask(raw, history)
        print()
        if answer.refused:
            print(f"bot > {answer.text}")
            best = answer.retrieved[0] if answer.retrieved else None
            if best:
                print(f"       (best match {best.score:.4f}, below the {pipeline.threshold} gate)")
        else:
            print(f"bot > {answer.text}")
            for index, citation in enumerate(answer.citations, start=1):
                print(f"       [{index}] {citation.source_file}  {citation.anchor}")
            print(f"       ({answer.latency_ms.get('generate', 0)}ms generate, {answer.latency_ms.get('total', 0)}ms total)")

        history.append(Turn(role="user", content=raw))
        history.append(Turn(role="assistant", content=answer.text))


def cmd_export(out: str | None, width: int, full: bool) -> int:
    from .export import export_chunks

    path = export_chunks(out_path=out, width=width, full=full)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    # A model answer can contain any Unicode. The Windows console defaults to a legacy
    # codepage and raises UnicodeEncodeError on characters like 【, which would crash a
    # perfectly good answer. errors="replace" keeps the turn alive instead.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(prog="ragchat.cli", description="Headless retrieval and generation tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    retrieve = sub.add_parser("retrieve", help="print top-k retrieved chunks for a question")
    retrieve.add_argument("question")
    retrieve.add_argument("--k", type=int, default=CONFIG.TOP_K)
    retrieve.add_argument("--turns", type=int, default=CONFIG.RETRIEVAL_HISTORY_TURNS, help="prior turns to condition on")
    retrieve.add_argument("--backend", default=None, choices=[None, "hosted", "local"])

    calibrate = sub.add_parser("calibrate", help="score the demo question set and recommend a threshold")
    calibrate.add_argument("--questions", default=str(Path(__file__).resolve().parent.parent / "scripts" / "demo_questions.jsonl"))
    calibrate.add_argument("--backend", default=None, choices=[None, "hosted", "local"])

    ask = sub.add_parser("ask", help="retrieve, gate, and answer a question end to end")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=CONFIG.TOP_K)
    ask.add_argument("--backend", default=None, choices=[None, "hosted", "local"])
    ask.add_argument("--threshold", type=float, default=None, help="override the calibrated threshold")
    ask.add_argument("--show-prompt", action="store_true", help="print the assembled prompt and stop; never calls the model")

    chat = sub.add_parser("chat", help="interactive terminal chat, no UI needed")
    chat.add_argument("--backend", default=None, choices=[None, "hosted", "local"])

    demo = sub.add_parser("run-demo", help="run the demo question set end to end")
    demo.add_argument("--questions", default=str(Path(__file__).resolve().parent.parent / "scripts" / "demo_questions.jsonl"))
    demo.add_argument("--backend", default=None, choices=[None, "hosted", "local"])
    demo.add_argument("--check", action="store_true", help="pre-flight: retrieval and gate only, never generates")

    export = sub.add_parser("export", help="write chunks and embeddings to a readable text file")
    export.add_argument("--out", default=None, help="default: data/chunks.txt")
    export.add_argument("--dims", type=int, default=12, help="how many vector dimensions to show per chunk")
    export.add_argument("--full", action="store_true", help="print every dimension instead of the first --dims")

    args = parser.parse_args(argv)
    try:
        if args.command == "retrieve":
            return cmd_retrieve(args.question, [], args.k, args.backend)
        if args.command == "export":
            return cmd_export(args.out, args.dims, args.full)
        if args.command == "ask":
            return cmd_ask(args.question, args.k, args.backend, args.threshold, args.show_prompt)
        if args.command == "chat":
            return cmd_chat(args.backend)
        if args.command == "run-demo":
            return cmd_run_demo(Path(args.questions), args.backend, args.check)
        return cmd_calibrate(Path(args.questions), args.backend)
    except RagChatError as exc:
        print(f"error: {exc}")
        return 1
    except ValueError as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
