"""Phase 2b -- score the v1 pipeline against the frozen gold set.

Produces eval_data/results/v1_baseline.json: the numbers every later phase is
measured against. Run once, commit the output, do not regenerate casually --
a baseline that moves is not a baseline.

Reported at several k, and also against a CONTEXT BUDGET. The second is the
honest comparison. v1 returns 800-word chunks and v2 returns ~66-word clauses,
so "top-3" means ~2400 words for one system and ~200 for the other. Comparing
at equal k would flatter v2 for reasons that have nothing to do with retrieval
quality; comparing at equal words answers the question that actually matters,
which is how much useful regulation reaches the LLM per token spent.

Usage:
    python tools/score_baseline_v1.py
    python tools/score_baseline_v1.py --k 1 3 5 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auditor.baselines.coverage import (  # noqa: E402
    build_spans,
    chunk_word_range,
    clauses_in_chunk,
)
from auditor.baselines.v1_retriever import (  # noqa: E402
    V1_OVERLAP,
    V1_WORDS_PER_CHUNK,
    V1Retriever,
    chunk_text_v1,
    extract_text_v1,
)
from auditor.evaluation.gold import load_gold, scope_recall  # noqa: E402
from auditor.evaluation.metrics import (  # noqa: E402
    aggregate,
    average_precision,
    context_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

SPAN_CACHE = PROJECT_ROOT / "eval_data" / ".span_cache.json"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def get_spans(document_text: str, clauses: list[dict], refresh: bool) -> list:
    """Locating 1320 clauses takes ~30s, so cache it. Keyed on corpus size and
    document length so a rebuilt corpus invalidates the cache automatically."""
    from auditor.baselines.coverage import ClauseSpan

    key = {"n_clauses": len(clauses), "n_words": len(document_text.split())}
    if not refresh and SPAN_CACHE.exists():
        cached = json.loads(SPAN_CACHE.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return [ClauseSpan(**s) for s in cached["spans"]]

    spans, missing = build_spans(document_text, clauses)
    if missing:
        print(f"  warning: {len(missing)} clauses unlocatable in v1 text", file=sys.stderr)
    SPAN_CACHE.write_text(
        json.dumps(
            {"key": key, "spans": [{"clause_id": s.clause_id, "start": s.start, "end": s.end}
                                   for s in spans]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return spans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--refresh-spans", action="store_true")
    ap.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "eval_data" / "results" / "v1_baseline.json"
    )
    args = ap.parse_args()

    started = time.time()

    print("[1/5] extracting guideline text the way v1 did ...")
    guideline_text = extract_text_v1(PROJECT_ROOT / "data" / "guideline.pdf")
    chunks = chunk_text_v1(guideline_text)
    total_words = len(guideline_text.split())
    print(f"      {total_words:,} words -> {len(chunks)} chunks "
          f"({V1_WORDS_PER_CHUNK}w, {V1_OVERLAP}w overlap)")

    print("[2/5] locating clauses in v1's word sequence ...")
    clauses = load_jsonl(PROJECT_ROOT / "eval_data" / "clauses.jsonl")
    spans = get_spans(guideline_text, clauses, args.refresh_spans)
    print(f"      {len(spans)}/{len(clauses)} clauses located")

    print("[3/5] precomputing chunk -> clause coverage ...")
    coverage = [
        clauses_in_chunk(i, spans, V1_WORDS_PER_CHUNK, V1_OVERLAP)
        for i in range(len(chunks))
    ]
    mean_cov = sum(len(c) for c in coverage) / max(len(coverage), 1)
    print(f"      mean {mean_cov:.1f} clauses per chunk")

    print("[4/5] embedding chunks with all-MiniLM-L6-v2 (first run downloads ~90 MB) ...")
    t0 = time.time()
    retriever = V1Retriever(chunks)
    index_seconds = time.time() - t0
    print(f"      indexed in {index_seconds:.1f}s")

    print("[5/6] measuring the encoder's effective input window ...")
    window_words = retriever.measure_embedding_window(chunks[len(chunks) // 2])
    visible_clauses: set[str] = set()
    for i in range(len(chunks)):
        lo, _ = chunk_word_range(i, V1_WORDS_PER_CHUNK, V1_OVERLAP)
        win_hi = lo + window_words
        for span in spans:
            if span.start < win_hi and span.end > lo:
                visible_clauses.add(span.clause_id)
    print(f"      only {window_words} of every {V1_WORDS_PER_CHUNK} words are embedded "
          f"({100 * window_words / V1_WORDS_PER_CHUNK:.0f}%)")

    print("[6/6] scoring against the gold set ...")
    passages = {p["passage_id"]: p for p in load_jsonl(PROJECT_ROOT / "eval_data" / "passages.jsonl")}
    gold = load_gold(
        PROJECT_ROOT / "eval_data" / "gold_retrieval.jsonl",
        PROJECT_ROOT / "eval_data" / "clauses.jsonl",
    )

    max_k = max(args.k)
    per_query: list[dict] = []
    query_seconds = 0.0

    for g in gold:
        passage = passages[g.passage_id]
        grades = g.labels
        primary = g.primary

        t0 = time.time()
        hits = retriever.search(passage["text"], k=max_k)
        query_seconds += time.time() - t0

        # A chunk delivers every clause it covers, in document order.
        ranked: list[str] = []
        words_at_k: list[int] = []
        words = 0
        for hit in hits:
            ranked.extend(coverage[hit.chunk_index])
            lo, hi = chunk_word_range(hit.chunk_index, V1_WORDS_PER_CHUNK, V1_OVERLAP)
            words += min(hi, total_words) - lo
            words_at_k.append(words)

        row = {
            "passage_id": g.passage_id,
            "section": g.section,
            "n_primary": len(primary),
            "chunks": [h.chunk_index for h in hits],
            "top_score": hits[0].score if hits else 0.0,
            "metrics": {},
        }
        for k in args.k:
            # k here is a number of CHUNKS; the clause list it yields is longer.
            ranked_k: list[str] = []
            for hit in hits[:k]:
                ranked_k.extend(coverage[hit.chunk_index])
            row["metrics"][str(k)] = {
                "recall": recall_at_k(ranked_k, primary, k=len(ranked_k) or 1),
                "precision": precision_at_k(ranked_k, primary, k=len(ranked_k) or 1),
                "ndcg": ndcg_at_k(ranked_k, grades, k=len(ranked_k) or 1),
                "context_precision": context_precision_at_k(
                    ranked_k, grades, k=len(ranked_k) or 1
                ),
                "mrr": reciprocal_rank(ranked_k, primary),
                "map": average_precision(ranked_k, primary),
                "hit_rate": hit_rate_at_k(ranked_k, primary, k=len(ranked_k) or 1),
                "scope_recall": scope_recall(ranked_k, g.primary_scopes,
                                             k=len(ranked_k) or 1),
                "clauses_returned": len(set(ranked_k)),
                "context_words": words_at_k[k - 1] if k <= len(words_at_k) else words,
            }
        per_query.append(row)

    summary = {}
    for k in args.k:
        key = str(k)
        summary[key] = {
            metric: aggregate([q["metrics"][key][metric] for q in per_query])["mean"]
            for metric in ("recall", "precision", "ndcg", "context_precision",
                           "mrr", "map", "hit_rate", "scope_recall")
        }
        summary[key]["mean_clauses_returned"] = aggregate(
            [q["metrics"][key]["clauses_returned"] for q in per_query]
        )["mean"]
        summary[key]["mean_context_words"] = aggregate(
            [q["metrics"][key]["context_words"] for q in per_query]
        )["mean"]

    reachable = sum(
        1
        for g in gold
        for cid, grade in g.labels.items()
        if grade == 2 and cid in visible_clauses
    )
    total_primary = sum(1 for g in gold for grade in g.labels.values() if grade == 2)

    result = {
        "system": "v1_baseline",
        "description": "fixed 800w/100w chunking, all-MiniLM-L6-v2 dense-only, top-k cosine",
        "config": {
            "words_per_chunk": V1_WORDS_PER_CHUNK,
            "overlap": V1_OVERLAP,
            "model": retriever.model_name,
            "chunks": len(chunks),
            "guideline_words": total_words,
        },
        "gold": {
            "passages": len(gold),
            "clauses": len(clauses),
            "clauses_located": len(spans),
        },
        "encoder_truncation": {
            "embedded_words_per_chunk": window_words,
            "chunk_words": V1_WORDS_PER_CHUNK,
            "fraction_of_chunk_embedded": round(window_words / V1_WORDS_PER_CHUNK, 4),
            "clauses_ever_embedded": len(visible_clauses),
            "clauses_total": len(clauses),
            "gold_primary_reachable": reachable,
            "gold_primary_total": total_primary,
            "recall_ceiling": round(reachable / max(total_primary, 1), 4),
            "note": (
                "all-MiniLM-L6-v2 accepts 256 word-pieces. Dense regulatory English "
                "exceeds that well before 800 words, so the tail of every chunk is "
                "truncated before embedding. recall_ceiling is the highest Recall any "
                "top-k over this index could reach: the remaining clauses were never "
                "encoded, so no value of k can retrieve them."
            ),
        },
        "timing": {
            "index_seconds": round(index_seconds, 2),
            "mean_query_ms": round(1000 * query_seconds / max(len(gold), 1), 2),
            "total_seconds": round(time.time() - started, 1),
        },
        "summary": summary,
        "per_query": per_query,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 78)
    print("v1 BASELINE".center(78))
    print("=" * 78)
    hdr = f"{'chunks':>6} {'clauses':>8} {'words':>7} {'Recall':>8} {'ScopeR':>8} {'nDCG':>8} {'MRR':>8}"
    print(hdr)
    print("-" * 78)
    for k in args.k:
        s = summary[str(k)]
        print(f"{k:>6} {s['mean_clauses_returned']:>8.1f} {s['mean_context_words']:>7.0f} "
              f"{s['recall']:>8.3f} {s['scope_recall']:>8.3f} {s['ndcg']:>8.3f} {s['mrr']:>8.3f}")
    print("-" * 78)
    print(f"encoder reads {window_words}/{V1_WORDS_PER_CHUNK} words per chunk "
          f"({100 * window_words / V1_WORDS_PER_CHUNK:.0f}%); "
          f"{len(visible_clauses)}/{len(clauses)} clauses ever embedded")
    print(f"RECALL CEILING: {reachable}/{total_primary} = "
          f"{100 * reachable / max(total_primary, 1):.1f}% "
          f"-- no k can beat this on this index")
    print("-" * 78)
    print(f"written to {args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
