"""Phases 4-6 -- score the v2 retriever as an ablation.

Each configuration adds exactly one thing to the previous, so every row of the
results table attributes a delta to a single change. A run that changed two
things at once would produce a number nobody can explain.

    v2_dense    clause-level units + bge-base-en-v1.5, dense only
    v2_hybrid   + BM25 sparse arm, fused with RRF
    v2_rerank   + bge-reranker-base cross-encoder over the fused candidates

Compared against eval_data/results/v1_baseline.json on the identical frozen
gold set, with the same metric code.

Two comparisons are reported, and the second is the honest one:

  at equal k          v1's k counts 800-word chunks, v2's counts ~66-word
                      clauses. Equal k is not equal information, and reading
                      only this column flatters v2 for reasons that have
                      nothing to do with retrieval quality.
  at equal budget     recall as a function of context WORDS delivered. This
                      answers the question that actually matters: how much
                      governing regulation reaches the LLM per token paid for.

Usage:
    python tools/score_v2.py --reindex
    python tools/score_v2.py --configs v2_dense v2_hybrid v2_rerank
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auditor.embedding import get_dense, get_sparse  # noqa: E402
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
from auditor.retrieval.fusion import fuse_to_ids  # noqa: E402
from auditor.retrieval.qdrant_store import QdrantClauseStore  # noqa: E402

K_VALUES = [1, 3, 5, 10]
# Budgets chosen to bracket what v1 actually delivered (800w per chunk, so
# 800 / 2400 / 4000 at its k=1 / 3 / 5).
BUDGETS = [200, 400, 800, 2400, 4000]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def recall_at_budget(
    ranked: list[str], words: dict[str, int], relevant: set[str], budget: int
) -> float:
    """Recall over as many results as fit in `budget` context words.

    Takes whole units only: a clause half inside the budget is not delivered,
    because half a clause is not something the LLM can cite.
    """
    if not relevant:
        return 0.0
    spent = 0
    taken: set[str] = set()
    for cid in ranked:
        cost = words.get(cid, 0)
        if spent + cost > budget:
            break
        spent += cost
        taken.add(cid)
    return len(taken & relevant) / len(relevant)


def evaluate(name: str, retrieve, gold, passages, clause_words, max_k) -> dict:
    per_query: list[dict] = []
    elapsed = 0.0

    for g in gold:
        passage = passages[g.passage_id]
        grades = g.labels
        primary = g.primary

        t0 = time.time()
        ranked = retrieve(passage["text"], max_k)
        elapsed += time.time() - t0

        row = {"passage_id": g.passage_id, "section": g.section, "metrics": {},
               "budget_recall": {}}
        for k in K_VALUES:
            top = ranked[:k]
            row["metrics"][str(k)] = {
                "recall": recall_at_k(top, primary, k=k),
                "precision": precision_at_k(top, primary, k=k),
                "ndcg": ndcg_at_k(top, grades, k=k),
                "context_precision": context_precision_at_k(top, grades, k=k),
                "mrr": reciprocal_rank(top, primary),
                "map": average_precision(top, primary),
                "hit_rate": hit_rate_at_k(top, primary, k=k),
                "scope_recall": scope_recall(top, g.primary_scopes, k=k),
                "clauses_returned": len(set(top)),
                "context_words": sum(clause_words.get(c, 0) for c in dict.fromkeys(top)),
            }
        for b in BUDGETS:
            row["budget_recall"][str(b)] = recall_at_budget(ranked, clause_words, primary, b)
        per_query.append(row)

    summary = {}
    for k in K_VALUES:
        key = str(k)
        summary[key] = {
            m: aggregate([q["metrics"][key][m] for q in per_query])["mean"]
            for m in ("recall", "precision", "ndcg", "context_precision", "mrr",
                      "map", "hit_rate", "scope_recall")
        }
        for extra in ("clauses_returned", "context_words"):
            summary[key][f"mean_{extra}"] = aggregate(
                [q["metrics"][key][extra] for q in per_query]
            )["mean"]

    budget_summary = {
        str(b): aggregate([q["budget_recall"][str(b)] for q in per_query])["mean"]
        for b in BUDGETS
    }

    return {
        "system": name,
        "summary": summary,
        "budget_recall": budget_summary,
        "timing": {"mean_query_ms": round(1000 * elapsed / max(len(gold), 1), 2)},
        "per_query": per_query,
    }


def rel(path: Path) -> str:
    """Display path, tolerant of outputs written outside the repo.

    Path.relative_to raises when the target is not under the base, which
    crashed a completed run at the final print line.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+",
                    default=["v2_dense", "v2_hybrid", "v2_rerank"])
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--collection", default="mdr_clauses_v2")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--candidates", type=int, default=25,
                    help="First-stage pool handed to the reranker.")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "eval_data" / "results" / "v2_ablation.json")
    args = ap.parse_args()

    clauses = load_jsonl(PROJECT_ROOT / "eval_data" / "clauses.jsonl")
    passages = {p["passage_id"]: p for p in load_jsonl(PROJECT_ROOT / "eval_data" / "passages.jsonl")}
    gold = load_gold(
        PROJECT_ROOT / "eval_data" / "gold_retrieval.jsonl",
        PROJECT_ROOT / "eval_data" / "clauses.jsonl",
    )
    clause_words = {c["clause_id"]: c["n_words"] for c in clauses}

    print(f"corpus   : {len(clauses)} clauses")
    print(f"queries  : {len(gold)} passages")

    dense = get_dense()
    sparse = get_sparse()
    store = QdrantClauseStore(args.collection, dense, sparse, url=args.qdrant_url)

    if args.reindex or not store.client.collection_exists(args.collection):
        print(f"indexing into '{args.collection}' ...")
        t0 = time.time()
        store.recreate()
        written = store.index(clauses)
        index_seconds = time.time() - t0
        print(f"  {written} points in {index_seconds:.1f}s")
    else:
        index_seconds = 0.0
        print(f"reusing collection '{args.collection}' ({store.count()} points)")

    reranker = None
    if "v2_rerank" in args.configs:
        from auditor.retrieval.rerank import CrossEncoderReranker

        print("loading cross-encoder (first run downloads ~1 GB) ...")
        reranker = CrossEncoderReranker()

    clause_text = {c["clause_id"]: c["text"] for c in clauses}

    def dense_only(query: str, k: int) -> list[str]:
        return [h.clause_id for h in store.search_dense(query, limit=k)]

    def hybrid(query: str, k: int) -> list[str]:
        pool = max(k, args.candidates)
        d = [h.clause_id for h in store.search_dense(query, limit=pool)]
        s = [h.clause_id for h in store.search_sparse(query, limit=pool)]
        return fuse_to_ids([d, s], limit=k)

    def hybrid_reranked(query: str, k: int) -> list[str]:
        pool = max(k, args.candidates)
        d = [h.clause_id for h in store.search_dense(query, limit=pool)]
        s = [h.clause_id for h in store.search_sparse(query, limit=pool)]
        candidates = fuse_to_ids([d, s], limit=pool)
        if not candidates:
            return []
        order = reranker.rerank(query, [clause_text[c] for c in candidates], top_k=k)
        return [candidates[item.index] for item in order]

    runners = {"v2_dense": dense_only, "v2_hybrid": hybrid, "v2_rerank": hybrid_reranked}

    results = {}
    for name in args.configs:
        print(f"\nevaluating {name} ...")
        results[name] = evaluate(
            name, runners[name], gold, passages, clause_words, max(K_VALUES)
        )

    payload = {
        "config": {
            "dense_model": dense.model_name,
            "sparse_model": sparse.model_name,
            "rerank_model": reranker.model_name if reranker else None,
            "candidates": args.candidates,
            "collection": args.collection,
            "clauses": len(clauses),
            "index_seconds": round(index_seconds, 2),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- report ----
    v1_path = PROJECT_ROOT / "eval_data" / "results" / "v1_baseline.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8")) if v1_path.exists() else None

    print()
    print("=" * 84)
    print("RETRIEVAL AT k=5".center(84))
    print("=" * 84)
    print(f"{'system':<14} {'Recall':>8} {'ScopeR':>8} {'nDCG':>8} {'MRR':>8} "
          f"{'CtxP':>8} {'words':>8} {'ms':>7}")
    print("-" * 84)
    if v1:
        s = v1["summary"]["5"]
        print(f"{'v1_baseline':<14} {s['recall']:>8.3f} {s.get('scope_recall', 0):>8.3f} "
              f"{s['ndcg']:>8.3f} {s['mrr']:>8.3f} {s['context_precision']:>8.3f} "
              f"{s['mean_context_words']:>8.0f} {v1['timing']['mean_query_ms']:>7.1f}")
    for name in args.configs:
        s = results[name]["summary"]["5"]
        print(f"{name:<14} {s['recall']:>8.3f} {s['scope_recall']:>8.3f} "
              f"{s['ndcg']:>8.3f} {s['mrr']:>8.3f} {s['context_precision']:>8.3f} "
              f"{s['mean_context_words']:>8.0f} "
              f"{results[name]['timing']['mean_query_ms']:>7.1f}")
    print("-" * 84)

    print()
    print("=" * 84)
    print("RECALL AT MATCHED CONTEXT BUDGET (words delivered to the LLM)".center(84))
    print("=" * 84)
    header = f"{'system':<14}" + "".join(f"{b:>10}w" for b in BUDGETS)
    print(header)
    print("-" * 84)
    if v1:
        # v1 delivers whole 800-word chunks, so its budget curve is a step
        # function derived from its per-k recall.
        row = f"{'v1_baseline':<14}"
        for b in BUDGETS:
            usable = max(1, b // 800)
            key = str(min(usable, 10))
            row += f"{v1['summary'][key]['recall']:>11.3f}" if key in v1["summary"] else f"{'-':>11}"
        print(row)
    for name in args.configs:
        row = f"{name:<14}"
        for b in BUDGETS:
            row += f"{results[name]['budget_recall'][str(b)]:>11.3f}"
        print(row)
    print("-" * 84)
    print(f"written to {rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
