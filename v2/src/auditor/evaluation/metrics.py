"""Retrieval metrics.

Every function here takes a ranked list of clause_ids and the gold judgements
for that query, and returns a single float. No I/O, no configuration, no
model -- so each one can be tested against a hand-computed answer, which is the
only way to be sure a metric is right. A metric implemented incorrectly does
not crash; it returns a plausible number that is quietly wrong, and every
comparison built on it inherits the error.

Conventions used throughout:

  retrieved   ranked list of clause_ids, best first, possibly with duplicates
  relevant    set of clause_ids with grade 2 (primary)
  grades      clause_id -> grade, where 2 is primary and 1 is secondary

Binary metrics (recall, precision, MRR, MAP, hit rate) count PRIMARY labels
only. Graded metrics (nDCG) use both grades. That split is deliberate: see
tools/mdr_section_map.py for why relevance here is not binary.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "dedupe_preserving_order",
    "hit_rate_at_k",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "context_precision_at_k",
    "aggregate",
]


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    """Drop repeat hits, keeping the best-ranked occurrence.

    A fused hybrid retriever can legitimately return the same clause from both
    the dense and sparse arms. Left in place, a duplicate at ranks 1 and 2
    would let a system score Recall@2 = 2 relevant "hits" from one document.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _prepare(retrieved: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return dedupe_preserving_order(retrieved)[:k]


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any primary clause appears in the top k, else 0.0.

    The most forgiving metric: it answers "did we find anything useful at all".
    Useful as a floor -- a system failing this is not merely ranking badly.
    """
    if not relevant:
        return 0.0
    return 1.0 if set(_prepare(retrieved, k)) & relevant else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of primary clauses found in the top k.

    The headline retrieval metric for this project. The LLM cannot cite a
    clause it was never shown, so recall bounds how well the whole pipeline can
    possibly do -- no prompt or guardrail downstream recovers a clause that
    retrieval missed.

    Queries with no primary labels return 0.0 rather than 1.0. Such a query is
    a gold-set defect, and returning a perfect score for it would hide the
    defect inside a good-looking average; a test asserts none exist.
    """
    if not relevant:
        return 0.0
    top = set(_prepare(retrieved, k))
    return len(top & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that is primary.

    Divided by k, not by the number retrieved, so that returning fewer than k
    results is not rewarded. A system that returns one correct clause and stops
    should not score the same as one that returns k correct clauses.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top = _prepare(retrieved, k)
    return sum(1 for cid in top if cid in relevant) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first primary clause; 0.0 if none is retrieved.

    Averaged across queries this is MRR. It cares only about the first hit, so
    it answers "how far down the list must a reviewer read before finding
    something that governs this passage".
    """
    if not relevant:
        return 0.0
    for idx, cid in enumerate(dedupe_preserving_order(retrieved), start=1):
        if cid in relevant:
            return 1.0 / idx
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """Mean of precision@i taken at each rank i holding a primary clause.

    Averaged across queries this is MAP. Unlike MRR it rewards finding ALL the
    governing clauses, and unlike recall it rewards finding them early.

    The denominator is the total number of primary clauses, not the number
    found, so unretrieved ones are penalised rather than ignored.
    """
    if not relevant:
        return 0.0
    ranked = dedupe_preserving_order(retrieved)
    if k is not None:
        ranked = ranked[:k]
    hits = 0
    total = 0.0
    for idx, cid in enumerate(ranked, start=1):
        if cid in relevant:
            hits += 1
            total += hits / idx
    return total / len(relevant)


def dcg_at_k(retrieved: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Discounted cumulative gain, exponential-gain form.

        DCG@k = sum_i (2^grade_i - 1) / log2(i + 1)

    The exponential gain is what makes the two grades behave differently: a
    primary clause (2^2-1 = 3) is worth three times a secondary one (2^1-1 = 1),
    rather than merely twice as with linear gain. That matches the intent --
    surfacing the governing clause matters much more than surfacing related
    context.
    """
    top = _prepare(retrieved, k)
    return sum(
        (2 ** grades.get(cid, 0) - 1) / math.log2(rank + 1)
        for rank, cid in enumerate(top, start=1)
    )


def ndcg_at_k(retrieved: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """DCG@k normalised by the best achievable DCG@k for this query.

    The ideal ranking is every graded clause sorted by grade descending, then
    truncated to k -- so a query with more relevant clauses than k is not
    punished for the impossibility of returning them all.
    """
    ideal_grades = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum(
        (2 ** g - 1) / math.log2(rank + 1)
        for rank, g in enumerate(ideal_grades, start=1)
    )
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, grades, k) / idcg


def context_precision_at_k(
    retrieved: Sequence[str], grades: Mapping[str, int], k: int
) -> float:
    """Fraction of the top k that carries ANY relevance (grade 1 or 2).

    Distinct from precision_at_k, which counts primary only. This one measures
    context pollution: how much of what gets stuffed into the LLM prompt is
    simply irrelevant. It matters directly for cost and for hallucination --
    irrelevant context is tokens paid for and a surface for the model to
    confabulate against.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top = _prepare(retrieved, k)
    return sum(1 for cid in top if grades.get(cid, 0) > 0) / k


def aggregate(values: Sequence[float]) -> dict[str, float]:
    """Macro-average a per-query metric, with spread.

    Macro (mean over queries) rather than micro (pool all hits) because every
    passage should count equally. Micro-averaging would let the handful of
    sections split into six passages dominate the score.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
        "n": n,
    }
