"""Reciprocal Rank Fusion.

Combines several ranked lists into one:

    score(d) = sum over lists L of  1 / (k + rank_L(d))

Rank, not score. That is the whole point. Cosine similarity lives in [-1, 1]
and clusters tightly; BM25 is unbounded and depends on corpus statistics. The
two are not comparable, and any attempt to mix them numerically needs a
normalisation scheme that has to be retuned whenever the corpus changes. RRF
sidesteps that entirely by throwing away the magnitudes and keeping only the
ordering, which is the part both systems agree on the meaning of.

k (conventionally 60) damps the top of each list. Without it, rank 1 would
score 1.0 and rank 2 only 0.5, so a single list could dominate the fusion on
its own. With k = 60 the gap between rank 1 and rank 2 is about 1.6%, so a
document has to do well across MULTIPLE lists to rise -- which is exactly the
behaviour worth having when dense and sparse disagree.

Implemented here rather than delegated to Qdrant's server-side fusion, for two
reasons: it can be unit-tested against hand-computed values, and the ablation
needs to run dense-only, sparse-only and fused against identical candidates.
At this corpus size the round trip costs nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists into one, best first.

    Ties break on the ID so runs are reproducible rather than dependent on
    dict iteration order.

    `weights` lets one arm count for more than another. Left at None every list
    counts equally, which is the honest default until an ablation says
    otherwise.
    """
    if k < 1:
        raise ValueError(f"rrf k must be >= 1, got {k}")
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"{len(weights)} weights for {len(ranked_lists)} lists"
        )

    scores: dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists, strict=True):
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranked, start=1):
            # A list must not vote for the same document twice; a duplicate
            # would otherwise let one arm inflate a document on its own.
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def fuse_to_ids(
    ranked_lists: Sequence[Sequence[str]],
    limit: int,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[str]:
    return [doc_id for doc_id, _ in reciprocal_rank_fusion(ranked_lists, k, weights)][:limit]
