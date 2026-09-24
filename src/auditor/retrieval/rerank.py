"""Cross-encoder reranking.

The second stage of a two-stage retrieval. The first stage casts a wide net
cheaply; this one reads each candidate properly and reorders.

Why a cross-encoder beats the bi-encoder that produced the candidates
---------------------------------------------------------------------
A bi-encoder embeds the query and the document SEPARATELY and compares the two
vectors. The document's vector is computed once, at index time, with no
knowledge of any query -- it has to be a single point that serves every
possible question at once. A cross-encoder takes the pair TOGETHER and runs
attention across both, so "does this clause govern this passage" is answered
with the passage in view.

The cost is that nothing can be precomputed. Every (query, document) pair is a
forward pass, so scoring the whole 1320-clause corpus per query is out of the
question. That asymmetry is exactly why the architecture is two-stage: the
bi-encoder's precomputed index cheaply reduces 1320 to ~25, and the
cross-encoder spends real compute only on those.

Runs on CPU, like the embedders. The 4 GB of VRAM belongs to the LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE = Path.home() / ".cache" / "fastembed"
RERANK_MODEL = "BAAI/bge-reranker-base"


@dataclass(frozen=True)
class RerankedItem:
    index: int      # position in the input list
    score: float    # cross-encoder relevance, NOT comparable to a cosine


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL, cache_dir: Path | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(
            model_name=model_name, cache_dir=str(cache_dir or DEFAULT_CACHE)
        )

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return [float(s) for s in self._model.rerank(query, list(documents))]

    def rerank(
        self, query: str, documents: Sequence[str], top_k: int | None = None
    ) -> list[RerankedItem]:
        """Reorder `documents` by relevance to `query`, best first.

        Ties break on the original index, so a reranker that cannot separate
        two candidates leaves them in first-stage order rather than shuffling
        them unpredictably between runs.
        """
        scores = self.score(query, documents)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        if top_k is not None:
            order = order[:top_k]
        return [RerankedItem(index=i, score=scores[i]) for i in order]
