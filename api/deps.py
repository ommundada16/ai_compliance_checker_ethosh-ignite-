"""Shared, lazily-built singletons for the API.

Models are expensive to load -- the cross-encoder alone is ~1 GB of ONNX -- and
loading them per request would make the first call to every endpoint time out.
They are built once on demand and reused.

Lazily rather than at import, so the process starts (and /health answers) even
when Qdrant or Ollama is down. An API that refuses to boot because a dependency
is unavailable cannot tell anyone WHICH dependency is unavailable.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env", override=False)

EVAL_DIR = PROJECT_ROOT / "eval_data"
RESULTS_DIR = EVAL_DIR / "results"


@lru_cache(maxsize=1)
def get_store():
    from auditor.embedding import get_dense, get_sparse
    from auditor.retrieval.qdrant_store import QdrantClauseStore

    return QdrantClauseStore(
        os.getenv("QDRANT_COLLECTION", "mdr_clauses_v2"),
        get_dense(),
        get_sparse(),
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", ""),
    )


@lru_cache(maxsize=1)
def get_reranker():
    from auditor.retrieval.rerank import CrossEncoderReranker

    return CrossEncoderReranker()


@lru_cache(maxsize=1)
def get_provider():
    """Groq first, local Ollama as the floor.

    Same chain the evaluation harness uses, so the API cannot quietly behave
    differently from the thing that was measured.
    """
    from auditor.llm.providers import FailoverProvider, GroqProvider, OllamaProvider

    chain: list = []
    if os.getenv("GROQ_API_KEY"):
        chain.append(
            GroqProvider(
                os.getenv("GROQ_API_KEY"),
                os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                os.getenv("GROQ_REASONING_EFFORT", "low"),
            )
        )
    chain.append(OllamaProvider(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b")))
    return FailoverProvider(chain)


@lru_cache(maxsize=1)
def get_judge():
    from auditor.llm.providers import OllamaProvider

    return OllamaProvider(model=os.getenv("JUDGE_MODEL", "llama3.1:8b"))


@lru_cache(maxsize=1)
def load_passages() -> list[dict]:
    import json

    path = EVAL_DIR / "passages.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@lru_cache(maxsize=1)
def load_clauses() -> dict[str, dict]:
    import json

    path = EVAL_DIR / "clauses.jsonl"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {
            (row := json.loads(line))["clause_id"]: row for line in fh if line.strip()
        }


def build_retriever(top_k_pool: int = 25, use_reranker: bool = True):
    """The v2 retrieval path, as a callable the pipeline can take.

    The returned `score` is always the score of the stage that decided the
    ORDER -- the RRF score when fusion is last, the cross-encoder score when
    reranking is. Returning the raw dense or sparse score instead (the obvious
    shortcut, since those objects are already in hand) produces a list whose
    displayed scores contradict its displayed order, which a ranking test
    caught and a user would have reported as "the sorting is broken".
    """
    from dataclasses import replace

    from auditor.retrieval.fusion import reciprocal_rank_fusion

    store = get_store()
    reranker = get_reranker() if use_reranker else None

    def retrieve(query: str, k: int):
        pool = max(k, top_k_pool)
        dense = store.search_dense(query, limit=pool)
        sparse = store.search_sparse(query, limit=pool)
        by_id = {c.clause_id: c for c in (*dense, *sparse)}

        fused = reciprocal_rank_fusion(
            [[c.clause_id for c in dense], [c.clause_id for c in sparse]]
        )[:pool]

        if reranker and fused:
            ids = [cid for cid, _ in fused]
            order = reranker.rerank(query, [by_id[c].text for c in ids], top_k=k)
            return [replace(by_id[ids[i.index]], score=i.score) for i in order]

        return [replace(by_id[cid], score=score) for cid, score in fused[:k]]

    return retrieve
