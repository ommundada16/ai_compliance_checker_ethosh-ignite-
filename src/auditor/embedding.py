"""Embedding providers: dense and sparse, both ONNX on CPU.

The GPU budget on the development machine is 4 GB and it belongs to the LLM.
Embedding and reranking are batch operations over short texts, which CPU
handles perfectly well; generation is the latency-critical path, which it does
not. So nothing here touches the GPU, and nothing here depends on torch.

Two vector kinds, for two different failure modes:

  dense   captures meaning. "the device must not harm the patient" retrieves a
          clause about minimising risk even with no shared vocabulary.
  sparse  captures exact tokens. Regulatory text is full of identifiers --
          "Annex XIV", "PMCF", "Article 61(4)", "Class IIb" -- where the
          literal string IS the meaning, and a dense model cheerfully returns
          a semantically adjacent clause with the wrong number on it.

Neither is sufficient alone, which is why retrieval fuses both.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

# Model artefacts live outside the project. The repository sits in a
# OneDrive-synced folder, and a sync client grinding through hundreds of
# megabytes of ONNX weights on every checkout is a self-inflicted wound.
DEFAULT_CACHE = Path.home() / ".cache" / "fastembed"

DENSE_MODEL = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"

# bge-base-en-v1.5 was trained with an instruction prefix on the QUERY side
# only. Omitting it costs a few points of recall; applying it to documents as
# well costs more. The asymmetry is deliberate, not an oversight.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class SparseVector:
    """Qdrant's sparse representation: parallel index and value arrays."""

    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class DenseEmbedder:
    """bge-base-en-v1.5, 768 dimensions.

    Chosen over the MiniLM v1 used partly for capacity: MiniLM accepts 256
    word-pieces, and measuring that limit against 800-word chunks is what
    revealed v1 was embedding 13% of its corpus. bge-base accepts 512, and --
    more importantly -- v2 feeds it clause-sized units that fit inside the
    window rather than slabs that do not.
    """

    def __init__(self, model_name: str = DENSE_MODEL, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir or DEFAULT_CACHE)
        )

    @property
    def dim(self) -> int:
        for spec in self._model.list_supported_models():
            if spec["model"] == self.model_name:
                return int(spec["dim"])
        raise ValueError(f"unknown model {self.model_name}")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return l2_normalise(np.asarray(list(self._model.embed(list(texts))), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.asarray(
            list(self._model.embed([QUERY_PREFIX + text])), dtype=np.float32
        )
        return l2_normalise(vector)[0]

    def effective_window_words(self, sample: str, max_words: int = 1000) -> int:
        """Shortest prefix whose embedding is identical to the whole text's.

        The same measurement that exposed v1's truncation. Kept on the v2
        embedder so the claim "v2 chunks fit inside the window" is checked
        rather than assumed -- a chunker that drifts past the limit would
        reintroduce v1's defect silently.
        """
        words = sample.split()[:max_words]
        full = self.embed_documents([" ".join(words)])[0]
        lo, hi = 1, len(words)
        while lo < hi:
            mid = (lo + hi) // 2
            prefix = self.embed_documents([" ".join(words[:mid])])[0]
            if float(prefix @ full) > 0.99999:
                hi = mid
            else:
                lo = mid + 1
        return lo


class SparseEmbedder:
    """BM25 term weights, as a sparse vector.

    Qdrant's BM25 implementation, so corpus statistics and stemming stay
    consistent between what is indexed and what is queried.
    """

    def __init__(self, model_name: str = SPARSE_MODEL, cache_dir: Path | None = None) -> None:
        from fastembed import SparseTextEmbedding

        self.model_name = model_name
        self._model = SparseTextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir or DEFAULT_CACHE)
        )

    @staticmethod
    def _convert(raw: Iterable) -> list[SparseVector]:
        return [
            SparseVector(indices=[int(i) for i in r.indices],
                         values=[float(v) for v in r.values])
            for r in raw
        ]

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return self._convert(self._model.embed(list(texts)))

    def embed_query(self, text: str) -> SparseVector:
        # No QUERY_PREFIX here. It is a dense-model instruction; as literal
        # tokens it would only add noise terms to a BM25 vector.
        return self._convert(self._model.query_embed([text]))[0]


@lru_cache(maxsize=4)
def get_dense(model_name: str = DENSE_MODEL) -> DenseEmbedder:
    """Cached: loading an ONNX session costs seconds, and an evaluation sweep
    constructs retrievers repeatedly."""
    return DenseEmbedder(model_name)


@lru_cache(maxsize=4)
def get_sparse(model_name: str = SPARSE_MODEL) -> SparseEmbedder:
    return SparseEmbedder(model_name)
