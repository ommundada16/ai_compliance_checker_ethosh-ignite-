"""Faithful reimplementation of the v1 retrieval path.

v1 is the measurement baseline, so it has to be reproducible for as long as the
comparison is quoted. The original lives at the repository root (ingest.py,
retriever.py) and still runs; this module reproduces its ALGORITHM inside the
v2 environment so both systems can be scored by one harness, in one process,
against one gold set.

What is reproduced exactly:

  extraction   pdfplumber page text, concatenated with newlines, no cleaning
  chunking     text.split() then 800-word windows with 100-word overlap
  embedding    all-MiniLM-L6-v2, L2-normalised
  scoring      cosine similarity (a dot product, given normalised vectors)
  selection    top-k, no reranking, no filtering

What differs, and why it is acceptable:

  runtime      ONNX Runtime via fastembed, instead of PyTorch via
               sentence-transformers. Same weights, same architecture, same
               tokeniser. Float arithmetic differs in the last few decimal
               places, which cannot change a top-k ordering that is decided by
               gaps orders of magnitude larger. The alternative -- a 2.5 GB
               torch install whose only job is to reproduce a baseline -- buys
               no additional fidelity.

Nothing here is used by the v2 pipeline. It exists to be beaten.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pdfplumber

# Model weights and ONNX artefacts are cached OUTSIDE the project directory.
# The repository lives in a OneDrive-synced folder, and letting a sync client
# chew through hundreds of megabytes of model files on every run is a
# self-inflicted wound.
DEFAULT_CACHE = Path.home() / ".cache" / "fastembed"

V1_WORDS_PER_CHUNK = 800
V1_OVERLAP = 100
V1_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_text_v1(pdf_path: Path) -> str:
    """v1's ingest.extract_text: every page, joined with newlines, uncleaned.

    Front matter, tables of contents and the running header are all included,
    exactly as v1 saw them. Cleaning any of it here would make the baseline
    better than the thing actually being compared against.
    """
    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def chunk_text_v1(
    text: str,
    words_per_chunk: int = V1_WORDS_PER_CHUNK,
    overlap: int = V1_OVERLAP,
) -> list[str]:
    """v1's ingest.chunk_text, unchanged.

    The `text.split()` is the consequential line: it discards every newline,
    heading and clause number before chunking can use them, which is why no
    structure-aware strategy was possible downstream.
    """
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + words_per_chunk]))
        i += words_per_chunk - overlap
    return chunks


@dataclass
class V1Result:
    chunk_index: int
    score: float


class V1Retriever:
    """Dense-only, brute-force cosine over in-memory embeddings.

    Mirrors v1's retriever.Retriever, including the parts that are weaknesses:
    no persistence (re-embedded every run), no metadata, no filtering, and a
    linear scan over the whole matrix for every query.
    """

    def __init__(
        self,
        chunks: list[str],
        model_name: str = V1_MODEL,
        cache_dir: Path | None = None,
    ) -> None:
        from fastembed import TextEmbedding

        self.chunks = chunks
        self.model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir or DEFAULT_CACHE),
        )
        self.embeddings = self._encode(chunks)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return self._normalise(vectors)

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise, matching v1's normalize_embeddings=True.

        With unit vectors a dot product IS cosine similarity, which is the
        shortcut v1 relied on.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def measure_embedding_window(self, sample: str, max_words: int = 800) -> int:
        """How many words of `sample` the encoder actually reads.

        all-MiniLM-L6-v2 accepts 256 word-pieces. Dense regulatory English runs
        well over two pieces per word, so a great deal less than 800 words
        survives -- everything past the limit is truncated before the vector is
        produced, silently.

        Binary-searches the shortest prefix whose embedding is identical to the
        full text's. Identical, not merely similar: if a prefix reproduces the
        vector exactly, nothing after it contributed.

        This is the measurement behind v1's headline defect, so it is computed
        rather than asserted, and recorded with the baseline results.
        """
        words = sample.split()[:max_words]
        full = self._encode([" ".join(words)])[0]
        lo, hi = 1, len(words)
        while lo < hi:
            mid = (lo + hi) // 2
            prefix = self._encode([" ".join(words[:mid])])[0]
            if float(prefix @ full) > 0.99999:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def search(self, query: str, k: int = 3) -> list[V1Result]:
        vector = self._encode([query])[0]
        scores = self.embeddings @ vector
        # Descending by score; ties broken by chunk order so runs are
        # reproducible rather than dependent on sort implementation.
        order = np.argsort(-scores, kind="stable")[:k]
        return [V1Result(int(i), float(scores[i])) for i in order]
