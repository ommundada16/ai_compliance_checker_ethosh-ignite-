"""Qdrant-backed clause index.

Replaces v1's in-memory numpy matrix, which was rebuilt from scratch on every
run and supported nothing but a brute-force scan. This index is persisted,
carries payload metadata, and holds two vector kinds per clause so dense and
sparse retrieval can be run separately or fused.

Named vectors rather than two collections: one clause is one point with one
payload, so the two arms can never disagree about what a clause is, and a
metadata filter written once applies to both.

Point IDs are a deterministic hash of the clause_id. Qdrant requires integer or
UUID ids, but re-indexing must be idempotent -- an incrementing counter would
make every rebuild create duplicate points under new ids while the old ones
lingered.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from auditor.embedding import DenseEmbedder, SparseEmbedder, SparseVector

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


def point_id(clause_id: str) -> int:
    """Stable 63-bit id derived from the clause id, so re-indexing overwrites
    rather than duplicating."""
    digest = hashlib.sha1(clause_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


@dataclass(frozen=True)
class ScoredClause:
    clause_id: str
    score: float
    text: str
    path: str
    page_start: int
    n_words: int

    @property
    def as_context(self) -> str:
        """How the clause is presented to the LLM.

        The path is included deliberately: "Annex XIV > Part A > (3)" tells the
        model what it is reading, and a clause quoted without its location is
        one the auditor cannot cite in a finding.
        """
        return f"[{self.clause_id}] {self.path}\n{self.text}"


class QdrantClauseStore:
    def __init__(
        self,
        collection: str,
        dense: DenseEmbedder,
        sparse: SparseEmbedder | None = None,
        url: str = "http://localhost:6333",
        api_key: str = "",
        timeout: int = 60,
    ) -> None:
        self.collection = collection
        self.dense = dense
        self.sparse = sparse
        self.client = QdrantClient(url=url, api_key=api_key or None, timeout=timeout)

    # -- lifecycle ---------------------------------------------------------

    def recreate(self, m: int = 16, ef_construct: int = 128) -> None:
        """Drop and rebuild the collection.

        HNSW parameters are stated explicitly rather than left to defaults so
        the index configuration travels with the results that were measured on
        it. `m` is the graph's out-degree -- higher means better recall and a
        larger index; `ef_construct` is how hard the builder searches while
        linking. At 1320 points these barely matter, which is exactly why they
        should be written down: the number that will matter at 10^6 points is
        the one nobody recorded at 10^3.
        """
        vectors = {
            DENSE_VECTOR: models.VectorParams(
                size=self.dense.dim,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
            )
        }
        sparse_vectors = (
            {SPARSE_VECTOR: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )}
            if self.sparse
            else None
        )
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=vectors,
            sparse_vectors_config=sparse_vectors,
        )
        # Payload indexes for the fields a filter would actually use. Without
        # them Qdrant scans payloads linearly, which defeats the point of
        # having filters at all once the corpus grows.
        for field, schema in (
            ("clause_id", models.PayloadSchemaType.KEYWORD),
            ("article", models.PayloadSchemaType.INTEGER),
            ("annex", models.PayloadSchemaType.KEYWORD),
            ("kind", models.PayloadSchemaType.KEYWORD),
        ):
            self.client.create_payload_index(self.collection, field, schema)

    def index(self, clauses: Sequence[dict], batch_size: int = 128) -> int:
        """Embed and upsert. Returns the number of points written."""
        written = 0
        for start in range(0, len(clauses), batch_size):
            batch = clauses[start : start + batch_size]
            texts = [c["text"] for c in batch]
            dense_vectors = self.dense.embed_documents(texts)
            sparse_vectors: list[SparseVector] | None = (
                self.sparse.embed_documents(texts) if self.sparse else None
            )

            points = []
            for i, clause in enumerate(batch):
                vector: dict[str, object] = {DENSE_VECTOR: dense_vectors[i].tolist()}
                if sparse_vectors is not None:
                    sv = sparse_vectors[i]
                    vector[SPARSE_VECTOR] = models.SparseVector(
                        indices=sv.indices, values=sv.values
                    )
                points.append(
                    models.PointStruct(
                        id=point_id(clause["clause_id"]),
                        vector=vector,
                        payload={
                            "clause_id": clause["clause_id"],
                            "text": clause["text"],
                            "path": clause["path"],
                            "kind": clause["kind"],
                            "article": clause.get("article"),
                            "annex": clause.get("annex"),
                            "page_start": clause["page_start"],
                            "n_words": clause["n_words"],
                        },
                    )
                )
            self.client.upsert(self.collection, points=points, wait=True)
            written += len(points)
        return written

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # -- search ------------------------------------------------------------

    @staticmethod
    def _to_scored(points) -> list[ScoredClause]:
        out = []
        for p in points:
            payload = p.payload or {}
            out.append(
                ScoredClause(
                    clause_id=payload.get("clause_id", ""),
                    score=float(p.score),
                    text=payload.get("text", ""),
                    path=payload.get("path", ""),
                    page_start=int(payload.get("page_start", 0)),
                    n_words=int(payload.get("n_words", 0)),
                )
            )
        return out

    def search_dense(self, query: str, limit: int, query_filter=None) -> list[ScoredClause]:
        vector = self.dense.embed_query(query)
        result = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            using=DENSE_VECTOR,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return self._to_scored(result.points)

    def search_sparse(self, query: str, limit: int, query_filter=None) -> list[ScoredClause]:
        if not self.sparse:
            raise RuntimeError("store was built without a sparse embedder")
        sv = self.sparse.embed_query(query)
        if not sv.indices:
            # A query whose terms are all stopwords or all out-of-vocabulary
            # produces an empty sparse vector. Qdrant rejects that, so return
            # nothing and let fusion fall back to the dense arm.
            return []
        result = self.client.query_points(
            self.collection,
            query=models.SparseVector(indices=sv.indices, values=sv.values),
            using=SPARSE_VECTOR,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return self._to_scored(result.points)
