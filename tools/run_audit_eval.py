"""Phase 7 -- run the audit end to end and score the findings.

Produces eval_data/results/audit_eval.json.

Three configurations, each differing from the last in ONE thing, so a change in
findings is attributable:

    v1_retrieval    v2 prompt and schema, but context from V1's retriever
                    -> isolates what retrieval quality alone is worth
    v2_no_guards    v2 retrieval, guardrails off except schema validation
                    -> shows what the model produces unchecked
    v2_full         v2 retrieval plus grounding, citation check, abstention
                    -> the shipped system

Holding the prompt and the output contract constant across all three is
deliberate. v1 pasted raw 800-word chunks with no clause identifiers, so its
findings could never be verified against a clause at all; comparing that
directly would measure the prompt rather than the retrieval. Instead v1's
chunks are mapped to the clauses they contain, which gives v1 the best possible
version of its own context and keeps the comparison about retrieval.

Runs on the evaluation subset (the passages with authored expectations plus the
clean set) rather than all 75, because those are the only passages the metrics
read and a full pass costs an hour of local inference for no extra signal.

Usage:
    python tools/run_audit_eval.py
    python tools/run_audit_eval.py --configs v2_full --all-passages
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from audit_expectations import CLEAN_PASSAGES, EXPECTED_FINDINGS  # noqa: E402

from auditor.audit.pipeline import AuditConfig, AuditPipeline  # noqa: E402
from auditor.baselines.coverage import (  # noqa: E402
    ClauseSpan,
    clauses_in_chunk,
)
from auditor.baselines.v1_retriever import (  # noqa: E402
    V1_OVERLAP,
    V1_WORDS_PER_CHUNK,
    V1Retriever,
    chunk_text_v1,
    extract_text_v1,
)
from auditor.embedding import get_dense, get_sparse  # noqa: E402
from auditor.evaluation.audit_metrics import (  # noqa: E402
    ExpectedFinding,
    PredictedFinding,
    score_audit,
)
from auditor.retrieval.fusion import fuse_to_ids  # noqa: E402
from auditor.retrieval.qdrant_store import QdrantClauseStore, ScoredClause  # noqa: E402

SPAN_CACHE = PROJECT_ROOT / "eval_data" / ".span_cache.json"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@dataclass
class V1ClauseRetriever:
    """v1's chunk retrieval, surfaced as clauses.

    v1 handed the LLM raw 800-word chunks with no identifiers, so its findings
    could never be tied to a clause. Mapping its chunks to the clauses they
    contain gives v1 the best possible version of its own context and keeps the
    comparison about RETRIEVAL rather than about prompt format.
    """

    retriever: V1Retriever
    spans: list[ClauseSpan]
    clause_by_id: dict[str, dict]
    chunks_per_query: int = 3

    def __call__(self, query: str, k: int) -> list[ScoredClause]:
        hits = self.retriever.search(query, k=self.chunks_per_query)
        out: list[ScoredClause] = []
        seen: set[str] = set()
        for hit in hits:
            for cid in clauses_in_chunk(
                hit.chunk_index, self.spans, V1_WORDS_PER_CHUNK, V1_OVERLAP
            ):
                if cid in seen or cid not in self.clause_by_id:
                    continue
                seen.add(cid)
                clause = self.clause_by_id[cid]
                out.append(
                    ScoredClause(
                        clause_id=cid,
                        score=hit.score,
                        text=clause["text"],
                        path=clause["path"],
                        page_start=clause["page_start"],
                        n_words=clause["n_words"],
                    )
                )
                if len(out) >= k:
                    return out
        return out


def build_v2_retriever(store: QdrantClauseStore, candidates: int, reranker):
    clause_text: dict[str, str] = {}

    def retrieve(query: str, k: int) -> list[ScoredClause]:
        pool = max(k, candidates)
        dense = store.search_dense(query, limit=pool)
        sparse = store.search_sparse(query, limit=pool)
        by_id = {c.clause_id: c for c in (*dense, *sparse)}
        fused = fuse_to_ids(
            [[c.clause_id for c in dense], [c.clause_id for c in sparse]], limit=pool
        )
        if reranker is not None and fused:
            for cid in fused:
                clause_text.setdefault(cid, by_id[cid].text)
            order = reranker.rerank(query, [clause_text[c] for c in fused], top_k=k)
            fused = [fused[item.index] for item in order]
        return [by_id[c] for c in fused[:k]]

    return retrieve


def to_predicted(report) -> list[PredictedFinding]:
    out: list[PredictedFinding] = []
    for passage in report.passages:
        retrieved = set(passage.retrieved_clauses)
        for finding in passage.findings:
            out.append(
                PredictedFinding(
                    passage_id=finding.passage_id,
                    clause_id=finding.clause_id,
                    severity=finding.severity.value,
                    quote=finding.source_quote,
                    grounded=finding.grounding_score > 0,
                    clause_was_retrieved=finding.clause_id in retrieved,
                )
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+",
                    default=["v1_retrieval", "v2_no_guards", "v2_full"])
    ap.add_argument("--collection", default="mdr_clauses_v2")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--candidates", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--all-passages", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "eval_data" / "results" / "audit_eval.json")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=True)

    from auditor.llm.providers import FailoverProvider, GroqProvider, OllamaProvider

    chain: list = []
    if os.getenv("GROQ_API_KEY"):
        chain.append(GroqProvider(os.getenv("GROQ_API_KEY"),
                                  os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                                  os.getenv("GROQ_REASONING_EFFORT", "low")))
    chain.append(OllamaProvider(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b")))
    provider = FailoverProvider(chain)
    # The judge must differ from the generator, or it measures self-consistency.
    judge = OllamaProvider(model=os.getenv("JUDGE_MODEL", "llama3.1:8b"))

    clauses = load_jsonl(PROJECT_ROOT / "eval_data" / "clauses.jsonl")
    clause_by_id = {c["clause_id"]: c for c in clauses}
    all_passages = load_jsonl(PROJECT_ROOT / "eval_data" / "passages.jsonl")

    wanted = set(EXPECTED_FINDINGS) | set(CLEAN_PASSAGES)
    passages = all_passages if args.all_passages else [
        p for p in all_passages if p["passage_id"] in wanted
    ]

    expected = [
        ExpectedFinding(passage_id=pid, clause_scope=e["clause_scope"],
                        summary=e["summary"], min_severity=e.get("min_severity", "Low"))
        for pid, entries in EXPECTED_FINDINGS.items()
        for e in entries
    ]

    print(f"provider  : {provider.describe()}")
    print(f"judge     : {judge.describe()}")
    print(f"passages  : {len(passages)} ({len(EXPECTED_FINDINGS)} with expectations, "
          f"{len(CLEAN_PASSAGES)} clean)")
    print(f"expected  : {len(expected)} findings")
    print()

    retrievers: dict[str, object] = {}

    if "v1_retrieval" in args.configs:
        print("building v1 retriever ...")
        text = extract_text_v1(PROJECT_ROOT / "data" / "guideline.pdf")
        v1_chunks = chunk_text_v1(text)
        cached = json.loads(SPAN_CACHE.read_text(encoding="utf-8"))
        spans = [ClauseSpan(**s) for s in cached["spans"]]
        retrievers["v1_retrieval"] = V1ClauseRetriever(
            V1Retriever(v1_chunks), spans, clause_by_id
        )

    if any(c.startswith("v2") for c in args.configs):
        store = QdrantClauseStore(args.collection, get_dense(), get_sparse(),
                                  url=args.qdrant_url)
        from auditor.retrieval.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        v2 = build_v2_retriever(store, args.candidates, reranker)
        retrievers["v2_no_guards"] = v2
        retrievers["v2_full"] = v2

    configs = {
        "v1_retrieval": AuditConfig(top_k=args.top_k, enable_judge=False,
                                    min_confidence=0.0, min_grounding=0.82),
        "v2_no_guards": AuditConfig(top_k=args.top_k, enable_judge=False,
                                    min_confidence=0.0, min_grounding=0.0,
                                    enable_sanitisation=False),
        "v2_full": AuditConfig(top_k=args.top_k, enable_judge=True,
                               min_confidence=0.35, min_grounding=0.82),
    }

    results: dict[str, dict] = {}
    for name in args.configs:
        print(f"--- {name} ---")
        started = time.time()
        pipeline = AuditPipeline(
            provider,
            retrievers[name],
            configs[name],
            judge=judge if configs[name].enable_judge else None,
        )
        report = pipeline.audit_document(passages, "Clinical Evaluation Report")
        predicted = to_predicted(report)
        scores = score_audit(predicted, expected, CLEAN_PASSAGES)

        errors = sum(1 for p in report.passages if p.error and "neutralised" not in p.error)
        results[name] = {
            "scores": scores.as_dict(),
            "readiness_score": report.readiness_score,
            "findings": len(report.findings),
            "dropped": report.drop_counts(),
            "categories": report.category_counts(),
            "errors": errors,
            "elapsed_s": round(time.time() - started, 1),
            "matched": scores.matched_detail,
            "missed": scores.missed_detail,
            "report": report.model_dump(mode="json"),
        }
        s = scores
        print(f"  findings {len(report.findings):>3} | recall {s.recall:.2f} "
              f"({s.matched}/{s.expected_total}) | FP-rate {s.false_positive_rate:.2f} "
              f"| halluc {s.hallucination_rate:.2f} | {results[name]['elapsed_s']}s")
        if report.drop_counts():
            print(f"  dropped: {report.drop_counts()}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 86)
    print("AUDIT QUALITY".center(86))
    print("=" * 86)
    print(f"{'system':<15} {'found':>6} {'recall':>8} {'prec':>7} {'F1':>7} "
          f"{'FP-rate':>9} {'halluc':>8} {'clean+':>7}")
    print("-" * 86)
    for name in args.configs:
        s = results[name]["scores"]
        print(f"{name:<15} {results[name]['findings']:>6} {s['recall']:>8.3f} "
              f"{s['precision']:>7.3f} {s['f1']:>7.3f} {s['false_positive_rate']:>9.3f} "
              f"{s['hallucination_rate']:>8.3f} {s['clean_passages_with_findings']:>7}")
    print("-" * 86)
    print("recall  = of violations known to be present (a lower bound)")
    print("FP-rate = share of clean passages that got at least one finding")
    print(f"written to {rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
