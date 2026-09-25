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
    v2_title        v2_full plus the section title in the retrieval query
                    -> isolates what a better query is worth in findings
    v2_reconcile    v2_title plus document-level reconciliation of "X is missing"
                    findings -> isolates what cross-section context is worth
    v2_context      v2_reconcile plus a prompt that tells the model it is reading
                    one section of a longer CER (development-set tuned; see
                    DOCUMENT_CONTEXT in audit/pipeline.py)

A single audit run is NOT a stable measurement: the identical v2_full config
scored FP-rate 0.533 and then 0.733 with temperature 0, because the hosted model
is not bit-deterministic. Use --repeats to report mean and range, and do not
claim a difference between configs whose ranges overlap.

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
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from audit_expectations import CLEAN_PASSAGES, EXPECTED_FINDINGS  # noqa: E402

from auditor.audit.guardrails import sanitise_passage  # noqa: E402
from auditor.audit.pipeline import (  # noqa: E402
    AuditConfig,
    AuditPipeline,
    retrieval_query,
)
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
from auditor.resources import check_memory, describe_plan  # noqa: E402
from auditor.retrieval.fusion import fuse_to_ids  # noqa: E402
from auditor.retrieval.qdrant_store import QdrantClauseStore, ScoredClause  # noqa: E402

SPAN_CACHE = PROJECT_ROOT / "eval_data" / ".span_cache.json"
RETRIEVAL_CACHE = PROJECT_ROOT / "eval_data" / ".retrieval_cache.json"


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


class TableRetriever:
    """Replays retrieval results computed up front.

    Retrieval is deterministic and does not depend on the LLM, so it is done
    once per configuration and the models that did it are released BEFORE any
    generation. On a 16 GB machine that is the difference between the run fitting
    beside a local LLM and being stopped by the memory guard; it also means the
    cross-encoder's ~14 s/query is paid once, not once per repeat.
    """

    def __init__(self, table: dict) -> None:
        self.table = table

    def __call__(self, query: str, k: int) -> list:
        return self.table[(query, k)]


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
    ap.add_argument("--qdrant-path", default="qdrant_local",
                    help="Embedded index directory. Pass '' to use a server instead. "
                         "Embedded runs in-process: no container, no Docker VM, "
                         "~2 GB less RAM on a 16 GB machine.")
    ap.add_argument("--provider", choices=["auto", "groq", "local"], default="auto",
                    help="groq keeps inference off this machine entirely, which is "
                         "both safer and faster than an 8B model that does not fit "
                         "in 4 GB of VRAM.")
    ap.add_argument("--local-model", default=None,
                    help="Ollama model for the generator (default: $OLLAMA_MODEL).")
    ap.add_argument("--judge", choices=["auto", "groq", "local"], default="auto",
                    help="auto: groq when --provider groq, else local. A local "
                         "generator with a Groq judge keeps the judge independent "
                         "of the generator and off this machine.")
    ap.add_argument("--memory-floor-gb", type=float, default=1.5,
                    help="Abort cleanly if free RAM drops below this.")
    ap.add_argument("--candidates", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--all-passages", action="store_true")
    ap.add_argument("--precompute-only", action="store_true",
                    help="Fill the retrieval cache and exit. Run this once, in its own "
                         "process, so the LLM runs never load the retrieval models.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Run each config this many times and report mean and range.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Groq sampling seed (best effort; -1 disables).")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "eval_data" / "results" / "audit_eval.json")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=True)

    from auditor.llm.providers import FailoverProvider, GroqProvider, OllamaProvider

    chain: list = []
    if args.provider in ("auto", "groq") and os.getenv("GROQ_API_KEY"):
        chain.append(GroqProvider(os.getenv("GROQ_API_KEY"),
                                  os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                                  os.getenv("GROQ_REASONING_EFFORT", "low"),
                                  seed=None if args.seed < 0 else args.seed))
    if args.provider in ("auto", "local") or not chain:
        chain.append(OllamaProvider(
            model=args.local_model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")))
    provider = FailoverProvider(chain)

    # The judge must differ from the GENERATOR, or it measures self-consistency
    # rather than correctness. With Groq generating, a local model is a genuinely
    # independent second reader AND costs nothing extra to run.
    use_groq_judge = args.judge == "groq" or (
        args.judge == "auto" and args.provider == "groq")
    judge = (
        OllamaProvider(model=os.getenv("JUDGE_MODEL", "llama3.1:8b"))
        if not use_groq_judge
        else GroqProvider(os.getenv("GROQ_API_KEY"),
                          os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b"),
                          "low", seed=None if args.seed < 0 else args.seed)
    )

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

    configs = {
        "v1_retrieval": AuditConfig(top_k=args.top_k, enable_judge=False,
                                    min_confidence=0.0, min_grounding=0.82),
        "v2_no_guards": AuditConfig(top_k=args.top_k, enable_judge=False,
                                    min_confidence=0.0, min_grounding=0.0,
                                    enable_sanitisation=False),
        "v2_full": AuditConfig(top_k=args.top_k, enable_judge=True,
                               min_confidence=0.35, min_grounding=0.82),
        "v2_title": AuditConfig(top_k=args.top_k, enable_judge=True,
                                min_confidence=0.35, min_grounding=0.82,
                                title_in_query=True),
        "v2_reconcile": AuditConfig(top_k=args.top_k, enable_judge=True,
                                    min_confidence=0.35, min_grounding=0.82,
                                    title_in_query=True, enable_reconcile=True),
        "v2_context": AuditConfig(top_k=args.top_k, enable_judge=True,
                                  min_confidence=0.35, min_grounding=0.82,
                                  title_in_query=True, enable_reconcile=True,
                                  document_context=True),
    }

    # --- retrieval cache ------------------------------------------------------
    # Retrieval is deterministic and independent of the LLM, so it is computed
    # once per (config, passage) and stored. LLM runs then replay it without
    # loading the embedder, reranker or index at all -- which is what lets a
    # local LLM fit beside the pipeline on a 16 GB machine, and means the
    # cross-encoder's ~14 s/query is not paid again on every repeat.
    def text_for(passage: dict, cfg: AuditConfig) -> str:
        body = (sanitise_passage(passage["text"])[0]
                if cfg.enable_sanitisation else passage["text"])
        return retrieval_query(body, passage.get("section_title", ""),
                               cfg.title_in_query)

    def cache_key(family: str, text: str, k: int) -> str:
        return f"{family}|{hashlib.sha1(text.encode('utf-8')).hexdigest()}|{k}|{args.candidates}"

    cache: dict[str, list] = (
        json.loads(RETRIEVAL_CACHE.read_text(encoding="utf-8"))
        if RETRIEVAL_CACHE.exists() else {}
    )
    wanted: dict[str, tuple[str, str, int]] = {}
    for name in args.configs:
        cfg = configs[name]
        family = "v1" if name == "v1_retrieval" else "v2"
        for passage in passages:
            text = text_for(passage, cfg)
            wanted[cache_key(family, text, cfg.top_k)] = (family, text, cfg.top_k)
    missing = {key: v for key, v in wanted.items() if key not in cache}
    missing_families = {v[0] for v in missing.values()}

    local_llm = any(p.name == "ollama" for p in chain)
    # Measured, not guessed. A Groq-only run with the embedded index peaked at
    # ~4 GB, not the 2 GB an earlier version of this estimate claimed: the
    # v1_retrieval config loads MiniLM AND v1's 145 chunk embeddings on top of
    # bge-base, the reranker and the in-process Qdrant index. An estimate that
    # under-reports is worse than none, because the headroom verdict beside it
    # is then wrong in the dangerous direction.
    estimated = (
        (2.0 if "v2" in missing_families else 0.3)   # bge-base + reranker, only if computing
        + (1.5 if "v1" in missing_families else 0.0)         # MiniLM + v1 chunk matrix
        + (0.6 if args.qdrant_path and "v2" in missing_families else 0.0)  # embedded index
        + (3.5 if local_llm else 0.0)                # 8B model spilling out of 4 GB VRAM
        + (0.0 if args.qdrant_path else 2.0)         # Docker Desktop VM
    )
    print(describe_plan(
        [
            f"LLM        : {provider.describe()}"
            + ("   <-- runs on THIS machine" if local_llm else "   (remote; ~0 GB local)"),
            f"retrieval  : {len(missing)} of {len(wanted)} results to compute"
            + (" (models load)" if missing else " (all cached; no retrieval models)"),
            "vector db  : " + ("embedded, in-process (no Docker)" if args.qdrant_path
                               else "server at " + args.qdrant_url + " (Docker VM ~2 GB)"),
            f"passages   : {len(passages)} x {len(args.configs)} configs",
        ],
        estimated,
    ))
    check_memory(args.memory_floor_gb, "starting")
    print()
    print(f"provider  : {provider.describe()}")
    print(f"judge     : {judge.describe()}")
    print(f"passages  : {len(passages)} ({len(EXPECTED_FINDINGS)} with expectations, "
          f"{len(CLEAN_PASSAGES)} clean)")
    print(f"expected  : {len(expected)} findings")
    print()

    live: dict[str, object] = {}
    if "v1" in missing_families:
        print("building v1 retriever ...")
        text = extract_text_v1(PROJECT_ROOT / "data" / "guideline.pdf")
        v1_chunks = chunk_text_v1(text)
        cached_spans = json.loads(SPAN_CACHE.read_text(encoding="utf-8"))
        spans = [ClauseSpan(**s) for s in cached_spans["spans"]]
        live["v1"] = V1ClauseRetriever(V1Retriever(v1_chunks), spans, clause_by_id)
    if "v2" in missing_families:
        store = QdrantClauseStore(
            args.collection, get_dense(), get_sparse(),
            url=args.qdrant_url,
            path=str(PROJECT_ROOT / args.qdrant_path) if args.qdrant_path else None,
        )
        from auditor.retrieval.rerank import CrossEncoderReranker

        live["v2"] = build_v2_retriever(store, args.candidates, CrossEncoderReranker())
    for done, (key, (family, text, k)) in enumerate(missing.items(), 1):
        cache[key] = [asdict(c) for c in live[family](text, k)]
        if done % 10 == 0 or done == len(missing):
            print(f"  retrieved {done}/{len(missing)}")
    if missing:
        RETRIEVAL_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    live.clear()
    if args.precompute_only:
        print(f"retrieval cache complete: {rel(RETRIEVAL_CACHE)}")
        return 0

    retrievers: dict[str, object] = {}
    for name in args.configs:
        cfg = configs[name]
        family = "v1" if name == "v1_retrieval" else "v2"
        retrievers[name] = TableRetriever({
            (text_for(p, cfg), cfg.top_k): [
                ScoredClause(**d)
                for d in cache[cache_key(family, text_for(p, cfg), cfg.top_k)]
            ]
            for p in passages
        })

    cer_index = None
    if {"v2_reconcile", "v2_context"} & set(args.configs):
        from auditor.audit.reconcile import CerIndex

        # Indexed over ALL passages, not just the evaluated subset: the answer
        # to "is X missing?" can be in any section of the document.
        cer_index = CerIndex(all_passages, get_dense())

    from auditor.audit.schema import AuditReport, readiness_score

    def run_once(name: str) -> dict:
        started = time.time()
        pipeline = AuditPipeline(
            provider,
            retrievers[name],
            configs[name],
            judge=judge if configs[name].enable_judge else None,
            cer_index=cer_index if configs[name].enable_reconcile else None,
        )
        # Guard per passage, not just at the start: memory pressure builds as
        # models warm up and caches fill, and the point is to exit cleanly
        # while the machine is still responsive.
        audited = []
        for passage in passages:
            check_memory(args.memory_floor_gb, f"passage {passage['passage_id']}")
            audited.append(pipeline.audit_passage(passage))

        report = AuditReport(document_name="Clinical Evaluation Report",
                             passages=audited)
        report.readiness_score = readiness_score(report.findings)
        scores = score_audit(to_predicted(report), expected, CLEAN_PASSAGES)
        errors = sum(1 for p in report.passages if p.error and "neutralised" not in p.error)
        run = {
            "scores": scores.as_dict(),
            "readiness_score": report.readiness_score,
            "findings": len(report.findings),
            "dropped": report.drop_counts(),
            "categories": report.category_counts(),
            "errors": errors,
            "judge_errors": pipeline.judge_errors,
            "elapsed_s": round(time.time() - started, 1),
            "matched": scores.matched_detail,
            "missed": scores.missed_detail,
            "report": report.model_dump(mode="json"),
        }
        print(f"  findings {len(report.findings):>3} | recall {scores.recall:.2f} "
              f"({scores.matched}/{scores.expected_total}) | "
              f"FP-rate {scores.false_positive_rate:.2f} "
              f"| halluc {scores.hallucination_rate:.2f} | {run['elapsed_s']}s")
        if report.drop_counts():
            print(f"  dropped: {report.drop_counts()}")
        if pipeline.judge_errors or errors:
            print(f"  !! CONTAMINATED: {pipeline.judge_errors} judge/reconciler call(s) "
                  f"failed open and {errors} passage(s) errored -- the numbers above "
                  f"include findings that skipped a check. Do not report this run.")
        return run

    results: dict[str, dict] = {}
    for name in args.configs:
        print(f"--- {name} ---")
        runs = []
        for rep in range(args.repeats):
            if args.repeats > 1:
                print(f" run {rep + 1}/{args.repeats}")
            runs.append(run_once(name))
        entry = dict(runs[0])          # first run, full detail; keeps old readers working
        entry["repeats"] = [
            {"scores": r["scores"], "findings": r["findings"],
             "dropped": r["dropped"], "elapsed_s": r["elapsed_s"],
             "matched": r["matched"]}
            for r in runs
        ]
        entry["summary"] = {
            key: {"mean": round(sum(r["scores"][key] for r in runs) / len(runs), 4),
                  "min": min(r["scores"][key] for r in runs),
                  "max": max(r["scores"][key] for r in runs)}
            for key in ("recall", "precision", "f1",
                        "false_positive_rate", "hallucination_rate")
        }
        entry["contaminated_runs"] = sum(
            1 for r in runs if r["judge_errors"] or r["errors"])
        results[name] = entry
        print()
        # Persist after every config: a long run that dies on config 5 should
        # not take configs 1-4 with it.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 86)
    print("AUDIT QUALITY".center(86))
    print("=" * 86)
    print(f"{'system':<15} {'found':>6} {'recall':>8} {'prec':>7} {'F1':>7} "
          f"{'FP-rate':>9} {'halluc':>8} {'clean+':>7}")
    print("-" * 86)
    for name in args.configs:
        m = results[name]["summary"]
        print(f"{name:<15} {results[name]['findings']:>6} {m['recall']['mean']:>8.3f} "
              f"{m['precision']['mean']:>7.3f} {m['f1']['mean']:>7.3f} "
              f"{m['false_positive_rate']['mean']:>9.3f} "
              f"{m['hallucination_rate']['mean']:>8.3f} "
              f"{results[name]['scores']['clean_passages_with_findings']:>7}")
        if args.repeats > 1:
            fp, rc = m["false_positive_rate"], m["recall"]
            print(f"{'':<15}  n={args.repeats}  recall {rc['min']:.2f}-{rc['max']:.2f}"
                  f"   FP-rate {fp['min']:.2f}-{fp['max']:.2f}")
    print("-" * 86)
    if args.repeats > 1:
        print(f"means over {args.repeats} runs; 'found' and 'clean+' are from run 1")
    print("recall  = of violations known to be present (a lower bound)")
    print("FP-rate = share of clean passages that got at least one finding")
    print(f"written to {rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
