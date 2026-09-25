"""Measure document-level reconciliation on FROZEN first-pass audit output.

Reconciliation is a post-filter: it only ever removes findings the first pass
already produced. So its effect can be measured by replaying it over a stored
audit run, which has two advantages over re-running the whole pipeline:

  * The generator's output is IDENTICAL before and after. The audit is not
    bit-deterministic (the same config scored FP-rate 0.53 and 0.73), so an
    end-to-end A/B mixes the effect being measured with sampling noise. Here
    the only thing that changes is the filter.
  * It costs a handful of small calls to the judge model instead of a full
    audit run, which matters: the Groq free tier caps gpt-oss-120b at 200,000
    tokens per day and one audit run costs roughly half of that.

Each stored run is an independent first-pass sample, so replaying over several
also shows how stable the effect is.

Usage:
    python tools/eval_reconcile_offline.py \
        eval_data/results/audit_eval.json eval_data/results/audit_cap250.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from audit_expectations import CLEAN_PASSAGES, EXPECTED_FINDINGS  # noqa: E402
from run_audit_eval import load_jsonl, to_predicted  # noqa: E402

from auditor.audit.reconcile import (  # noqa: E402
    CerIndex,
    is_absence_claim,
    reconcile_finding,
)
from auditor.audit.schema import AuditReport  # noqa: E402
from auditor.embedding import get_dense  # noqa: E402
from auditor.evaluation.audit_metrics import ExpectedFinding, score_audit  # noqa: E402
from auditor.resources import check_memory  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="audit result JSON files")
    ap.add_argument("--config", default="v2_full")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "eval_data" / "results" / "reconcile_offline.json")
    ap.add_argument("--memory-floor-gb", type=float, default=1.5)
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    from auditor.llm.providers import GroqProvider

    judge = GroqProvider(os.getenv("GROQ_API_KEY"),
                         os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b"), "low", seed=0)

    check_memory(args.memory_floor_gb, "starting")
    passages = load_jsonl(PROJECT_ROOT / "eval_data" / "passages.jsonl")
    clauses = {c["clause_id"]: c for c in
               load_jsonl(PROJECT_ROOT / "eval_data" / "clauses.jsonl")}
    by_id = {p["passage_id"]: p for p in passages}
    index = CerIndex(passages, get_dense())

    expected = [
        ExpectedFinding(passage_id=pid, clause_scope=e["clause_scope"],
                        summary=e["summary"], min_severity=e.get("min_severity", "Low"))
        for pid, entries in EXPECTED_FINDINGS.items() for e in entries
    ]

    out: dict[str, dict] = {}
    for path in args.runs:
        stored = json.loads(path.read_text(encoding="utf-8"))[args.config]
        report = AuditReport.model_validate(stored["report"])
        before = score_audit(to_predicted(report), expected, CLEAN_PASSAGES)

        decisions = []
        for audit in report.passages:
            passage = by_id[audit.passage_id]
            kept = []
            for finding in audit.findings:
                if not is_absence_claim(finding.violating_statement, finding.explanation):
                    kept.append(finding)
                    decisions.append({"passage": audit.passage_id, "clause": finding.clause_id,
                                      "absence_claim": False, "dropped": False})
                    continue
                check_memory(args.memory_floor_gb, f"reconciling {audit.passage_id}")
                outcome = reconcile_finding(
                    claim=finding.violating_statement,
                    clause_id=finding.clause_id,
                    clause_text=clauses.get(finding.clause_id, {}).get("text", ""),
                    section_title=passage.get("section_title", ""),
                    passage_id=audit.passage_id,
                    index=index, judge=judge,
                )
                decisions.append({
                    "passage": audit.passage_id, "clause": finding.clause_id,
                    "absence_claim": True, "dropped": outcome.addressed,
                    "detail": outcome.detail, "proof": outcome.quote,
                    "claim": finding.violating_statement,
                })
                if not outcome.addressed:
                    kept.append(finding)
            audit.findings = kept

        after = score_audit(to_predicted(report), expected, CLEAN_PASSAGES)
        b, a = before.as_dict(), after.as_dict()
        out[path.name] = {"before": b, "after": a, "decisions": decisions}
        dropped = [d for d in decisions if d["dropped"]]
        print(f"\n{path.name}: {b['predicted_total']} findings -> {a['predicted_total']} "
              f"({len(dropped)} dropped)")
        print(f"  recall {b['recall']:.2f} -> {a['recall']:.2f}   "
              f"FP-rate {b['false_positive_rate']:.2f} -> {a['false_positive_rate']:.2f}   "
              f"precision {b['precision']:.3f} -> {a['precision']:.3f}")
        for d in dropped:
            print(f"  dropped {d['passage']} / {d['clause']}: {d['claim'][:70]}")
            print(f"     proof: {d['proof'][:110]}")

    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
