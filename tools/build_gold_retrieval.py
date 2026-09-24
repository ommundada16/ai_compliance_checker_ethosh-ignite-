"""Phase 1c -- expand the expert section map into passage-level gold labels.

Output: eval_data/gold_retrieval.jsonl

One record per passage, carrying the graded relevance judgements that
retrieval metrics are computed against:

    {"passage_id": "CER.4.5.3#2",
     "section": "4.5.3",
     "labels": {"Annex.I.1": 2, "Annex.I.8": 2, "Art.61.1": 2, "Annex.I.2": 1},
     "provenance": {"Annex.I.1": ["expert_map"], ...},
     "n_primary": 3}

Labels are inherited from the section, not assigned per passage. That is a
deliberate simplification with a real justification: a CER section is written
as one argument against one set of obligations, and splitting section 4.5.3
into four passages does not change which clauses govern it. It does mean a
passage that happens to discuss only part of its section's topic will carry a
slightly generous label set -- recorded here as a known property rather than
discovered later as a surprise.

Provenance is tracked per clause so that a later LLM cross-check pass can add
candidates without erasing where anything came from. A label is never silently
overwritten; conflicting grades are surfaced.

Usage:
    python tools/build_gold_retrieval.py
    python tools/build_gold_retrieval.py --merge eval_data/llm_proposals.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdr_section_map import GRADE_PRIMARY, SECTION_MAP, graded_labels  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build(passages: list[dict], corpus_ids: set[str]) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    problems: list[str] = []

    for p in passages:
        section = p["section"]
        labels = graded_labels(section)
        if not labels:
            problems.append(f"{p['passage_id']}: section {section} has no expert mapping")
            continue

        unknown = sorted(set(labels) - corpus_ids)
        if unknown:
            problems.append(f"{p['passage_id']}: clause ids absent from corpus: {unknown}")

        records.append(
            {
                "passage_id": p["passage_id"],
                "section": section,
                "section_title": p["section_title"],
                "labels": dict(sorted(labels.items())),
                "provenance": {cid: ["expert_map"] for cid in sorted(labels)},
                "n_primary": sum(1 for g in labels.values() if g == GRADE_PRIMARY),
                "n_total": len(labels),
            }
        )
    return records, problems


def merge_proposals(records: list[dict], proposals: list[dict], corpus_ids: set[str]) -> list[str]:
    """Fold LLM-proposed clause ids into existing records.

    Agreements are recorded by appending to provenance. Clauses only the model
    suggested enter at grade 1 (secondary) and are tagged so they can be
    filtered out entirely when reporting a pure expert-map baseline. The model
    never promotes anything to primary on its own -- that judgement stays with
    the expert map.
    """
    notes: list[str] = []
    by_id = {r["passage_id"]: r for r in records}

    for prop in proposals:
        rec = by_id.get(prop["passage_id"])
        if rec is None:
            notes.append(f"proposal for unknown passage {prop['passage_id']}")
            continue
        for cid in prop.get("clause_ids", []):
            if cid not in corpus_ids:
                notes.append(f"{prop['passage_id']}: model proposed non-existent clause {cid}")
                continue
            if cid in rec["labels"]:
                if "llm" not in rec["provenance"][cid]:
                    rec["provenance"][cid].append("llm")
            else:
                rec["labels"][cid] = 1
                rec["provenance"][cid] = ["llm"]
        rec["labels"] = dict(sorted(rec["labels"].items()))
        rec["provenance"] = dict(sorted(rec["provenance"].items()))
        rec["n_primary"] = sum(1 for g in rec["labels"].values() if g == GRADE_PRIMARY)
        rec["n_total"] = len(rec["labels"])
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passages", type=Path, default=PROJECT_ROOT / "eval_data" / "passages.jsonl")
    ap.add_argument("--clauses", type=Path, default=PROJECT_ROOT / "eval_data" / "clauses.jsonl")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval_data" / "gold_retrieval.jsonl")
    ap.add_argument("--merge", type=Path, default=None,
                    help="Optional LLM proposal file to fold in (tools/propose_labels_llm.py).")
    args = ap.parse_args()

    for path in (args.passages, args.clauses):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    passages = load_jsonl(args.passages)
    corpus_ids = {c["clause_id"] for c in load_jsonl(args.clauses)}

    records, problems = build(passages, corpus_ids)

    if args.merge:
        if not args.merge.exists():
            print(f"error: {args.merge} not found", file=sys.stderr)
            return 1
        problems += merge_proposals(records, load_jsonl(args.merge), corpus_ids)

    if problems:
        print("PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        # Unresolvable references are fatal: a gold set that points at clauses
        # which do not exist scores every retriever as wrong for free.
        if any("absent from corpus" in p or "no expert mapping" in p for p in problems):
            return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_primary = sum(r["n_primary"] for r in records)
    total_labels = sum(r["n_total"] for r in records)
    llm_only = sum(
        1 for r in records for cid, src in r["provenance"].items() if src == ["llm"]
    )
    both = sum(1 for r in records for cid, src in r["provenance"].items() if len(src) > 1)

    print(f"gold records written : {len(records)}  -> {args.out.relative_to(PROJECT_ROOT)}")
    print(f"  sections covered   : {len({r['section'] for r in records})} / {len(SECTION_MAP)}")
    print(f"  primary labels     : {total_primary}  (mean {total_primary / max(len(records), 1):.1f}/passage)")
    print(f"  total labels       : {total_labels}  (mean {total_labels / max(len(records), 1):.1f}/passage)")
    if args.merge:
        print(f"  expert+llm agreed  : {both}")
        print(f"  llm-only additions : {llm_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
