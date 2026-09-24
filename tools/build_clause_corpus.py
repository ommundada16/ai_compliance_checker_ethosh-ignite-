"""Freeze the EU MDR clause corpus as eval_data/clauses.jsonl.

Thin CLI over auditor.parsing.mdr. The parsing logic lives in src/ because
the v2 pipeline uses it too; this wrapper only freezes its output as the
gold-set artefact. See that module for why sharing the parser is not circular.

Usage:
    python tools/build_clause_corpus.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auditor.parsing.mdr import FIRST_BODY_PAGE, build  # noqa: E402


def rel(path: Path) -> str:
    """Display path, tolerant of outputs written outside the repo.

    Path.relative_to raises when the target is not under the base, which
    crashed a completed run at the final print line.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, default=PROJECT_ROOT / "data" / "guideline.pdf")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval_data" / "clauses.jsonl")
    ap.add_argument("--first-page", type=int, default=FIRST_BODY_PAGE)
    ap.add_argument("--min-words", type=int, default=8,
                    help="Drop fragments shorter than this; they are almost always "
                         "stray table cells or heading spill, not obligations.")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 1

    clauses = build(args.pdf, args.first_page)
    kept = [c for c in clauses if c.n_words >= args.min_words]
    dropped = len(clauses) - len(kept)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for c in kept:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    arts = sum(1 for c in kept if c.kind == "article_paragraph")
    anns = sum(1 for c in kept if c.kind == "annex_section")
    words = sum(c.n_words for c in kept)
    print(f"clauses written : {len(kept)}  -> {rel(args.out)}")
    print(f"  article paras : {arts}")
    print(f"  annex sections: {anns}")
    print(f"  dropped (<{args.min_words}w): {dropped}")
    print(f"  total words   : {words:,}")
    print(f"  mean words    : {words // max(len(kept), 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
