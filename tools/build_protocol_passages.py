"""Freeze the CER protocol passages as eval_data/passages.jsonl.

Thin CLI over auditor.parsing.cer. The parsing logic lives in src/ because
the v2 pipeline uses it too; this wrapper only freezes its output as the
gold-set artefact. See that module for why sharing the parser is not circular.

Usage:
    python tools/build_protocol_passages.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auditor.parsing.cer import build  # noqa: E402


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
    ap.add_argument("--pdf", type=Path, default=PROJECT_ROOT / "data" / "source_file.pdf")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval_data" / "passages.jsonl")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 1

    passages = build(args.pdf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for p in passages:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    words = sum(p.n_words for p in passages)
    sections = len({p.section for p in passages})
    print(f"passages written : {len(passages)}  -> {rel(args.out)}")
    print(f"  sections       : {sections}")
    print(f"  total words    : {words:,}")
    print(f"  mean words     : {words // max(len(passages), 1)}")
    print(f"  pages covered  : {min(p.page_start for p in passages)}-{max(p.page_end for p in passages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
