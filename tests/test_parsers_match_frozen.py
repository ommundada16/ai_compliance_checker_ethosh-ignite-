"""The parsers in src/ must still reproduce the frozen gold artefacts exactly.

The gold set is frozen, but the parser that produced it is live code that the
v2 pipeline also uses. Any change to it -- a cleaning tweak, a boundary rule,
a normalisation fix -- silently redefines what a clause or a passage IS, while
eval_data/*.jsonl keeps the old definition and the labels keep pointing at the
old IDs. Nothing fails; the numbers just quietly stop meaning what they did.

These tests re-parse the PDFs and compare against the committed artefacts. They
are the alarm for exactly that. Marked slow because they open both PDFs.

If one fails, the question is not "how do I make the test pass" but "did I mean
to change the definition of a clause?" If yes, rebuild the artefacts, re-check
the gold labels still resolve, and commit both together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDELINE = PROJECT_ROOT / "data" / "guideline.pdf"
SOURCE = PROJECT_ROOT / "data" / "source_file.pdf"
CLAUSES = PROJECT_ROOT / "eval_data" / "clauses.jsonl"
PASSAGES = PROJECT_ROOT / "eval_data" / "passages.jsonl"

pytestmark = pytest.mark.slow


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_mdr_parser_reproduces_the_frozen_clause_corpus() -> None:
    if not GUIDELINE.exists() or not CLAUSES.exists():
        pytest.skip("guideline.pdf or clauses.jsonl missing")
    from dataclasses import asdict

    from auditor.parsing.mdr import FIRST_BODY_PAGE, build

    rebuilt = [asdict(c) for c in build(GUIDELINE, FIRST_BODY_PAGE) if c.n_words >= 8]
    frozen = _load(CLAUSES)

    assert len(rebuilt) == len(frozen), (
        f"parser now yields {len(rebuilt)} clauses, corpus has {len(frozen)}"
    )
    for got, want in zip(rebuilt, frozen, strict=True):
        assert got["clause_id"] == want["clause_id"], "clause IDs drifted"
        assert got["text_sha1"] == want["text_sha1"], (
            f"{want['clause_id']} text changed -- the gold labels still point at the old text"
        )


def test_cer_parser_reproduces_the_frozen_passages() -> None:
    if not SOURCE.exists() or not PASSAGES.exists():
        pytest.skip("source_file.pdf or passages.jsonl missing")
    from dataclasses import asdict

    from auditor.parsing.cer import build

    rebuilt = [asdict(p) for p in build(SOURCE)]
    frozen = _load(PASSAGES)

    assert len(rebuilt) == len(frozen), (
        f"parser now yields {len(rebuilt)} passages, artefact has {len(frozen)}"
    )
    for got, want in zip(rebuilt, frozen, strict=True):
        assert got["passage_id"] == want["passage_id"], "passage IDs drifted"
        assert got["text_sha1"] == want["text_sha1"], (
            f"{want['passage_id']} text changed -- gold labels point at the old text"
        )


def test_parsers_are_importable_from_the_package() -> None:
    """The v2 pipeline imports these; tools/ only wraps them.

    Guards against the parsing logic drifting back into tools/, which would
    reintroduce two definitions of a clause.
    """
    from auditor.parsing import cer, mdr

    assert hasattr(mdr, "build")
    assert hasattr(cer, "build")
    for module in (mdr, cer):
        assert not hasattr(module, "PROJECT_ROOT"), (
            f"{module.__name__} resolves project paths; that belongs in the CLI, "
            "and resolved wrongly once already after the move into src/"
        )
