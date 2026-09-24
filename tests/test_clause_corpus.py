"""Invariants for the frozen clause corpus (eval_data/clauses.jsonl).

These are not "does the code run" tests. The corpus is the fixed reference
frame for every retrieval label in the gold set, so each assertion here guards
a specific way the corpus could silently rot and quietly invalidate every
recall number computed against it:

  * a missing Article  -> labels pointing at it become unmatchable
  * a duplicate ID     -> one label silently addresses two different texts
  * running-head leak  -> boilerplate pollutes the embedding of every clause
  * a recital creeping in -> non-binding text scored as an obligation
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS = PROJECT_ROOT / "eval_data" / "clauses.jsonl"


@pytest.fixture(scope="module")
def clauses() -> list[dict]:
    if not CORPUS.exists():
        pytest.skip(f"{CORPUS} not built; run tools/build_clause_corpus.py")
    with CORPUS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_every_article_present(clauses: list[dict]) -> None:
    """EU MDR 2017/745 has Articles 1..123 with no gaps."""
    found = {c["article"] for c in clauses if c["article"]}
    assert found == set(range(1, 124)), f"missing articles: {sorted(set(range(1, 124)) - found)}"


def test_every_annex_present(clauses: list[dict]) -> None:
    """Annexes I..XVII."""
    expected = {
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX",
        "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII",
    }
    found = {c["annex"] for c in clauses if c["annex"]}
    assert found == expected, f"missing annexes: {sorted(expected - found)}"


def test_clause_ids_unique(clauses: list[dict]) -> None:
    ids = [c["clause_id"] for c in clauses]
    dupes = {i for i in ids if ids.count(i) > 1} if len(ids) != len(set(ids)) else set()
    assert not dupes, f"duplicate clause_ids: {sorted(dupes)[:10]}"


def test_subpoints_nest_under_their_paragraph(clauses: list[dict]) -> None:
    """Article 61 has points (a)-(c) under paragraph 3 AND (a)-(b) under 6.

    A flat scheme would collide these into one ambiguous "Art.61.a".
    """
    ids = {c["clause_id"] for c in clauses}
    for expected in ("Art.61.3.a", "Art.61.3.b", "Art.61.3.c", "Art.61.6.a", "Art.61.6.b"):
        assert expected in ids, f"{expected} missing -- sub-point nesting regressed"
    assert "Art.61.a" not in ids, "flat sub-point ID reappeared"


def test_no_running_header_leaked(clauses: list[dict]) -> None:
    """The OJ running head appears on all 175 pages and must never reach a clause.

    Matched on the full header shape -- "5.5.2017 EN Official Journal of the
    European Union L 117/55" -- rather than on the phrase "Official Journal".
    Article 123(1) legitimately cites the Official Journal in its own text
    ("...following that of its publication in the Official Journal of the
    European Union"), so a phrase match flags real legislative content as
    contamination.
    """
    import re

    header = re.compile(r"EN\s+Official Journal of the European Union\s+L\s*\d+/\d+")
    offenders = [c["clause_id"] for c in clauses if header.search(c["text"])]
    assert not offenders, f"running header leaked into: {offenders[:5]}"


def test_article_123_keeps_its_legitimate_oj_reference(clauses: list[dict]) -> None:
    """Regression guard for the fix above.

    If header stripping is ever tightened to a bare "Official Journal" match,
    this clause loses its operative text and the test above starts passing for
    the wrong reason.
    """
    by_id = {c["clause_id"]: c for c in clauses}
    art123 = by_id.get("Art.123.1")
    assert art123 is not None, "Art.123.1 missing"
    assert "Official Journal of the European Union" in art123["text"]
    assert "enter into force" in art123["text"]


def test_recitals_excluded(clauses: list[dict]) -> None:
    """Recitals (pp. 1-12) are non-normative and must not be auditable clauses."""
    early = [c["clause_id"] for c in clauses if c["page_start"] < 13]
    assert not early, f"pre-body clauses present: {early[:5]}"


def test_every_clause_has_anchoring_metadata(clauses: list[dict]) -> None:
    """A clause must be locatable in the source document by a human reviewer."""
    for c in clauses:
        assert c["text"].strip(), f"{c['clause_id']} has empty text"
        assert c["page_start"] >= 13, f"{c['clause_id']} has bad page_start"
        assert c["page_end"] >= c["page_start"], f"{c['clause_id']} has inverted page range"
        assert c["path"], f"{c['clause_id']} has no human-readable path"
        assert c["article"] or c["annex"], f"{c['clause_id']} anchored to neither article nor annex"


def test_known_clause_content(clauses: list[dict]) -> None:
    """Spot-check a clause whose wording is central to this auditor's domain.

    Article 61(1) is the clinical-evaluation obligation -- the single most
    relevant clause for auditing a Clinical Evaluation Report.
    """
    by_id = {c["clause_id"]: c for c in clauses}
    art61 = by_id.get("Art.61.1")
    assert art61 is not None, "Art.61.1 missing"
    assert art61["article_title"] == "Clinical evaluation"
    assert art61["chapter"] == "VI"
    for phrase in ("clinical data", "benefit-risk", "Annex I"):
        assert phrase.lower() in art61["text"].lower(), f"Art.61.1 lost the phrase {phrase!r}"


def test_hyphen_rejoin_did_not_corrupt_words(clauses: list[dict]) -> None:
    """normalise() rejoins line-broken compounds; make sure it did not eat spaces
    between genuinely separate words."""
    by_id = {c["clause_id"]: c for c in clauses}
    art61 = by_id["Art.61.1"]
    assert "benefit-risk-ratio" in art61["text"] or "benefit-risk- ratio" not in art61["text"]
    assert "  " not in art61["text"], "double spaces survived normalisation"


def test_corpus_is_deterministic(clauses: list[dict]) -> None:
    """text_sha1 must match the text it claims to summarise.

    If these drift, the corpus was hand-edited, and a hand-edited frozen
    artefact is no longer frozen.
    """
    import hashlib

    for c in clauses:
        expect = hashlib.sha1(c["text"].encode("utf-8")).hexdigest()[:12]
        assert c["text_sha1"] == expect, f"{c['clause_id']} text/sha1 mismatch"


def test_corpus_size_sane(clauses: list[dict]) -> None:
    """Guard against a parser change that silently halves the corpus."""
    assert len(clauses) > 1000, f"corpus collapsed to {len(clauses)} clauses"
    words = sum(c["n_words"] for c in clauses)
    assert words > 70_000, f"corpus holds only {words} words; text is being dropped"
