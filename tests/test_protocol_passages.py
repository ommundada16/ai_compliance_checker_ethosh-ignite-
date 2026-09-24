"""Invariants for the frozen protocol passages (eval_data/passages.jsonl).

Passages are the QUERIES in this system -- the corpus is the regulation and the
query is a chunk of the document under audit. Each assertion guards a way the
passage set could silently degrade and take every metric with it:

  * template boilerplate leaking -> the same 6 lines embedded into all 68
    passages, pulling every query vector toward a common centroid
  * undecoded bullet glyphs      -> corrupt text, and list items welded into
    run-on sentences
  * a raised word floor          -> terse but substantive claims (a nine-word
    contraindications statement) silently dropped from the test set
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PASSAGES = PROJECT_ROOT / "eval_data" / "passages.jsonl"

BOILERPLATE = (
    "BIORAD MEDISYS PVT. LTD.",
    "Document No. CER/DJS/61",
    "Revision. No 01",
    "Revision Date 26-12-2025",
    "PRODUCT- DOUBLE J STENT",
)


@pytest.fixture(scope="module")
def passages() -> list[dict]:
    if not PASSAGES.exists():
        pytest.skip(f"{PASSAGES} not built; run tools/build_protocol_passages.py")
    with PASSAGES.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_passage_ids_unique(passages: list[dict]) -> None:
    ids = [p["passage_id"] for p in passages]
    assert len(ids) == len(set(ids)), "duplicate passage_ids"


def test_no_template_boilerplate(passages: list[dict]) -> None:
    """Six template lines appear on all 91 pages and must never reach a passage."""
    for p in passages:
        for junk in BOILERPLATE:
            assert junk not in p["text"], f"{p['passage_id']} carries boilerplate {junk!r}"


def test_no_undecodable_glyphs(passages: list[dict]) -> None:
    """The CER bullets use a symbol font pdfplumber cannot map.

    Left unrepaired they surface as U+FFFD, corrupting the text and collapsing
    list items into a run-on sentence.
    """
    for p in passages:
        assert "�" not in p["text"], f"{p['passage_id']} has an undecoded glyph"


def test_bullets_survived_normalisation(passages: list[dict]) -> None:
    """The repair must preserve list structure, not merely delete the bad byte."""
    total = sum(p["text"].count("•") for p in passages)
    assert total > 20, f"only {total} bullets survived; list structure was flattened"


def test_terse_but_substantive_sections_kept(passages: list[dict]) -> None:
    """Regression guard on the word floor.

    2.11 is nine words -- "There are no known absolute contraindications for
    this device." -- and is exactly the sort of claim Annex I makes auditable.
    An earlier 40-word floor discarded it along with 2.9 and 2.10.
    """
    ids = {p["passage_id"] for p in passages}
    for pid in ("CER.2.9", "CER.2.10", "CER.2.11"):
        assert pid in ids, f"{pid} dropped by the word floor"


def test_pure_cross_references_excluded(passages: list[dict]) -> None:
    """2.16 is "Refer IFU Document No: BRP/IFU/DJS/01/03" -- a pointer, not a claim."""
    ids = {p["passage_id"] for p in passages}
    assert "CER.2.16" not in ids, "a content-free cross-reference became a passage"


def test_front_matter_excluded(passages: list[dict]) -> None:
    """Cover page, TOC and abbreviations (pp. 1-9) carry nothing auditable."""
    early = [p["passage_id"] for p in passages if p["page_start"] < 10]
    assert not early, f"front-matter passages present: {early[:5]}"


def test_every_passage_locatable(passages: list[dict]) -> None:
    for p in passages:
        assert p["text"].strip(), f"{p['passage_id']} is empty"
        assert p["section"], f"{p['passage_id']} has no section number"
        assert p["section_title"], f"{p['passage_id']} has no section title"
        assert p["page_end"] >= p["page_start"], f"{p['passage_id']} has an inverted page range"
        assert 1 <= p["part"] <= p["n_parts"], f"{p['passage_id']} has bad part numbering"


def test_word_counts_within_bounds(passages: list[dict]) -> None:
    for p in passages:
        assert p["n_words"] == len(p["text"].split()), f"{p['passage_id']} n_words is stale"
        assert p["n_words"] >= 8, f"{p['passage_id']} is below the floor"
        assert p["n_words"] <= 700, f"{p['passage_id']} is oversized ({p['n_words']}w)"


def test_sha1_matches_text(passages: list[dict]) -> None:
    """Detects hand-editing of a supposedly frozen artefact."""
    for p in passages:
        expect = hashlib.sha1(p["text"].encode("utf-8")).hexdigest()[:12]
        assert p["text_sha1"] == expect, f"{p['passage_id']} text/sha1 mismatch"


def test_key_audit_targets_present(passages: list[dict]) -> None:
    """Sections that carry the obligations this auditor exists to check."""
    sections = {p["section"] for p in passages}
    for section in ("2.11", "3.5", "3.7", "4.5.3"):
        assert section in sections, f"section {section} missing from the passage set"


def test_corpus_size_sane(passages: list[dict]) -> None:
    assert len(passages) >= 60, f"passage set collapsed to {len(passages)}"
    words = sum(p["n_words"] for p in passages)
    assert words > 14_000, f"only {words} words captured; text is being dropped"
