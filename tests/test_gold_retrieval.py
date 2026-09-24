"""Invariants for the expert map and the gold retrieval labels.

The gold set decides what "correct retrieval" means, so a defect here does not
show up as a failing pipeline -- it shows up as a confidently wrong number. The
assertions below target the ways that happens:

  * a label pointing at a clause_id that does not exist -> every retriever is
    marked wrong for free, and Recall@k is capped below 1.0 for a reason
    nobody can see
  * a section with no primary labels -> that passage silently contributes 0 to
    recall no matter what any system returns
  * grades outside {1, 2} -> nDCG gain terms become meaningless
  * expert labels demoted by an LLM merge -> the human rubric quietly stops
    being the ground truth
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from mdr_section_map import (  # noqa: E402
    GRADE_PRIMARY,
    GRADE_SECONDARY,
    SECTION_MAP,
    graded_labels,
    referenced_clause_ids,
)

GOLD = PROJECT_ROOT / "eval_data" / "gold_retrieval.jsonl"
CLAUSES = PROJECT_ROOT / "eval_data" / "clauses.jsonl"
PASSAGES = PROJECT_ROOT / "eval_data" / "passages.jsonl"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        pytest.skip(f"{path} not built")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def gold() -> list[dict]:
    return _load(GOLD)


@pytest.fixture(scope="module")
def corpus_ids() -> set[str]:
    return {c["clause_id"] for c in _load(CLAUSES)}


@pytest.fixture(scope="module")
def passages() -> list[dict]:
    return _load(PASSAGES)


# --- the expert map itself ------------------------------------------------

def test_every_mapped_clause_exists(corpus_ids: set[str]) -> None:
    missing = sorted(referenced_clause_ids() - corpus_ids)
    assert not missing, f"expert map points at non-existent clauses: {missing}"


def test_every_section_has_a_primary(corpus_ids: set[str]) -> None:
    for section, entry in SECTION_MAP.items():
        assert entry["primary"], f"section {section} has no primary clause"
        assert entry["rationale"].strip(), f"section {section} has no rationale"


def test_primary_and_secondary_do_not_overlap() -> None:
    """A clause cannot be both grades; the primary grade must win unambiguously."""
    for section, entry in SECTION_MAP.items():
        overlap = set(entry["primary"]) & set(entry["secondary"])
        assert not overlap, f"section {section} lists {overlap} as both primary and secondary"


def test_graded_labels_prefers_primary() -> None:
    for section in SECTION_MAP:
        labels = graded_labels(section)
        for cid in SECTION_MAP[section]["primary"]:
            assert labels[cid] == GRADE_PRIMARY, f"{section}/{cid} lost its primary grade"


def test_map_covers_every_section_with_passages(passages: list[dict]) -> None:
    sections = {p["section"] for p in passages}
    unmapped = sorted(sections - set(SECTION_MAP))
    assert not unmapped, f"passages exist for unmapped sections: {unmapped}"


def test_map_has_no_dead_entries(passages: list[dict]) -> None:
    """A mapped section with no passages is dead weight that will rot unnoticed."""
    sections = {p["section"] for p in passages}
    dead = sorted(set(SECTION_MAP) - sections)
    assert not dead, f"mapped sections with no passages: {dead}"


# --- the generated gold file ----------------------------------------------

def test_one_record_per_passage(gold: list[dict], passages: list[dict]) -> None:
    assert len(gold) == len(passages), "gold labels and passages are out of sync"
    assert {g["passage_id"] for g in gold} == {p["passage_id"] for p in passages}


def test_all_gold_clauses_resolve(gold: list[dict], corpus_ids: set[str]) -> None:
    for g in gold:
        unknown = sorted(set(g["labels"]) - corpus_ids)
        assert not unknown, f"{g['passage_id']} labels unknown clauses: {unknown}"


def test_grades_are_valid(gold: list[dict]) -> None:
    for g in gold:
        for cid, grade in g["labels"].items():
            assert grade in (GRADE_SECONDARY, GRADE_PRIMARY), (
                f"{g['passage_id']}/{cid} has grade {grade}, expected 1 or 2"
            )


def test_every_passage_has_at_least_one_primary(gold: list[dict]) -> None:
    """Without a primary label a passage contributes 0 to recall regardless of
    what any retriever returns, silently depressing the metric."""
    for g in gold:
        assert g["n_primary"] >= 1, f"{g['passage_id']} has no primary label"


def test_counts_are_consistent(gold: list[dict]) -> None:
    for g in gold:
        assert g["n_total"] == len(g["labels"]), f"{g['passage_id']} n_total is stale"
        actual = sum(1 for v in g["labels"].values() if v == GRADE_PRIMARY)
        assert g["n_primary"] == actual, f"{g['passage_id']} n_primary is stale"


def test_provenance_covers_every_label(gold: list[dict]) -> None:
    """Every judgement must say where it came from, or the audit trail is broken."""
    for g in gold:
        assert set(g["provenance"]) == set(g["labels"]), (
            f"{g['passage_id']} provenance and labels disagree"
        )
        for cid, sources in g["provenance"].items():
            assert sources, f"{g['passage_id']}/{cid} has empty provenance"
            for src in sources:
                assert src in ("expert_map", "llm"), f"unknown provenance {src!r}"


def test_expert_labels_are_never_llm_only(gold: list[dict]) -> None:
    """A primary label must always have human authorship behind it.

    The LLM cross-check may add secondary candidates and may corroborate expert
    ones, but it must never be the sole source of a grade-2 judgement.
    """
    for g in gold:
        for cid, grade in g["labels"].items():
            if grade == GRADE_PRIMARY:
                assert "expert_map" in g["provenance"][cid], (
                    f"{g['passage_id']}/{cid} is primary but only the model proposed it"
                )


def test_label_density_is_plausible(gold: list[dict]) -> None:
    """Guards both directions: a near-empty gold set, and one so permissive that
    almost any retrieval looks correct."""
    total = sum(g["n_total"] for g in gold)
    mean = total / max(len(gold), 1)
    assert 1.5 <= mean <= 12, f"mean {mean:.1f} labels/passage is implausible"
    for g in gold:
        assert g["n_total"] <= 20, f"{g['passage_id']} has {g['n_total']} labels"
