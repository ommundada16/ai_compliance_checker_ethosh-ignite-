"""Audit-quality scoring, against hand-computed values.

These decide whether the product works, so the arithmetic is checked rather
than trusted. The false-positive rate gets the most attention here because it
is the metric most audit evaluations omit -- a system that flags every passage
has perfect recall and is useless.
"""

from __future__ import annotations

import pytest

from auditor.evaluation.audit_metrics import (
    ExpectedFinding,
    PredictedFinding,
    matches,
    score_audit,
)

EXPECTED = [
    ExpectedFinding("CER.4.3.2.1", "Art.61.4", "no clinical investigation"),
    ExpectedFinding("CER.2.11", "Annex.I.23", "blanket contraindications claim"),
]
CLEAN = ["CER.2.1", "CER.2.2", "CER.2.3", "CER.2.4"]


def pred(passage: str, clause: str, **kw) -> PredictedFinding:
    return PredictedFinding(
        passage_id=passage, clause_id=clause,
        severity=kw.get("severity", "High"), quote=kw.get("quote", "q"),
        grounded=kw.get("grounded", True),
        clause_was_retrieved=kw.get("clause_was_retrieved", True),
    )


# --- matching -------------------------------------------------------------

def test_exact_clause_matches() -> None:
    assert matches(pred("CER.2.11", "Annex.I.23"), EXPECTED[1])


def test_clause_inside_the_expected_scope_matches() -> None:
    """Expecting Annex.I.23 must be satisfied by Annex.I.23.a.

    The expectation names a provision; the system cites a sub-point of it.
    Demanding exact equality would score a correct finding as a miss.
    """
    assert matches(pred("CER.2.11", "Annex.I.23.a"), EXPECTED[1])


def test_clause_outside_the_scope_does_not_match() -> None:
    assert not matches(pred("CER.2.11", "Annex.I.8"), EXPECTED[1])


def test_right_clause_on_the_wrong_passage_does_not_match() -> None:
    assert not matches(pred("CER.9.9", "Annex.I.23"), EXPECTED[1])


def test_quote_text_is_not_required_to_match() -> None:
    """Two reviewers can cite the same breach from different sentences."""
    assert matches(pred("CER.2.11", "Annex.I.23", quote="totally different wording"),
                   EXPECTED[1])


# --- recall and precision -------------------------------------------------

def test_perfect_run() -> None:
    got = score_audit(
        [pred("CER.4.3.2.1", "Art.61.4"), pred("CER.2.11", "Annex.I.23")],
        EXPECTED, CLEAN,
    )
    assert got.recall == 1.0
    assert got.precision == 1.0
    assert got.f1 == 1.0
    assert got.false_positive_rate == 0.0


def test_half_recall() -> None:
    got = score_audit([pred("CER.2.11", "Annex.I.23")], EXPECTED, CLEAN)
    assert got.recall == 0.5
    assert got.precision == 1.0
    assert got.f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert got.missed_detail == ["CER.4.3.2.1 -> Art.61.4"]


def test_one_prediction_cannot_satisfy_two_expectations() -> None:
    """Two expectations on the same passage with overlapping scopes must each
    need their own finding, or recall could exceed what was actually found."""
    expected = [
        ExpectedFinding("CER.2.11", "Annex.I.23", "first"),
        ExpectedFinding("CER.2.11", "Annex.I.23", "second"),
    ]
    got = score_audit([pred("CER.2.11", "Annex.I.23")], expected, CLEAN)
    assert got.matched == 1
    assert got.recall == 0.5


def test_finding_nothing_scores_zero_not_an_error() -> None:
    got = score_audit([], EXPECTED, CLEAN)
    assert got.recall == 0.0
    assert got.precision == 0.0
    assert got.f1 == 0.0


# --- false positives ------------------------------------------------------

def test_false_positive_rate_counts_passages_not_findings() -> None:
    """Three findings on one clean passage is ONE passage a reviewer has to
    dismiss, not three independent failures."""
    got = score_audit(
        [pred("CER.2.1", "Art.61.1"), pred("CER.2.1", "Annex.I.8"),
         pred("CER.2.1", "Art.83.1")],
        EXPECTED, CLEAN,
    )
    assert got.findings_on_clean == 3
    assert got.clean_passages_with_findings == 1
    assert got.false_positive_rate == 0.25  # 1 of 4 clean passages


def test_a_system_that_flags_everything_is_caught() -> None:
    """Perfect recall, and the FP rate exposes it as worthless."""
    noisy = [pred(p, "Art.61.1") for p in CLEAN]
    noisy += [pred("CER.4.3.2.1", "Art.61.4"), pred("CER.2.11", "Annex.I.23")]
    got = score_audit(noisy, EXPECTED, CLEAN)
    assert got.recall == 1.0
    assert got.false_positive_rate == 1.0
    assert got.precision < 0.4


def test_clean_passages_with_no_findings_score_zero_fp() -> None:
    got = score_audit([pred("CER.4.3.2.1", "Art.61.4")], EXPECTED, CLEAN)
    assert got.false_positive_rate == 0.0


# --- hallucination --------------------------------------------------------

def test_ungrounded_finding_counts_as_hallucinated() -> None:
    got = score_audit(
        [pred("CER.2.11", "Annex.I.23", grounded=False)], EXPECTED, CLEAN
    )
    assert got.hallucinated == 1
    assert got.hallucination_rate == 1.0


def test_unretrieved_clause_counts_as_hallucinated() -> None:
    """Citing a clause that was never in context is recall from training data,
    not from the regulation in front of the model."""
    got = score_audit(
        [pred("CER.2.11", "Annex.I.23", clause_was_retrieved=False)], EXPECTED, CLEAN
    )
    assert got.hallucinated == 1


def test_guardrails_should_drive_hallucination_to_zero() -> None:
    got = score_audit(
        [pred("CER.4.3.2.1", "Art.61.4"), pred("CER.2.11", "Annex.I.23")],
        EXPECTED, CLEAN,
    )
    assert got.hallucination_rate == 0.0


def test_empty_inputs_do_not_divide_by_zero() -> None:
    got = score_audit([], [], [])
    assert got.recall == 0.0
    assert got.false_positive_rate == 0.0
    assert got.hallucination_rate == 0.0
