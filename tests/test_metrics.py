"""Metric correctness, checked against hand-computed values.

A wrong metric does not raise; it returns a believable number, and every
comparison built on it inherits the error silently. So each expected value here
is worked out by hand in the docstring rather than captured from a previous
run -- a snapshot test would happily freeze a bug in place.
"""

from __future__ import annotations

import math

import pytest

from auditor.evaluation.metrics import (
    aggregate,
    average_precision,
    context_precision_at_k,
    dcg_at_k,
    dedupe_preserving_order,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

TOL = 1e-9


# --- dedupe ---------------------------------------------------------------

def test_dedupe_keeps_best_rank() -> None:
    assert dedupe_preserving_order(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]


def test_dedupe_is_a_noop_when_unique() -> None:
    assert dedupe_preserving_order(["A", "B", "C"]) == ["A", "B", "C"]


# --- recall ---------------------------------------------------------------

def test_recall_at_k() -> None:
    """retrieved[:3] = [A, B, C]; relevant = {A, C, E}; hits {A, C} = 2 of 3."""
    got = recall_at_k(["A", "B", "C", "D"], {"A", "C", "E"}, k=3)
    assert abs(got - 2 / 3) < TOL


def test_recall_perfect() -> None:
    assert abs(recall_at_k(["A", "C"], {"A", "C"}, k=2) - 1.0) < TOL


def test_recall_zero_when_nothing_found() -> None:
    assert recall_at_k(["X", "Y"], {"A"}, k=2) == 0.0


def test_recall_dedupes_before_truncating() -> None:
    """[A, A, B] with k=2 must dedupe to [A, B] and score 1.0, not 0.5.

    Without dedupe the top-2 is [A, A], one distinct hit out of two relevant --
    a hybrid retriever returning the same clause from its dense and sparse arms
    would be punished for a duplicate rather than credited for the hit.
    """
    assert abs(recall_at_k(["A", "A", "B"], {"A", "B"}, k=2) - 1.0) < TOL


def test_recall_with_k_larger_than_result_list() -> None:
    assert abs(recall_at_k(["A"], {"A"}, k=5) - 1.0) < TOL


def test_recall_empty_relevant_is_zero_not_one() -> None:
    """A query with no primary labels is a gold-set defect.

    Scoring it 1.0 would bury the defect inside a healthy-looking average.
    """
    assert recall_at_k(["A"], set(), k=3) == 0.0


# --- precision ------------------------------------------------------------

def test_precision_at_k() -> None:
    """top-3 = [A, B, C]; two of the three are relevant."""
    got = precision_at_k(["A", "B", "C", "D"], {"A", "C", "E"}, k=3)
    assert abs(got - 2 / 3) < TOL


def test_precision_divides_by_k_not_by_results_returned() -> None:
    """One correct result out of a requested 5 is 0.2, not 1.0.

    Dividing by len(retrieved) would let a system game precision by returning a
    single confident guess.
    """
    assert abs(precision_at_k(["A"], {"A"}, k=5) - 0.2) < TOL


# --- MRR ------------------------------------------------------------------

def test_reciprocal_rank_first_hit_at_three() -> None:
    assert abs(reciprocal_rank(["X", "Y", "A"], {"A"}) - 1 / 3) < TOL


def test_reciprocal_rank_first_position() -> None:
    assert abs(reciprocal_rank(["A", "X"], {"A"}) - 1.0) < TOL


def test_reciprocal_rank_no_hit() -> None:
    assert reciprocal_rank(["X", "Y"], {"A"}) == 0.0


# --- MAP ------------------------------------------------------------------

def test_average_precision() -> None:
    """retrieved = [A, X, C]; relevant = {A, C, E}.

    hit at rank 1 -> precision 1/1 = 1.0
    hit at rank 3 -> precision 2/3
    E never retrieved.
    AP = (1.0 + 2/3) / 3 = 0.5555...

    The denominator is 3 (all primaries), not 2 (those found), so the missed
    clause is penalised rather than ignored.
    """
    got = average_precision(["A", "X", "C"], {"A", "C", "E"})
    assert abs(got - (1.0 + 2 / 3) / 3) < TOL


def test_average_precision_rewards_early_hits() -> None:
    early = average_precision(["A", "C", "X", "Y"], {"A", "C"})
    late = average_precision(["X", "Y", "A", "C"], {"A", "C"})
    assert early > late
    assert abs(early - 1.0) < TOL


# --- DCG / nDCG -----------------------------------------------------------

def test_dcg_exponential_gain() -> None:
    """grades {A:2, B:1}; retrieved [A, X, B], k=3.

    (2^2-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4)
      = 3/1 + 0 + 1/2 = 3.5
    """
    got = dcg_at_k(["A", "X", "B"], {"A": 2, "B": 1}, k=3)
    assert abs(got - 3.5) < TOL


def test_ndcg_hand_computed() -> None:
    """Same DCG of 3.5, against the ideal ranking [2, 2, 1].

    IDCG = 3/log2(2) + 3/log2(3) + 1/log2(4)
         = 3 + 1.8927892607143717 + 0.5 = 5.392789260714372
    nDCG = 3.5 / 5.392789260714372 = 0.6490147919365130
    """
    grades = {"A": 2, "B": 1, "C": 2}
    idcg = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    expected = 3.5 / idcg
    got = ndcg_at_k(["A", "X", "B"], grades, k=3)
    assert abs(got - expected) < TOL
    # Pinned literal as well, so a change to the formula above cannot make the
    # test agree with itself while drifting from the intended definition.
    assert abs(got - 0.6490147919365130) < 1e-12


def test_ndcg_perfect_ranking_is_one() -> None:
    grades = {"A": 2, "B": 1}
    assert abs(ndcg_at_k(["A", "B"], grades, k=2) - 1.0) < TOL


def test_ndcg_primary_outranks_secondary() -> None:
    """Putting the grade-2 clause first must beat putting the grade-1 first."""
    grades = {"A": 2, "B": 1}
    assert ndcg_at_k(["A", "B"], grades, k=2) > ndcg_at_k(["B", "A"], grades, k=2)


def test_ndcg_exponential_gain_is_not_linear() -> None:
    """A primary is worth 3x a secondary (2^2-1 vs 2^1-1), not 2x.

    This pins the gain formula: linear gain would make the ratio exactly 2.
    """
    top_primary = dcg_at_k(["A"], {"A": 2}, k=1)
    top_secondary = dcg_at_k(["B"], {"B": 1}, k=1)
    assert abs(top_primary / top_secondary - 3.0) < TOL


def test_ndcg_truncated_ideal_does_not_punish_unreachable_recall() -> None:
    """Five relevant clauses but k=2: returning the two best must score 1.0."""
    grades = {c: 2 for c in "ABCDE"}
    assert abs(ndcg_at_k(["A", "B"], grades, k=2) - 1.0) < TOL


def test_ndcg_zero_when_nothing_relevant_exists() -> None:
    assert ndcg_at_k(["A"], {}, k=3) == 0.0


# --- context precision ----------------------------------------------------

def test_context_precision_counts_both_grades() -> None:
    """[A(2), B(1), X(0)] at k=3 -> 2/3 carry some relevance."""
    got = context_precision_at_k(["A", "B", "X"], {"A": 2, "B": 1}, k=3)
    assert abs(got - 2 / 3) < TOL


def test_context_precision_differs_from_precision() -> None:
    """precision_at_k counts primary only; context_precision counts any grade.

    The gap between them is exactly the secondary context in the prompt.
    """
    grades = {"A": 2, "B": 1}
    retrieved = ["A", "B", "X"]
    assert precision_at_k(retrieved, {"A"}, k=3) == pytest.approx(1 / 3)
    assert context_precision_at_k(retrieved, grades, k=3) == pytest.approx(2 / 3)


# --- hit rate -------------------------------------------------------------

def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k(["X", "A"], {"A", "B"}, k=2) == 1.0
    assert hit_rate_at_k(["X", "Y"], {"A", "B"}, k=2) == 0.0


def test_hit_rate_respects_k() -> None:
    assert hit_rate_at_k(["X", "Y", "A"], {"A"}, k=2) == 0.0
    assert hit_rate_at_k(["X", "Y", "A"], {"A"}, k=3) == 1.0


# --- guards ---------------------------------------------------------------

@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_rejected(k: int) -> None:
    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, k=k)
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, k=k)


# --- aggregation ----------------------------------------------------------

def test_aggregate_mean_and_spread() -> None:
    got = aggregate([0.0, 0.5, 1.0])
    assert abs(got["mean"] - 0.5) < TOL
    assert got["n"] == 3
    assert got["min"] == 0.0
    assert got["max"] == 1.0
    # population std of [0, .5, 1] = sqrt(((.5)^2 + 0 + (.5)^2)/3) = sqrt(1/6)
    assert abs(got["std"] - math.sqrt(1 / 6)) < TOL


def test_aggregate_empty() -> None:
    assert aggregate([])["n"] == 0
