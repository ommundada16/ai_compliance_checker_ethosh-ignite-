"""RRF fusion and gold-scope resolution, against hand-computed values.

Both decide what counts as a correct retrieval, so a defect here shows up as a
confidently wrong comparison rather than as a failure.
"""

from __future__ import annotations

import pytest

from auditor.evaluation.gold import expand_labels, is_within, scope_recall
from auditor.retrieval.fusion import fuse_to_ids, reciprocal_rank_fusion

TOL = 1e-12


# --- RRF ------------------------------------------------------------------

def test_rrf_single_list_is_hand_computable() -> None:
    """k=60: rank 1 -> 1/61, rank 2 -> 1/62."""
    got = dict(reciprocal_rank_fusion([["A", "B"]], k=60))
    assert abs(got["A"] - 1 / 61) < TOL
    assert abs(got["B"] - 1 / 62) < TOL


def test_rrf_sums_across_lists() -> None:
    """A is rank 1 in both lists: 1/61 + 1/61."""
    got = dict(reciprocal_rank_fusion([["A", "B"], ["A", "C"]], k=60))
    assert abs(got["A"] - 2 / 61) < TOL
    assert abs(got["B"] - 1 / 62) < TOL
    assert abs(got["C"] - 1 / 62) < TOL


def test_rrf_prefers_agreement_over_one_strong_list() -> None:
    """A document ranked 2nd by BOTH arms beats one ranked 1st by only one.

    2/62 = 0.03226 > 1/61 = 0.01639. This is the entire point of fusion: it
    rewards cross-arm agreement rather than a single confident opinion.
    """
    ranked = fuse_to_ids([["X", "A"], ["Y", "A"]], limit=3)
    assert ranked[0] == "A"


def test_rrf_k_damps_the_head_of_each_list() -> None:
    """With k=60 the gap between rank 1 and rank 2 is ~1.6%, not 50%.

    A small k would let one list dominate the fusion by itself.
    """
    scores = dict(reciprocal_rank_fusion([["A", "B"]], k=60))
    ratio = scores["A"] / scores["B"]
    assert 1.0 < ratio < 1.02

    small = dict(reciprocal_rank_fusion([["A", "B"]], k=1))
    assert small["A"] / small["B"] == pytest.approx(1.5)


def test_rrf_ignores_duplicates_within_one_list() -> None:
    """One arm must not vote twice for the same document."""
    got = dict(reciprocal_rank_fusion([["A", "A", "B"]], k=60))
    assert abs(got["A"] - 1 / 61) < TOL
    assert abs(got["B"] - 1 / 63) < TOL  # B is still at rank 3 in the raw list


def test_rrf_uses_rank_not_score() -> None:
    """Cosine lives in [-1,1]; BM25 is unbounded. Fusion must not care.

    Both lists here have identical ordering, so the result is identical
    regardless of what magnitudes the arms would have reported.
    """
    a = fuse_to_ids([["A", "B", "C"], ["A", "B", "C"]], limit=3)
    assert a == ["A", "B", "C"]


def test_rrf_weights() -> None:
    got = dict(reciprocal_rank_fusion([["A"], ["B"]], k=60, weights=[2.0, 1.0]))
    assert abs(got["A"] - 2 / 61) < TOL
    assert abs(got["B"] - 1 / 61) < TOL


def test_rrf_ties_break_deterministically() -> None:
    first = fuse_to_ids([["B"], ["A"]], limit=2)
    second = fuse_to_ids([["B"], ["A"]], limit=2)
    assert first == second == ["A", "B"]


def test_rrf_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["A"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["A"], ["B"]], weights=[1.0])


# --- scope resolution -----------------------------------------------------

def test_is_within_exact_and_descendant() -> None:
    assert is_within("Annex.VIII", "Annex.VIII")
    assert is_within("Annex.VIII.5", "Annex.VIII")
    assert is_within("Annex.VIII.7.1", "Annex.VIII")


def test_is_within_rejects_a_sibling_sharing_a_prefix() -> None:
    """Art.610 must not fall under Art.61 just because the string starts the same."""
    assert not is_within("Art.610", "Art.61")
    assert not is_within("Annex.VIIIa", "Annex.VIII")


def test_is_within_does_not_fold_hash_siblings_into_their_base() -> None:
    """Annex.I.13.a#1 is a SECOND unnumbered sub-paragraph, a sibling of
    Annex.I.13.a -- not a child of it."""
    assert not is_within("Annex.I.13.a#1", "Annex.I.13.a")
    assert is_within("Annex.I.13.a#1", "Annex.I.13")


def test_narrow_labels_stay_narrow() -> None:
    """The fix must not quietly widen a leaf label."""
    corpus = ["Art.61.1", "Art.61.2", "Art.61.3", "Art.61.3.a"]
    assert set(expand_labels({"Art.61.1": 2}, corpus)) == {"Art.61.1"}


def test_container_labels_expand() -> None:
    corpus = ["Annex.VIII", "Annex.VIII.1", "Annex.VIII.5", "Art.51.1"]
    got = expand_labels({"Annex.VIII": 2}, corpus)
    assert set(got) == {"Annex.VIII", "Annex.VIII.1", "Annex.VIII.5"}
    assert all(v == 2 for v in got.values())


def test_overlapping_scopes_take_the_higher_grade() -> None:
    corpus = ["Art.61.3", "Art.61.3.a"]
    got = expand_labels({"Art.61.3": 1, "Art.61.3.a": 2}, corpus)
    assert got["Art.61.3.a"] == 2
    assert got["Art.61.3"] == 1


def test_scope_recall_counts_provisions_not_clauses() -> None:
    """One hit inside a scope satisfies that scope.

    Without this, a scope holding 59 clauses would dominate clause-level recall
    by sheer size, and a retriever that found one clause from each of two
    scopes would score worse than one that dumped 59 from a single scope and
    missed the other entirely.
    """
    scopes = {"Annex.VIII", "Art.51.1"}
    assert scope_recall(["Annex.VIII.5"], scopes, k=5) == pytest.approx(0.5)
    assert scope_recall(["Annex.VIII.5", "Art.51.1"], scopes, k=5) == pytest.approx(1.0)


def test_scope_recall_respects_k() -> None:
    scopes = {"Art.61.1"}
    assert scope_recall(["X", "Y", "Art.61.1"], scopes, k=2) == 0.0
    assert scope_recall(["X", "Y", "Art.61.1"], scopes, k=3) == pytest.approx(1.0)


def test_scope_recall_dedupes_before_truncating() -> None:
    scopes = {"Art.61.1"}
    assert scope_recall(["X", "X", "Art.61.1"], scopes, k=2) == pytest.approx(1.0)


def test_scope_recall_empty_scopes_is_zero() -> None:
    assert scope_recall(["Art.61.1"], set(), k=5) == 0.0
