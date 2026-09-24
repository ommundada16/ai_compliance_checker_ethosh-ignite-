"""Audit-quality metrics: did it find the right violations, and only those.

Retrieval metrics bound what the system CAN do. These measure what it actually
reports, which is the thing the product is judged on.

Four numbers, because they fail independently and a system can be good at one
while being useless:

  recall              of the violations known to be present, how many were
                      found. A lower bound, since the expectation set is not
                      exhaustive.
  false-positive rate findings raised on passages where none was expected.
                      The metric most audit evaluations omit, and the one that
                      decides whether a reviewer can trust the output -- a tool
                      that flags everything has perfect recall and no value.
  hallucination rate  findings whose quote could not be located in the source,
                      or whose clause was never retrieved. After guardrails
                      this should be zero BY CONSTRUCTION; measuring it anyway
                      is how a broken guardrail gets noticed.
  precision           of everything reported, how much was expected. Read with
                      care: the expectation set is not exhaustive, so a correct
                      finding outside it counts against precision here. It is
                      reported for completeness and the FP rate is the honest
                      version of the same question.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from auditor.evaluation.gold import is_within


@dataclass
class ExpectedFinding:
    passage_id: str
    clause_scope: str
    summary: str
    min_severity: str = "Low"


@dataclass
class PredictedFinding:
    passage_id: str
    clause_id: str
    severity: str
    quote: str
    grounded: bool = True
    clause_was_retrieved: bool = True


@dataclass
class AuditScores:
    expected_total: int = 0
    matched: int = 0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    clean_passages: int = 0
    clean_passages_with_findings: int = 0
    false_positive_rate: float = 0.0
    findings_on_clean: int = 0
    predicted_total: int = 0
    hallucinated: int = 0
    hallucination_rate: float = 0.0
    matched_detail: list[str] = field(default_factory=list)
    missed_detail: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "expected_total": self.expected_total,
            "matched": self.matched,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "predicted_total": self.predicted_total,
            "clean_passages": self.clean_passages,
            "clean_passages_with_findings": self.clean_passages_with_findings,
            "findings_on_clean": self.findings_on_clean,
            "false_positive_rate": round(self.false_positive_rate, 4),
            "hallucinated": self.hallucinated,
            "hallucination_rate": round(self.hallucination_rate, 4),
        }


def matches(predicted: PredictedFinding, expected: ExpectedFinding) -> bool:
    """Same passage, and a clause inside the expected scope.

    The quoted text is deliberately NOT required to match. Two reviewers can
    cite the same breach from different sentences of the same passage, and
    demanding string equality would score a correct finding as a miss.
    """
    if predicted.passage_id != expected.passage_id:
        return False
    return is_within(predicted.clause_id, expected.clause_scope)


def score_audit(
    predicted: Sequence[PredictedFinding],
    expected: Sequence[ExpectedFinding],
    clean_passages: Sequence[str],
) -> AuditScores:
    scores = AuditScores(
        expected_total=len(expected),
        predicted_total=len(predicted),
        clean_passages=len(clean_passages),
    )

    # --- recall over known violations ---
    unmatched = list(predicted)
    for want in expected:
        hit = next((p for p in unmatched if matches(p, want)), None)
        if hit is not None:
            # One prediction satisfies one expectation, so a single finding
            # cannot be credited against two different obligations.
            unmatched.remove(hit)
            scores.matched += 1
            scores.matched_detail.append(f"{want.passage_id} -> {want.clause_scope}")
        else:
            scores.missed_detail.append(f"{want.passage_id} -> {want.clause_scope}")

    scores.recall = scores.matched / len(expected) if expected else 0.0
    scores.precision = scores.matched / len(predicted) if predicted else 0.0
    if scores.recall + scores.precision > 0:
        scores.f1 = (
            2 * scores.recall * scores.precision / (scores.recall + scores.precision)
        )

    # --- false positives on passages that should be quiet ---
    clean = set(clean_passages)
    on_clean = [p for p in predicted if p.passage_id in clean]
    scores.findings_on_clean = len(on_clean)
    scores.clean_passages_with_findings = len({p.passage_id for p in on_clean})
    scores.false_positive_rate = (
        scores.clean_passages_with_findings / len(clean) if clean else 0.0
    )

    # --- hallucination ---
    scores.hallucinated = sum(
        1 for p in predicted if not p.grounded or not p.clause_was_retrieved
    )
    scores.hallucination_rate = (
        scores.hallucinated / len(predicted) if predicted else 0.0
    )

    return scores
