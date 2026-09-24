"""The shape of an audit finding.

Deliberately stricter than v1's schema, because the v1 version allowed findings
that could not be checked:

  * `category` was a free string with the allowed values listed only in the
    prompt, so a model that invented one produced a finding nobody could
    aggregate
  * `source_quote` was optional, so a finding could assert a violation with
    nothing to point at
  * `guideline_clause` was free text, so "Annex XIV" and "Annex 14" and "the
    clinical evaluation annex" were three different clauses as far as any
    downstream code could tell

Here the enums are enforced by the type system, the quote is mandatory, and the
clause is a corpus ID that either resolves or does not. A finding that cannot be
located in the source text and tied to a real clause is not a finding.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class Category(str, Enum):
    CLINICAL_EVALUATION = "Clinical Evaluation"
    RISK_MANAGEMENT = "Risk Management"
    POST_MARKET_SURVEILLANCE = "Post Market Surveillance"
    VERIFICATION_VALIDATION = "Verification & Validation"
    EQUIVALENCE = "Equivalence"
    SAFETY = "Safety"
    REGULATORY_DOCUMENTATION = "Regulatory Documentation"
    CLINICAL_INVESTIGATION = "Clinical Investigation"
    OTHER = "Other"


class DropReason(str, Enum):
    """Why a guardrail rejected a finding.

    Recorded rather than discarded. The distribution of these is a measurement
    in its own right -- it says which failure mode the model actually has, and
    whether a guardrail is earning its keep or just cutting recall.
    """

    UNGROUNDED_QUOTE = "quote not found in the source passage"
    UNKNOWN_CLAUSE = "cited clause is not in the retrieved context"
    LOW_CONFIDENCE = "confidence below the abstention threshold"
    SCHEMA_INVALID = "did not satisfy the finding schema"
    EMPTY_EXPLANATION = "explanation or correction was empty"
    JUDGE_REJECTED = "second model did not agree the finding is supported"


class RawFinding(BaseModel):
    """What the model is asked to produce, before any guardrail runs."""

    violating_statement: str = Field(min_length=1)
    source_quote: str = Field(min_length=1, description="Verbatim span from the passage")
    clause_id: str = Field(min_length=1, description="A clause ID from the provided context")
    category: Category
    severity: Severity
    explanation: str = Field(min_length=1)
    suggested_correction: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("source_quote", "violating_statement", "explanation",
                     "suggested_correction", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("clause_id", mode="before")
    @classmethod
    def _clean_clause(cls, value: object) -> object:
        """Models like to wrap the ID in brackets, echoing the context format."""
        if isinstance(value, str):
            return value.strip().strip("[]").strip()
        return value


class Finding(RawFinding):
    """A finding that survived the guardrails, with its evidence resolved."""

    passage_id: str
    page: int
    quote_start: int = Field(ge=0, description="Char offset into the passage text")
    quote_end: int = Field(ge=0)
    clause_path: str = ""
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0)
    judged: bool = False

    @property
    def span(self) -> tuple[int, int]:
        return self.quote_start, self.quote_end


class DroppedFinding(BaseModel):
    """A rejected finding, kept so the guardrails can be measured."""

    reason: DropReason
    detail: str = ""
    raw: dict


class PassageAudit(BaseModel):
    passage_id: str
    section: str
    page: int
    findings: list[Finding] = Field(default_factory=list)
    dropped: list[DroppedFinding] = Field(default_factory=list)
    retrieved_clauses: list[str] = Field(default_factory=list)
    llm_provider: str = ""
    llm_model: str = ""
    latency_s: float = 0.0
    error: str | None = None


class AuditReport(BaseModel):
    document_name: str
    passages: list[PassageAudit] = Field(default_factory=list)
    readiness_score: float = 100.0
    config: dict = Field(default_factory=dict)

    @property
    def findings(self) -> list[Finding]:
        return [f for p in self.passages for f in p.findings]

    @property
    def dropped(self) -> list[DroppedFinding]:
        return [d for p in self.passages for d in p.dropped]

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.category.value] = counts.get(finding.category.value, 0) + 1
        return counts

    def drop_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for dropped in self.dropped:
            counts[dropped.reason.value] = counts.get(dropped.reason.value, 0) + 1
        return counts


SEVERITY_WEIGHT = {
    Severity.CRITICAL: 20,
    Severity.HIGH: 15,
    Severity.MEDIUM: 10,
    Severity.LOW: 5,
}


def readiness_score(findings: list[Finding]) -> float:
    """100 minus a penalty, taking the worst severity per category.

    Same rule as v1, kept deliberately: the score is reported side by side with
    v1's, and changing how it is computed would make that comparison
    meaningless. Worst-per-category rather than a sum, so twenty Low findings
    in one category cannot outweigh a single Critical one in another.
    """
    worst: dict[Category, int] = {}
    for finding in findings:
        weight = SEVERITY_WEIGHT[finding.severity]
        if worst.get(finding.category, 0) < weight:
            worst[finding.category] = weight
    return float(max(0, 100 - sum(worst.values())))
