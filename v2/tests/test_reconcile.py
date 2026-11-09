"""Document-level reconciliation of "X is missing" findings.

The failure it exists to fix: a section is audited alone, the model correctly
sees that the section lacks something, and reports it -- though the content is
in another section. These tests pin the safeguards that stop the fix from
turning into a way to lose real findings.
"""

from __future__ import annotations

import numpy as np

from auditor.audit.pipeline import AuditConfig, AuditPipeline
from auditor.audit.reconcile import CerIndex, is_absence_claim, reconcile_finding
from auditor.audit.schema import DropReason
from auditor.llm.base import JSONProvider, JSONResult
from auditor.retrieval.qdrant_store import ScoredClause


class ScriptedProvider(JSONProvider):
    """Returns queued replies in order; records every prompt it was sent."""

    name = "scripted"
    model = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete_json(self, system, prompt, max_tokens=1024, temperature=0.0):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return JSONResult(data=reply, provider=self.name, model=self.model)


class KeywordEmbedder:
    """Deterministic stand-in: one dimension per keyword."""

    VOCAB = ["purpose", "equivalence", "stent", "surveillance"]

    def _vec(self, text):
        v = np.array([text.lower().count(w) for w in self.VOCAB], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


PASSAGES = [
    {"passage_id": "P1", "section": "2.1", "section_title": "Identification",
     "text": "The device is a ureteral stent sold under the name Double J."},
    {"passage_id": "P2", "section": "4.1", "section_title": "Intended purpose",
     "text": "The intended purpose of the stent is to relieve ureteral obstruction "
             "in adult patients. The purpose is documented in the IFU."},
    {"passage_id": "P3", "section": "5.1", "section_title": "Surveillance",
     "text": "Post market surveillance is performed annually."},
]


def _index():
    return CerIndex(PASSAGES, KeywordEmbedder())


# --- what counts as an absence claim ----------------------------------------

def test_absence_claims_are_recognised():
    for text in (
        "The section does not specify the intended purpose.",
        "No evidence is provided that equivalence was demonstrated.",
        "The plan is missing.",
        "Fails to describe the clinical benefits.",
        "The description is incomplete.",
    ):
        assert is_absence_claim(text), text


def test_a_contradiction_is_not_an_absence_claim():
    # "Wrong", not "missing": another section cannot make it right.
    assert not is_absence_claim("The stated dwell time of 12 months contradicts Annex I.")


# --- the index --------------------------------------------------------------

def test_index_never_returns_the_section_being_audited():
    hits = _index().neighbours("intended purpose of the stent", k=5, exclude_id="P2")
    assert "P2" not in [h.passage_id for h in hits]
    assert len(hits) == 2


def test_index_ranks_the_relevant_section_first():
    hits = _index().neighbours("intended purpose of the stent", k=3, exclude_id="P1")
    assert hits[0].passage_id == "P2"


# --- the verdict ------------------------------------------------------------

def _reconcile(judge):
    return reconcile_finding(
        claim="The section does not specify the intended purpose.",
        clause_id="Annex.XIV.A.1.a", clause_text="the intended purpose ...",
        section_title="Identification", passage_id="P1",
        index=_index(), judge=judge,
    )


def test_verified_answer_is_accepted():
    judge = ScriptedProvider({
        "addressed": True, "passage_id": "P2",
        "quote": "The intended purpose of the stent is to relieve ureteral obstruction",
    })
    outcome = _reconcile(judge)
    assert outcome.addressed and outcome.passage_id == "P2"


def test_quote_that_is_not_in_the_named_section_is_rejected():
    judge = ScriptedProvider({
        "addressed": True, "passage_id": "P2",
        "quote": "the manufacturer guarantees lifelong safety of all patients",
    })
    outcome = _reconcile(judge)
    assert not outcome.addressed
    assert "not in the named section" in outcome.detail


def test_naming_a_section_it_was_never_shown_is_rejected():
    judge = ScriptedProvider({
        "addressed": True, "passage_id": "P99", "quote": "anything at all here ok",
    })
    assert not _reconcile(judge).addressed


def test_a_negative_verdict_keeps_the_finding():
    assert not _reconcile(ScriptedProvider({"addressed": False})).addressed


def test_judge_error_fails_open():
    assert not _reconcile(ScriptedProvider(RuntimeError("429"))).addressed


def test_only_the_other_sections_are_shown_to_the_model():
    judge = ScriptedProvider({"addressed": False})
    _reconcile(judge)
    prompt = judge.prompts[0]
    assert "[P2]" in prompt
    assert "[P1]" not in prompt        # the audited section is not "elsewhere"


# --- end to end through the pipeline -----------------------------------------

CLAUSE = ScoredClause(clause_id="Annex.XIV.A.1.a", score=1.0,
                      text="the intended purpose must be specified", path="Annex XIV",
                      page_start=1, n_words=6)

FINDING = {
    "violating_statement": "The section does not specify the intended purpose.",
    "source_quote": "The device is a ureteral stent sold under the name Double J.",
    "clause_id": "Annex.XIV.A.1.a", "category": "Clinical Evaluation",
    "severity": "High", "explanation": "Annex XIV requires the intended purpose.",
    "suggested_correction": "State the intended purpose.", "confidence": 0.9,
}


def _pipeline(judge_replies, *, reconcile):
    generator = ScriptedProvider({"findings": [FINDING]})
    judge = ScriptedProvider(*judge_replies)
    pipeline = AuditPipeline(
        generator, lambda q, k: [CLAUSE],
        AuditConfig(enable_judge=True, enable_reconcile=reconcile),
        judge=judge, cer_index=_index(),
    )
    return pipeline.audit_passage(PASSAGES[0]), judge


SUPPORTED = {"supported": True, "reason": "ok"}
ADDRESSED = {"addressed": True, "passage_id": "P2",
             "quote": "The intended purpose of the stent is to relieve ureteral obstruction"}


def test_pipeline_drops_a_finding_answered_elsewhere_and_says_why():
    audit, _ = _pipeline([SUPPORTED, ADDRESSED], reconcile=True)
    assert audit.findings == []
    assert [d.reason for d in audit.dropped] == [DropReason.ADDRESSED_ELSEWHERE]


def test_pipeline_keeps_the_finding_when_reconciliation_is_off():
    audit, judge = _pipeline([SUPPORTED], reconcile=False)
    assert len(audit.findings) == 1
    assert len(judge.prompts) == 1     # no reconciliation call was made


def test_pipeline_keeps_the_finding_when_nothing_answers_it():
    audit, _ = _pipeline([SUPPORTED, {"addressed": False}], reconcile=True)
    assert len(audit.findings) == 1


def test_pipeline_does_not_reconcile_a_non_absence_claim():
    contradiction = dict(FINDING, violating_statement="Dwell time contradicts Annex I.",
                         explanation="The stated value is wrong.")
    generator = ScriptedProvider({"findings": [contradiction]})
    judge = ScriptedProvider(SUPPORTED)
    audit = AuditPipeline(
        generator, lambda q, k: [CLAUSE],
        AuditConfig(enable_judge=True, enable_reconcile=True),
        judge=judge, cer_index=_index(),
    ).audit_passage(PASSAGES[0])
    assert len(audit.findings) == 1
    assert len(judge.prompts) == 1     # only the ordinary judge call


# --- fail-open must be visible -------------------------------------------------

def test_judge_outage_is_counted_not_silently_absorbed():
    generator = ScriptedProvider({"findings": [FINDING]})
    judge = ScriptedProvider(RuntimeError("429"), RuntimeError("429"))
    pipeline = AuditPipeline(
        generator, lambda q, k: [CLAUSE],
        AuditConfig(enable_judge=True, enable_reconcile=True),
        judge=judge, cer_index=_index(),
    )
    audit = pipeline.audit_passage(PASSAGES[0])
    assert len(audit.findings) == 1          # failed open, as designed...
    assert pipeline.judge_errors == 2        # ...but the failures are on the record
