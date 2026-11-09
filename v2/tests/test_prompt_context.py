"""The document-context framing is opt-in and must not change the default prompt.

Earlier audit results were produced without it; if the default prompt drifted,
those numbers would silently stop describing the shipped system.
"""

from __future__ import annotations

from auditor.audit.pipeline import DOCUMENT_CONTEXT, AuditConfig, build_prompt
from auditor.retrieval.qdrant_store import ScoredClause

CLAUSE = ScoredClause(clause_id="Art.61.1", score=1.0, text="text", path="p",
                      page_start=1, n_words=1)


def test_default_prompt_has_no_framing():
    prompt = build_prompt("body", "2.1 Title", [CLAUSE])
    assert "DOCUMENT CONTEXT" not in prompt
    assert prompt.startswith("Audit the following section")


def test_framing_is_prepended_when_enabled():
    prompt = build_prompt("body", "2.1 Title", [CLAUSE], document_context=True)
    assert prompt.startswith(DOCUMENT_CONTEXT)
    assert prompt.endswith(build_prompt("body", "2.1 Title", [CLAUSE])[-50:])


def test_framing_is_off_by_default_in_config():
    assert AuditConfig().document_context is False


def test_framing_names_no_gold_set_clause_or_section():
    # Guard against tuning the prompt to the answers: it may describe what a CER
    # is, never point at a specific clause or passage of the evaluation set.
    import re

    assert not re.search(r"Annex\s+[IVX]+|Art(icle|\.)\s*\d|CER\.\d", DOCUMENT_CONTEXT)


# --- the retrieval query -----------------------------------------------------

def test_query_is_the_bare_passage_by_default():
    from auditor.audit.pipeline import retrieval_query

    assert retrieval_query("body text", "2.1 Title", use_title=False) == "body text"


def test_title_is_prefixed_when_enabled():
    from auditor.audit.pipeline import retrieval_query

    assert retrieval_query("body text", "Intended purpose", True) == "Intended purpose. body text"


def test_a_missing_title_does_not_add_a_stray_prefix():
    from auditor.audit.pipeline import retrieval_query

    assert retrieval_query("body text", "", True) == "body text"


def test_pipeline_queries_the_retriever_with_the_title_query():
    from auditor.audit.pipeline import AuditPipeline
    from auditor.llm.base import JSONProvider, JSONResult

    seen = []

    class Empty(JSONProvider):
        def complete_json(self, system, prompt, max_tokens=1024, temperature=0.0):
            return JSONResult(data={"findings": []}, provider="x", model="y")

    def retrieve(query, k):
        seen.append(query)
        return []

    passage = {"passage_id": "P", "section": "1", "section_title": "Scope", "text": "body"}
    AuditPipeline(Empty(), retrieve, AuditConfig(title_in_query=True)).audit_passage(passage)
    AuditPipeline(Empty(), retrieve, AuditConfig()).audit_passage(passage)
    assert seen == ["Scope. body", "body"]
