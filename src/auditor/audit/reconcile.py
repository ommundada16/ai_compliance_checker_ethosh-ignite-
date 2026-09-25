"""Document-level reconciliation of "X is missing" findings.

The pipeline audits each passage in isolation, so a section that legitimately
defers a topic to another section is reported as missing that topic. Measured on
the audit gold set: 8 of 15 clean passages drew a finding, at confidence
0.88-0.97, so no confidence threshold can separate them. The information needed
to reject them is not in the passage; it is elsewhere in the same document.

This module supplies it. For a finding that asserts an ABSENCE, the rest of the
document is searched, and a second model is asked whether any other section
already satisfies the requirement.

Two safeguards keep this from becoming a way to lose real findings:

  * The verdict only counts if the model names a passage it was actually shown
    AND quotes text that is verifiably in that passage. A reconciler that could
    exonerate a finding on an unverifiable claim would be a hallucination
    channel pointing the wrong way.
  * An error anywhere fails OPEN: the finding is kept. Infrastructure trouble
    must not silently delete findings.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from auditor.audit.guardrails import locate_quote
from auditor.llm.base import JSONProvider

# A finding that says something is missing can be answered by pointing at where
# it is. A finding that says something is WRONG cannot, so those are never
# reconciled. Matched against the finding's own wording, deliberately broad on
# the absence side: a missed absence claim merely stays a finding (status quo),
# while a wrongly-matched non-absence claim is still protected by the
# verifiable-quote requirement below.
_ABSENCE = re.compile(
    r"\b("
    r"(?:does|do|did|is|are|has|have)\s+not|not\s+(?:been\s+)?\w*"
    r"(?:specif|includ|provid|describ|identif|present|demonstrat|defin|address|"
    r"mention|document|state|contain|detail|discuss|justif|report|reference)\w*|"
    r"lacks?|lacking|missing|absent|absence|omit\w*|fails?\s+to|"
    r"no\s+(?:evidence|mention|description|reference|information|data|details?|"
    r"discussion|justification|indication|explicit)|without\s+(?:any\s+)?"
    r"(?:evidence|description|reference|justification)|insufficient|incomplete"
    r")\b",
    re.IGNORECASE,
)

MAX_EXCERPT_WORDS = 300
EMBED_BATCH = 8
MAX_REQUIREMENT_WORDS = 150

RECONCILE_SYSTEM = (
    "You check whether a requirement flagged as missing from one section of a "
    "medical-device report is in fact satisfied in another section of the same "
    "report. You are strict: you answer yes only when an excerpt clearly "
    "contains the required content, and you quote it. You answer only with "
    "valid JSON."
)


def is_absence_claim(violating_statement: str, explanation: str = "") -> bool:
    """True when the finding asserts something is missing, unstated or unclear."""
    return bool(_ABSENCE.search(f"{violating_statement} {explanation}"))


@dataclass
class Excerpt:
    passage_id: str
    section: str
    text: str
    score: float


class CerIndex:
    """Dense index over the document under audit, so one section can be
    checked against all the others."""

    def __init__(self, passages: Sequence[dict], embedder) -> None:
        self._passages = list(passages)
        self._embedder = embedder
        # Small batches: one 75-passage batch at 512 tokens each spiked RAM by
        # several GB on a 16 GB machine and tripped the memory guard.
        texts = [p["text"] for p in self._passages]
        parts = [embedder.embed_documents(texts[i:i + EMBED_BATCH])
                 for i in range(0, len(texts), EMBED_BATCH)]
        self._matrix = (np.vstack(parts) if parts
                        else np.zeros((0, 0), dtype=np.float32))

    def __len__(self) -> int:
        return len(self._passages)

    def neighbours(self, query: str, k: int, exclude_id: str) -> list[Excerpt]:
        if not self._passages:
            return []
        scores = self._matrix @ self._embedder.embed_query(query)
        order = np.argsort(-scores)
        out: list[Excerpt] = []
        for idx in order:
            passage = self._passages[int(idx)]
            if passage["passage_id"] == exclude_id:
                continue
            out.append(
                Excerpt(
                    passage_id=passage["passage_id"],
                    section=passage.get("section_title") or passage.get("section", ""),
                    text=passage["text"],
                    score=float(scores[idx]),
                )
            )
            if len(out) >= k:
                break
        return out


def _clip(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


def build_reconcile_prompt(claim: str, clause_id: str, clause_text: str,
                           section_title: str, excerpts: Sequence[Excerpt]) -> str:
    blocks = "\n\n".join(
        f"[{e.passage_id}] {e.section}\n{_clip(e.text, MAX_EXCERPT_WORDS)}"
        for e in excerpts
    )
    return f"""An auditor examined the section "{section_title}" of a report in isolation
and flagged the following as missing or insufficient.

CLAIM:
{claim}

REQUIREMENT ({clause_id}):
{_clip(clause_text, MAX_REQUIREMENT_WORDS)}

Below are excerpts from OTHER sections of the same report. Decide whether any
one of them already provides what the claim says is missing.

--- OTHER SECTIONS ---
{blocks}
--- END ---

Answer "addressed": true ONLY if one excerpt clearly contains that content. A
section that merely mentions the topic, or is about something related, does not
count. If true, give that excerpt's ID and quote 8 to 40 words from it VERBATIM
as proof. If none does, answer false.

Return JSON exactly like:
{{"addressed": false, "passage_id": "", "quote": ""}}
or
{{"addressed": true, "passage_id": "CER.4.3.2", "quote": "verbatim words"}}"""


@dataclass
class Reconciliation:
    addressed: bool
    passage_id: str = ""
    quote: str = ""
    detail: str = ""
    error: bool = False     # the judge call itself failed (finding was kept)


def reconcile_finding(
    *,
    claim: str,
    clause_id: str,
    clause_text: str,
    section_title: str,
    passage_id: str,
    index: CerIndex,
    judge: JSONProvider,
    k: int = 3,
    min_grounding: float = 0.82,
) -> Reconciliation:
    """Is the requirement this finding calls missing satisfied elsewhere?

    Returns addressed=True only for a verified answer; every other outcome,
    including every error, is addressed=False so the finding is kept.
    """
    excerpts = index.neighbours(claim, k=k, exclude_id=passage_id)
    if not excerpts:
        return Reconciliation(False, detail="no other sections to check")

    prompt = build_reconcile_prompt(claim, clause_id, clause_text, section_title, excerpts)
    try:
        verdict = judge.complete_json(RECONCILE_SYSTEM, prompt, max_tokens=400)
    except Exception as exc:  # noqa: BLE001 - fail open
        return Reconciliation(False, detail=f"reconciler error: {type(exc).__name__}",
                              error=True)

    data = verdict.data
    if data.get("addressed") is not True:
        return Reconciliation(False, detail="not addressed elsewhere")

    named = str(data.get("passage_id", "")).strip().strip("[]")
    quote = str(data.get("quote", "")).strip()
    shown = {e.passage_id: e for e in excerpts}
    if named not in shown:
        return Reconciliation(False, detail=f"named a section it was not shown: {named!r}")
    if len(quote.split()) < 4:
        return Reconciliation(False, detail="no usable proof quote")
    _, _, ratio = locate_quote(quote, shown[named].text, min_ratio=min_grounding)
    if ratio < min_grounding:
        return Reconciliation(False, detail="proof quote is not in the named section")
    return Reconciliation(True, passage_id=named, quote=quote,
                          detail=f"addressed in {named}")
