"""Guardrails between the model's output and a reported finding.

v1 had one check: does the quote appear in the page text, either exactly or by
its first 40 characters. That catches a fabricated quote but passes a real
quote attached to an invented clause, and it cannot say WHERE in the text the
quote sits, so the UI has nothing to highlight.

Each guardrail here is separable and each records why it rejected something, so
the ablation can answer the question that matters: does this guardrail remove
more false positives than it removes true ones? A guardrail that costs more
recall than it buys precision is a bad guardrail, and without the drop reasons
there is no way to tell.

Order matters. Cheap deterministic checks run before the expensive model call,
so the judge is only ever asked about findings that are already grounded and
correctly cited.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from auditor.audit.schema import (
    DropReason,
    Finding,
    RawFinding,
)

# --- input sanitisation ---------------------------------------------------

# A PDF is untrusted input. A document under audit can contain text addressed
# to the model rather than to a reader -- whether planted deliberately or
# pasted in by accident -- and the auditor would otherwise follow it.
INJECTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above)",
    r"you\s+are\s+now\s+(?:a|an)\s+",
    r"new\s+(?:system\s+)?(?:instructions?|prompt)\s*:",
    r"</?(?:system|assistant|user)>",
    r"\[\s*(?:system|INST)\s*\]",
    r"forget\s+(?:everything|all)\s+(?:you|above)",
    r"do\s+not\s+report\s+(?:any\s+)?(?:violations?|findings?|issues?)",
    r"return\s+(?:an\s+)?empty\s+(?:list|array|findings)",
)
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

NEUTRALISED = "[redacted: instruction-like text in source document]"


def sanitise_passage(text: str) -> tuple[str, list[str]]:
    """Neutralise instruction-like text in an untrusted document.

    Returns the cleaned text and whatever was removed, so a suppression attempt
    is visible in the report rather than silently erased. Suppressing findings
    is the payload that actually matters here -- an injected "do not report any
    violations" turns a compliance auditor into a rubber stamp, and it would
    leave no trace at all.
    """
    found = [m.group(0) for m in _INJECTION_RE.finditer(text)]
    if not found:
        return text, []
    return _INJECTION_RE.sub(NEUTRALISED, text), found


# --- grounding ------------------------------------------------------------

def _normalise_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def locate_quote(quote: str, passage: str, min_ratio: float = 0.82) -> tuple[int, int, float]:
    """Find a quote inside the passage and return (start, end, score).

    Three escalating strategies, because a model's quote is rarely byte-exact:

      1. exact substring
      2. whitespace-insensitive match, mapped back to original offsets
      3. best fuzzy window, accepted only above `min_ratio`

    Returns (-1, -1, 0.0) when nothing clears the bar. Character offsets are
    returned, not a boolean, because a finding without a span cannot be
    highlighted -- and "trust me, it's in there" is exactly what v1 offered.
    """
    if not quote or not passage:
        return -1, -1, 0.0

    idx = passage.find(quote)
    if idx != -1:
        return idx, idx + len(quote), 1.0

    # Whitespace-insensitive: build a map from normalised offsets back to raw.
    raw_positions: list[int] = []
    squeezed: list[str] = []
    previous_space = False
    for position, char in enumerate(passage):
        if char.isspace():
            if previous_space:
                continue
            squeezed.append(" ")
            raw_positions.append(position)
            previous_space = True
        else:
            squeezed.append(char.lower())
            raw_positions.append(position)
            previous_space = False
    flat = "".join(squeezed)
    needle = _normalise_for_match(quote)

    idx = flat.find(needle)
    if idx != -1:
        start = raw_positions[idx]
        end_index = min(idx + len(needle) - 1, len(raw_positions) - 1)
        return start, raw_positions[end_index] + 1, 0.98

    if len(needle) < 12:
        # Too short to fuzzy-match safely; a 3-word fragment matches anywhere.
        return -1, -1, 0.0

    matcher = SequenceMatcher(None, flat, needle, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
    if not blocks:
        return -1, -1, 0.0
    best = max(blocks, key=lambda b: b.size)
    window_start = max(0, best.a - best.b)
    window_end = min(len(flat), window_start + len(needle))
    ratio = SequenceMatcher(None, flat[window_start:window_end], needle).ratio()
    if ratio < min_ratio:
        return -1, -1, 0.0
    start = raw_positions[window_start]
    end_index = min(window_end - 1, len(raw_positions) - 1)
    return start, raw_positions[end_index] + 1, ratio


# --- the gate -------------------------------------------------------------

def check_finding(
    raw: RawFinding,
    passage_id: str,
    passage_text: str,
    page: int,
    allowed_clauses: dict[str, str],
    min_confidence: float = 0.35,
    min_grounding: float = 0.82,
) -> tuple[Finding | None, DropReason | None, str]:
    """Run every deterministic guardrail. Returns (finding, reason, detail).

    `allowed_clauses` maps clause_id -> path for exactly the clauses that were
    retrieved for this passage. Citing anything else is rejected: the model
    cannot have read a clause it was not shown, so a citation outside that set
    is recalled from training data rather than from the regulation in front of
    it. That is the failure mode most likely to look convincing and be wrong.
    """
    if not raw.explanation.strip() or not raw.suggested_correction.strip():
        return None, DropReason.EMPTY_EXPLANATION, ""

    if raw.confidence < min_confidence:
        return None, DropReason.LOW_CONFIDENCE, f"confidence={raw.confidence:.2f}"

    if raw.clause_id not in allowed_clauses:
        return None, DropReason.UNKNOWN_CLAUSE, f"cited {raw.clause_id!r}"

    start, end, score = locate_quote(raw.source_quote, passage_text, min_grounding)
    if start < 0:
        return None, DropReason.UNGROUNDED_QUOTE, raw.source_quote[:90]

    return (
        Finding(
            **raw.model_dump(),
            passage_id=passage_id,
            page=page,
            quote_start=start,
            quote_end=end,
            clause_path=allowed_clauses[raw.clause_id],
            grounding_score=score,
        ),
        None,
        "",
    )
