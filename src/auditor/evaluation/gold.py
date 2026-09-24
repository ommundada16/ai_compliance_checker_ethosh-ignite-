"""Loading gold labels, and resolving them to the corpus's granularity.

The problem this fixes
----------------------
The expert map was authored at the granularity a regulatory reviewer thinks in
-- "Annex VIII governs device classification", "Article 61(1) is the clinical
evaluation duty". The clause corpus is finer: Annex VIII alone splits into
Annex.VIII, Annex.VIII.1, Annex.VIII.4, Annex.VIII.5, Annex.VIII.7 and more.

Writing a single fine-grained ID to mean a coarse-grained intent silently
created an unanswerable question. For CER section 2.5 the map said
"Annex.VIII.1", meaning Annex VIII; Annex.VIII.1 in the corpus is specifically
"DURATION OF USE". A retriever returning Annex.VIII.5 (the invasive-device
classification rule that actually applies to a ureteral stent) was scored as
wrong for returning the right annex.

The fix
-------
A gold label names a SCOPE, not necessarily a leaf. A retrieved clause
satisfies label L when its ID is L itself or sits beneath L in the clause
hierarchy:

    label Annex.VIII      satisfied by Annex.VIII, Annex.VIII.5, Annex.VIII.7.1
    label Art.61.1        satisfied by Art.61.1 only -- it has no children
    label Annex.I.8       satisfied by Annex.I.8 and Annex.I.8.a

Narrow labels stay narrow. Only labels that genuinely name a container become
more permissive, which is exactly the intent that was lost when the map was
flattened to single IDs.

Fairness
--------
Expansion is a property of the GOLD SET, applied by the shared harness, so
every system is scored against the same resolved judgements. It is not a v2
concession: v1's chunk-to-clause coverage produces clause IDs from the same
corpus and is resolved identically.

The '#' disambiguation suffix is handled explicitly. Annex.I.13.a#1 is a
SIBLING of Annex.I.13.a (a second unnumbered sub-paragraph), not a child, so
prefix matching on raw strings would wrongly fold them together.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

GRADE_PRIMARY = 2
GRADE_SECONDARY = 1


def is_within(clause_id: str, scope: str) -> bool:
    """Does `clause_id` fall under `scope` in the clause hierarchy?

    Exact match, or a strict descendant separated by a dot. Comparison strips
    the '#N' disambiguation suffix from the SCOPE only: Annex.I.13.a#1 is a
    sibling of Annex.I.13.a, so a scope of Annex.I.13.a must not swallow it,
    but a scope of Annex.I.13 must contain both.
    """
    if clause_id == scope:
        return True
    return clause_id.startswith(scope + ".")


def expand_labels(
    labels: Mapping[str, int], corpus_ids: Iterable[str]
) -> dict[str, int]:
    """Resolve scope labels to every corpus clause they cover.

    Where two scopes overlap, the higher grade wins -- a clause that is primary
    under one label and secondary under another is primary.
    """
    corpus = list(corpus_ids)
    resolved: dict[str, int] = {}
    for scope, grade in labels.items():
        for cid in corpus:
            if is_within(cid, scope) and resolved.get(cid, 0) < grade:
                resolved[cid] = int(grade)
    return resolved


@dataclass(frozen=True)
class GoldQuery:
    passage_id: str
    section: str
    labels: dict[str, int]       # resolved to corpus granularity
    scopes: dict[str, int]       # as authored, before expansion

    @property
    def primary(self) -> set[str]:
        return {cid for cid, g in self.labels.items() if g == GRADE_PRIMARY}

    @property
    def primary_scopes(self) -> set[str]:
        return {s for s, g in self.scopes.items() if g == GRADE_PRIMARY}


def load_gold(gold_path: Path, clauses_path: Path) -> list[GoldQuery]:
    with clauses_path.open(encoding="utf-8") as fh:
        corpus_ids = [json.loads(line)["clause_id"] for line in fh if line.strip()]

    out: list[GoldQuery] = []
    with gold_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            scopes = {cid: int(g) for cid, g in row["labels"].items()}
            out.append(
                GoldQuery(
                    passage_id=row["passage_id"],
                    section=row["section"],
                    labels=expand_labels(scopes, corpus_ids),
                    scopes=scopes,
                )
            )
    return out


def scope_recall(retrieved: Sequence[str], primary_scopes: set[str], k: int) -> float:
    """Fraction of primary SCOPES hit by the top k.

    Complements clause-level recall. A scope containing forty clauses would
    otherwise dominate clause-level recall through sheer size, and a retriever
    that surfaced one clause from each of two scopes would score worse than one
    that dumped forty clauses from a single scope and missed the other entirely.

    This is the metric that most closely matches the question a reviewer asks:
    "did it find the provisions that govern this passage?"
    """
    if not primary_scopes:
        return 0.0
    top = list(dict.fromkeys(retrieved))[:k]
    hit = {scope for scope in primary_scopes if any(is_within(c, scope) for c in top)}
    return len(hit) / len(primary_scopes)
