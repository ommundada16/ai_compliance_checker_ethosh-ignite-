"""Map v1's 800-word text chunks onto clause IDs.

The problem this solves
-----------------------
v1 retrieves chunks of raw text. The gold set is expressed in clause IDs. To
score v1 against the same ground truth as v2, we need to know which clauses a
returned chunk actually delivered to the LLM.

The approach
------------
Locate every clause inside the exact word sequence v1 works on, then express
each chunk as a word range. A chunk covers a clause when it contains enough of
that clause's words for the LLM to have been able to use it.

Everything is done in v1's own coordinate system -- the word list produced by
`text.split()` over the concatenated page text -- so no assumption about v2's
parser leaks into v1's score.

Why this is deliberately generous to v1
---------------------------------------
An 800-word chunk spans many clauses, and this maps ALL of them as retrieved.
v1 therefore gets credit for every clause it happened to sweep up, including
ones it never "aimed" at. That bias is intentional: it makes the eventual v2
improvement a conservative claim rather than a flattering one. It is also why
the comparison must be read at a matched context budget -- see
`evaluation/compare.py` -- because three 800-word chunks is roughly 2400 words
of context against five ~66-word clauses, and equal `k` is not equal
information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split on ANY run of non-alphanumeric characters, rather than splitting on
# whitespace and then stripping punctuation inside each word.
#
# The difference matters. The clause corpus rejoins line-broken compounds, so
# "benefit-risk- ratio" becomes "benefit-risk-ratio"; v1's raw text keeps the
# break. Stripping punctuation within whitespace-delimited words yields
# ["benefitriskratio"] for the corpus and ["benefitrisk", "ratio"] for v1 --
# sequences that can never align. Splitting on non-alphanumerics yields
# ["benefit", "risk", "ratio"] for both.
#
# This is why Art.61.1 -- the central clinical-evaluation obligation, and a
# primary gold label -- was previously unlocatable, which would have silently
# capped v1's recall on the most important clause in the corpus.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class ClauseSpan:
    """Where a clause sits in v1's word sequence."""

    clause_id: str
    start: int
    end: int  # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start


class ClauseLocator:
    """Finds clause texts inside the v1 word sequence.

    Built once per document. Uses a first-token index so that locating 1320
    clauses in a ~100k-word document stays linear-ish rather than quadratic.
    """

    def __init__(self, document_text: str) -> None:
        self.tokens = tokenise(document_text)
        self._index: dict[str, list[int]] = {}
        for pos, tok in enumerate(self.tokens):
            self._index.setdefault(tok, []).append(pos)

    def locate(self, clause_text: str, min_ratio: float = 0.85) -> tuple[int, int] | None:
        """Return the word range holding this clause, or None.

        Anchors on the clause's first token, then walks forward counting
        matches. A candidate is accepted when at least `min_ratio` of the
        clause's tokens line up in order. Exact equality is too brittle:
        pdfplumber emits occasional stray tokens, and the corpus normalises
        hyphenation that the raw v1 text keeps.
        """
        needle = tokenise(clause_text)
        if not needle:
            return None
        required = max(1, int(len(needle) * min_ratio))
        # Hard ceiling on how far a match may stretch. Generous enough to step
        # over a spliced-in running header (~10 tokens) but tight enough to
        # reject a scattered match across unrelated text. Without it, a short
        # clause of common words matches a sparse scatter -- a 9-token clause
        # was spanning 34 tokens, and coverage computed from an inflated span
        # hands v1 credit for clauses a chunk never really contained.
        max_span = int(len(needle) * 1.5) + 24

        # (matched, -span, start, end): most tokens matched wins; ties go to
        # the TIGHTEST span, not merely the first one found.
        best: tuple[int, int, int, int] | None = None
        for start in self._index.get(needle[0], ()):
            matched = 0
            j = start
            limit = min(len(self.tokens), start + max_span)
            for tok in needle:
                # Bounded subsequence match: look ahead anywhere up to `limit`
                # for the next needle token, and if it is not there, leave j
                # alone rather than consuming haystack the remaining tokens
                # still need.
                #
                # A narrow fixed window does not work here. v1 concatenates raw
                # page text, so a clause spanning a page break has the running
                # header ("5.5.2017 EN Official Journal of the European Union
                # L 117/55", ~10 tokens) spliced into its middle. A 4-token
                # window cannot step over that, which is what made Art.8.1 and
                # Annex.XIV.A.3 -- both gold labels -- unlocatable.
                #
                # `limit` still bounds the match to roughly twice the clause
                # length, so this cannot stretch across unrelated text.
                probe = j
                while probe < limit and self.tokens[probe] != tok:
                    probe += 1
                if probe < limit:
                    matched += 1
                    j = probe + 1
                else:
                    break
            if matched < required:
                continue
            candidate = (matched, -(j - start), start, j)
            if best is None or candidate > best:
                best = candidate
            # Stop only on a perfectly contiguous full match, which by
            # definition cannot be beaten. A looser tolerance here (span <=
            # len + 8) exits at the first *good enough* candidate and never
            # reaches a tighter one further along, defeating the tie-break
            # above.
            if matched == len(needle) and (j - start) == len(needle):
                break
        if best is None:
            return None
        return best[2], best[3]


def build_spans(
    document_text: str, clauses: list[dict], min_ratio: float = 0.85
) -> tuple[list[ClauseSpan], list[str]]:
    """Locate every clause. Returns (spans, unlocatable_clause_ids)."""
    locator = ClauseLocator(document_text)
    spans: list[ClauseSpan] = []
    missing: list[str] = []
    for clause in clauses:
        found = locator.locate(clause["text"], min_ratio=min_ratio)
        if found is None:
            missing.append(clause["clause_id"])
            continue
        spans.append(ClauseSpan(clause["clause_id"], found[0], found[1]))
    return spans, missing


def chunk_word_range(index: int, words_per_chunk: int, overlap: int) -> tuple[int, int]:
    """The word range v1's chunk `index` covers.

    Mirrors v1's loop exactly: stride is words_per_chunk - overlap.
    """
    stride = words_per_chunk - overlap
    start = index * stride
    return start, start + words_per_chunk


def clauses_in_chunk(
    chunk_index: int,
    spans: list[ClauseSpan],
    words_per_chunk: int,
    overlap: int,
    min_coverage: float = 0.6,
) -> list[str]:
    """Clause IDs a chunk delivers, ordered by position in the chunk.

    `min_coverage` is the fraction of a clause that must fall inside the chunk.
    A clause straddling a chunk boundary can therefore be delivered by neither
    -- which is not a bug but the actual behaviour of fixed-size chunking, and
    precisely the weakness the v2 structural chunker is meant to remove.
    """
    lo, hi = chunk_word_range(chunk_index, words_per_chunk, overlap)
    hits: list[tuple[int, str]] = []
    for span in spans:
        if span.end <= lo or span.start >= hi:
            continue
        overlap_words = min(span.end, hi) - max(span.start, lo)
        if span.length and overlap_words / span.length >= min_coverage:
            hits.append((span.start, span.clause_id))
    return [cid for _, cid in sorted(hits)]
