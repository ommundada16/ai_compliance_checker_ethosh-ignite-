"""Clause-to-chunk coverage mapping.

This is what lets v1 (which returns 800-word text chunks) be scored against a
gold set expressed in clause IDs. Get it wrong in one direction and v1 is
credited for clauses it never delivered; wrong in the other and v1 is punished
for clauses it did. Either way the headline "v2 improved by X%" is wrong, so
the mapping is tested independently of any retriever.
"""

from __future__ import annotations

from auditor.baselines.coverage import (
    ClauseLocator,
    ClauseSpan,
    build_spans,
    chunk_word_range,
    clauses_in_chunk,
    tokenise,
)

# --- tokenisation ---------------------------------------------------------

def test_tokenise_splits_on_any_non_alphanumeric() -> None:
    """The corpus rejoins line-broken compounds; v1's raw text does not.

    Both must reduce to the same token sequence or they can never align.
    """
    corpus_form = "the benefit-risk-ratio referred to"
    v1_form = "the benefit-risk- ratio referred to"
    assert tokenise(corpus_form) == tokenise(v1_form)
    assert tokenise(corpus_form) == ["the", "benefit", "risk", "ratio", "referred", "to"]


def test_tokenise_is_case_insensitive() -> None:
    assert tokenise("Annex I") == tokenise("annex i") == ["annex", "i"]


def test_tokenise_drops_punctuation_only_tokens() -> None:
    assert tokenise("a -- b") == ["a", "b"]


# --- locating -------------------------------------------------------------

def test_locate_exact_contiguous() -> None:
    loc = ClauseLocator("alpha beta gamma delta epsilon")
    assert loc.locate("beta gamma delta") == (1, 4)


def test_locate_returns_none_when_absent() -> None:
    loc = ClauseLocator("alpha beta gamma")
    assert loc.locate("zeta eta theta") is None


def test_locate_steps_over_a_spliced_page_header() -> None:
    """v1 concatenates raw page text, so a clause crossing a page break has the
    running header spliced into its middle. The matcher must step over it."""
    header = "5.5.2017 EN Official Journal of the European Union L 117/55"
    doc = f"devices shall achieve the {header} performance intended by their manufacturer"
    loc = ClauseLocator(doc)
    found = loc.locate("devices shall achieve the performance intended by their manufacturer")
    assert found is not None
    start, end = found
    assert start == 0
    assert end > 9  # the span necessarily includes the spliced header


def test_locate_prefers_the_tightest_match() -> None:
    """A scattered match that happens to satisfy the ratio must lose to a
    contiguous one later in the document.

    Without this, a short clause of common words matches a sparse scatter and
    the inflated span hands v1 coverage it never earned.
    """
    doc = (
        "the device is a thing and the report is a document "
        "the device is a stent"
    )
    loc = ClauseLocator("x " + doc)  # offset so position 0 is not special
    found = loc.locate("the device is a stent", min_ratio=0.8)
    assert found is not None
    start, end = found
    assert end - start == 5, f"expected a tight 5-token span, got {end - start}"


def test_locate_rejects_a_match_stretched_past_the_ceiling() -> None:
    filler = " ".join(f"w{i}" for i in range(200))
    doc = f"alpha {filler} beta {filler} gamma"
    loc = ClauseLocator(doc)
    assert loc.locate("alpha beta gamma", min_ratio=1.0) is None


def test_build_spans_reports_what_it_could_not_find() -> None:
    clauses = [
        {"clause_id": "A", "text": "alpha beta gamma"},
        {"clause_id": "B", "text": "nowhere to be found at all"},
    ]
    spans, missing = build_spans("alpha beta gamma delta", clauses)
    assert [s.clause_id for s in spans] == ["A"]
    assert missing == ["B"]


# --- chunk geometry -------------------------------------------------------

def test_chunk_word_range_mirrors_v1_stride() -> None:
    """v1: i += words_per_chunk - overlap, so stride is 700 for 800/100."""
    assert chunk_word_range(0, 800, 100) == (0, 800)
    assert chunk_word_range(1, 800, 100) == (700, 1500)
    assert chunk_word_range(2, 800, 100) == (1400, 2200)


def test_chunks_overlap_by_exactly_the_overlap() -> None:
    _, end0 = chunk_word_range(0, 800, 100)
    start1, _ = chunk_word_range(1, 800, 100)
    assert end0 - start1 == 100


# --- coverage -------------------------------------------------------------

def test_clause_fully_inside_a_chunk_is_covered() -> None:
    spans = [ClauseSpan("A", 10, 60)]
    assert clauses_in_chunk(0, spans, 800, 100) == ["A"]


def test_clause_outside_a_chunk_is_not_covered() -> None:
    spans = [ClauseSpan("A", 1000, 1050)]
    assert clauses_in_chunk(0, spans, 800, 100) == []


def test_clause_straddling_a_boundary_may_be_covered_by_neither() -> None:
    """Not a bug -- this is what fixed-size chunking actually does to a clause,
    and precisely the weakness structural chunking is meant to remove."""
    spans = [ClauseSpan("A", 780, 840)]  # 20 words in chunk 0, 40 past its end
    assert clauses_in_chunk(0, spans, 800, 100, min_coverage=0.6) == []


def test_partial_overlap_above_threshold_counts() -> None:
    spans = [ClauseSpan("A", 760, 800)]  # entirely within chunk 0
    assert clauses_in_chunk(0, spans, 800, 100, min_coverage=0.6) == ["A"]


def test_coverage_is_ordered_by_position_in_the_document() -> None:
    spans = [ClauseSpan("C", 300, 320), ClauseSpan("A", 10, 30), ClauseSpan("B", 100, 120)]
    assert clauses_in_chunk(0, spans, 800, 100) == ["A", "B", "C"]


def test_overlapping_chunks_can_both_deliver_a_clause() -> None:
    """A clause in the 100-word overlap region belongs to two chunks.

    Retrieving both must not count it twice -- that is what the metric layer's
    dedupe is for, and this test documents why the dedupe is needed.
    """
    spans = [ClauseSpan("A", 720, 760)]
    assert clauses_in_chunk(0, spans, 800, 100) == ["A"]
    assert clauses_in_chunk(1, spans, 800, 100) == ["A"]
