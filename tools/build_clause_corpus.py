"""Phase 1a -- extract a stable clause corpus from EU MDR 2017/745.

This is GOLD-SET TOOLING, not part of the v2 runtime pipeline. Its output,
eval_data/clauses.jsonl, is a frozen artefact: every retrieval label in the
gold set points at a clause_id produced here, so the IDs must stay stable
across runs and across pipeline versions. Re-running this script on the same
PDF must produce byte-identical output.

Why a separate corpus at all, when v2 will have its own parser? Because the
labels have to outlive the parser. If retrieval ground truth were expressed in
terms of whatever chunks the current parser happens to emit, every parsing
change would silently invalidate the test set, and no two phases would be
comparable. The clause IDs are the fixed reference frame.

Structure of the regulation, as verified against the source PDF:

    pp.   1-12    recitals (non-normative preamble)  -- EXCLUDED
    pp.  13-91    Chapters I-X, Articles 1-123       -- normative
    pp.  94-174   Annexes I-XVII                     -- normative

Recitals are excluded deliberately: they explain legislative intent but impose
no obligation, so a "violation" of a recital is not a finding a regulator would
accept.

Usage:
    python tools/build_clause_corpus.py
    python tools/build_clause_corpus.py --pdf data/guideline.pdf --out eval_data/clauses.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- page geometry -------------------------------------------------------
# Body text begins after the recitals and the enacting formula.
FIRST_BODY_PAGE = 13

# --- line classifiers ----------------------------------------------------
# Every page carries a running header, in one of two orientations depending on
# whether the page is recto or verso. Both contain this phrase.
RE_RUNNING_HEAD = re.compile(r"Official Journal of the European Union", re.I)

# Standalone structural headings. These must be anchored to the whole line:
# "Article 114(3)" appears constantly as an inline cross-reference and must not
# be mistaken for the start of Article 114.
RE_CHAPTER = re.compile(r"^CHAPTER\s+([IVXL]+)$")
RE_ARTICLE = re.compile(r"^Article\s+(\d+)$")
RE_ANNEX = re.compile(r"^ANNEX\s+([IVXL]+)$")
RE_PART = re.compile(r"^PART\s+([A-Z])$")
RE_SECTION = re.compile(r"^SECTION\s+(\d+)$")

# Body enumerators.
RE_NUM_PARA = re.compile(r"^(\d+)\.\s+(.*)$")        # "1. By way of derogation..."
RE_ALPHA_PT = re.compile(r"^\(([a-z])\)\s+(.*)$")    # "(a) the device shall..."
RE_ROMAN_PT = re.compile(r"^\(([ivx]+)\)\s+(.*)$")   # "(i) ..."

# A heading line is short, has no terminal punctuation, and is not an enumerator.
MAX_TITLE_WORDS = 18


@dataclass
class Clause:
    clause_id: str
    kind: str
    path: str
    text: str
    page_start: int
    page_end: int
    chapter: str | None = None
    chapter_title: str | None = None
    article: int | None = None
    article_title: str | None = None
    annex: str | None = None
    annex_title: str | None = None
    part: str | None = None
    section: str | None = None
    paragraph: str | None = None
    n_words: int = 0
    text_sha1: str = ""

    def finalise(self) -> Clause:
        self.text = normalise(self.text)
        self.n_words = len(self.text.split())
        self.text_sha1 = hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:12]
        return self


@dataclass
class _Cursor:
    """Where we currently are in the document tree."""

    chapter: str | None = None
    chapter_title: str | None = None
    article: int | None = None
    article_title: str | None = None
    annex: str | None = None
    annex_title: str | None = None
    part: str | None = None
    section: str | None = None
    expect_title_for: str | None = None
    buf: list[str] = field(default_factory=list)
    buf_page_start: int = 0
    buf_labels: tuple[str, ...] = ()
    # The enumerator context a sub-point hangs off. Article 61(3) is followed by
    # points (a), (b), (c); those must become Art.61.3.a and not a bare Art.61.a,
    # which would collide with the points under Article 61(6).
    para_num: str | None = None
    para_alpha: str | None = None

    def reset_enumeration(self) -> None:
        self.para_num = None
        self.para_alpha = None


def normalise(text: str) -> str:
    """Conservative cleanup.

    Deliberately minimal. This is ground-truth data; aggressive rewriting here
    would silently change what the labels refer to. We only undo artefacts that
    are unambiguously extraction noise.
    """
    # Soft hyphens and non-breaking spaces from the OJ typesetting.
    text = text.replace("­", "").replace(" ", " ")
    # "benefit-risk- ratio" -> "benefit-risk-ratio": hyphen followed by a space
    # mid-compound is a line-break artefact, not real punctuation.
    text = re.sub(r"(\w)-\s+(\w)", r"\1-\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_running_head(lines: list[str]) -> list[str]:
    return [ln for ln in lines if not RE_RUNNING_HEAD.search(ln)]


def looks_like_title(line: str) -> bool:
    if not line or len(line.split()) > MAX_TITLE_WORDS:
        return False
    if line.endswith((".", ";", ":", ",")):
        return False
    return not (RE_NUM_PARA.match(line) or RE_ALPHA_PT.match(line) or RE_ROMAN_PT.match(line))


class CorpusBuilder:
    def __init__(self) -> None:
        self.cur = _Cursor()
        self.clauses: list[Clause] = []

    # -- buffer management -------------------------------------------------
    def _flush(self, page: int) -> None:
        """Emit whatever paragraph is currently buffered."""
        cur = self.cur
        if not cur.buf:
            return
        text = " ".join(cur.buf).strip()
        cur.buf = []
        if not text:
            cur.buf_labels = ()
            return

        labels = cur.buf_labels
        cur.buf_labels = ()
        label = ".".join(labels) if labels else None
        leaf = labels[-1] if labels else None

        if cur.annex:
            base = f"Annex.{cur.annex}"
            path_bits = [f"Annex {cur.annex}"]
            if cur.part:
                base += f".{cur.part}"
                path_bits.append(f"Part {cur.part}")
            if cur.section:
                base += f".S{cur.section}"
                path_bits.append(f"Section {cur.section}")
            if label:
                base += f".{label}"
                path_bits.extend(f"({x})" for x in labels)
            kind = "annex_section"
            clause = Clause(
                clause_id=base,
                kind=kind,
                path=" > ".join(path_bits),
                text=text,
                page_start=cur.buf_page_start or page,
                page_end=page,
                annex=cur.annex,
                annex_title=cur.annex_title,
                part=cur.part,
                section=cur.section,
                paragraph=leaf,
            )
        elif cur.article is not None:
            base = f"Art.{cur.article}"
            path_bits = []
            if cur.chapter:
                path_bits.append(f"Chapter {cur.chapter}")
            path_bits.append(f"Article {cur.article}")
            if label:
                base += f".{label}"
                path_bits.extend(f"({x})" for x in labels)
            clause = Clause(
                clause_id=base,
                kind="article_paragraph",
                path=" > ".join(path_bits),
                text=text,
                page_start=cur.buf_page_start or page,
                page_end=page,
                chapter=cur.chapter,
                chapter_title=cur.chapter_title,
                article=cur.article,
                article_title=cur.article_title,
                paragraph=leaf,
            )
        else:
            return  # text before any structural anchor: enacting formula, etc.

        self.clauses.append(clause.finalise())

    def _start(self, level: str, label: str, first_line: str, page: int) -> None:
        """Open a new clause at the given enumeration level.

        Levels nest: a numbered paragraph resets the alpha context, an alpha
        point hangs off the current number, a roman point off the current alpha.
        That is what turns a bare "Art.61.a" into "Art.61.3.a".
        """
        self._flush(page)
        cur = self.cur
        if level == "num":
            cur.para_num = label
            cur.para_alpha = None
            cur.buf_labels = (label,)
        elif level == "alpha":
            cur.para_alpha = label
            cur.buf_labels = tuple(x for x in (cur.para_num, label) if x)
        else:  # roman
            cur.buf_labels = tuple(x for x in (cur.para_num, cur.para_alpha, label) if x)
        cur.buf_page_start = page
        if first_line:
            cur.buf.append(first_line)

    # -- main loop ---------------------------------------------------------
    def feed_page(self, page_no: int, text: str) -> None:
        cur = self.cur
        for raw in strip_running_head(text.split("\n")):
            line = raw.strip()
            if not line:
                continue

            # A title line was promised by the previous structural heading.
            if cur.expect_title_for:
                target = cur.expect_title_for
                cur.expect_title_for = None
                if looks_like_title(line):
                    if target == "chapter":
                        cur.chapter_title = line
                    elif target == "article":
                        cur.article_title = line
                    elif target == "annex":
                        cur.annex_title = line
                    continue
                # Not a title after all: fall through and treat as body.

            if m := RE_CHAPTER.match(line):
                self._flush(page_no)
                # Chapters also appear *inside* annexes; do not let one reset
                # the annex context.
                cur.chapter = m.group(1)
                cur.chapter_title = None
                cur.expect_title_for = "chapter"
                cur.reset_enumeration()
                if cur.annex:
                    cur.section = None
                continue

            if m := RE_ARTICLE.match(line):
                self._flush(page_no)
                cur.article = int(m.group(1))
                cur.article_title = None
                cur.annex = None  # articles and annexes are disjoint regions
                cur.part = None
                cur.section = None
                cur.expect_title_for = "article"
                cur.reset_enumeration()
                continue

            if m := RE_ANNEX.match(line):
                self._flush(page_no)
                cur.annex = m.group(1)
                cur.annex_title = None
                cur.article = None
                cur.article_title = None
                cur.part = None
                cur.section = None
                cur.chapter = None
                cur.expect_title_for = "annex"
                cur.reset_enumeration()
                continue

            if m := RE_PART.match(line):
                self._flush(page_no)
                cur.part = m.group(1)
                cur.section = None
                cur.reset_enumeration()
                continue

            if m := RE_SECTION.match(line):
                self._flush(page_no)
                cur.section = m.group(1)
                cur.reset_enumeration()
                continue

            # Body enumerators start a new clause. Alpha is tested before roman
            # because "(i)" is ambiguous -- it is the ninth alpha point far more
            # often than the first roman one in this regulation.
            if m := RE_NUM_PARA.match(line):
                self._start("num", m.group(1), m.group(2), page_no)
                continue
            if m := RE_ALPHA_PT.match(line):
                self._start("alpha", m.group(1), m.group(2), page_no)
                continue
            if m := RE_ROMAN_PT.match(line):
                self._start("roman", m.group(1), m.group(2), page_no)
                continue

            # Continuation of the current paragraph.
            if not cur.buf:
                cur.buf_page_start = page_no
            cur.buf.append(line)

    def finish(self, last_page: int) -> list[Clause]:
        self._flush(last_page)
        return self.clauses


def dedupe_ids(clauses: list[Clause]) -> list[Clause]:
    """Disambiguate repeated IDs.

    Unnumbered trailing paragraphs and annex sub-points can legitimately
    produce the same base ID twice. Suffix them deterministically rather than
    dropping them -- silently losing normative text would bias every recall
    number computed against this corpus.
    """
    seen: dict[str, int] = {}
    for c in clauses:
        if c.clause_id in seen:
            seen[c.clause_id] += 1
            c.clause_id = f"{c.clause_id}#{seen[c.clause_id]}"
        else:
            seen[c.clause_id] = 0
    return clauses


def build(pdf_path: Path, first_page: int) -> list[Clause]:
    builder = CorpusBuilder()
    last = first_page
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i in range(first_page - 1, total):
            page_no = i + 1
            builder.feed_page(page_no, pdf.pages[i].extract_text() or "")
            last = page_no
    return dedupe_ids(builder.finish(last))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, default=PROJECT_ROOT / "data" / "guideline.pdf")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval_data" / "clauses.jsonl")
    ap.add_argument("--first-page", type=int, default=FIRST_BODY_PAGE)
    ap.add_argument("--min-words", type=int, default=8,
                    help="Drop fragments shorter than this; they are almost always "
                         "stray table cells or heading spill, not obligations.")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 1

    clauses = build(args.pdf, args.first_page)
    kept = [c for c in clauses if c.n_words >= args.min_words]
    dropped = len(clauses) - len(kept)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for c in kept:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    arts = sum(1 for c in kept if c.kind == "article_paragraph")
    anns = sum(1 for c in kept if c.kind == "annex_section")
    words = sum(c.n_words for c in kept)
    print(f"clauses written : {len(kept)}  -> {args.out.relative_to(PROJECT_ROOT)}")
    print(f"  article paras : {arts}")
    print(f"  annex sections: {anns}")
    print(f"  dropped (<{args.min_words}w): {dropped}")
    print(f"  total words   : {words:,}")
    print(f"  mean words    : {words // max(len(kept), 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
