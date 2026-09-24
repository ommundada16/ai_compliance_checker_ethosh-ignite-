"""Structural parser for a Clinical Evaluation Report.

Turns the CER into passage-level units boundaried on dotted section headings,
with template boilerplate stripped and symbol-font bullets repaired.

These passages are the QUERIES in this system: the corpus is the regulation and
the query is a chunk of the document under audit. Shared by the gold-set
builder and the v2 pipeline, for the reason given in parsing/mdr.py.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# Boilerplate stamped on all 91 pages by the document template. Counted, not
# guessed: each of these appears 91 or 92 times in a 91-page document.
BOILERPLATE = (
    "BIORAD MEDISYS PVT. LTD.",
    "Document No. CER/DJS/61",
    "Revision. No 01",
    "CLINICAL EVALUATION REPORT",
    "Revision Date 26-12-2025",
    "PRODUCT- DOUBLE J STENT",
    "Double J Stent (Short Term)",
    "Clinical Evaluation Report",
)
RE_PAGE_MARK = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.I)

# "4.3.1.1  Biocompatibility data" -- at least two levels, so that list items
# like "1 Pusher" inside a section are not mistaken for headings.
RE_HEADING = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\.?\s+(.{3,80})$")

TARGET_WORDS = 320
MAX_WORDS = 520

# Two different floors, because they answer two different questions.
#
# MIN_TAIL_WORDS: a leftover tail shorter than this is folded back into the
# previous passage rather than emitted as a fragment.
#
# MIN_PASSAGE_WORDS: the floor for keeping a section at all. Deliberately low.
# "2.11 Contraindications" is nine words long and is precisely the kind of
# statement MDR Annex I requires a manufacturer to make; dropping it because it
# is terse would remove real obligations from the test set and quietly bias
# every recall number computed against it. Genuinely empty parent headings
# ("4.3.2 Clinical data", 0 words, content lives in its children) still fall
# away on their own.
MIN_TAIL_WORDS = 40
# 8 words keeps "2.11 Contraindications -- There are no known absolute
# contraindications for this device." (9 words), which is a substantive claim a
# regulator would test against Annex I. It still drops "2.16 Instructions for
# use -- Refer IFU Document No: BRP/IFU/DJS/01/03" (6 words), which is a
# cross-reference carrying no assertion to audit.
MIN_PASSAGE_WORDS = 8


@dataclass
class Passage:
    passage_id: str
    section: str
    section_title: str
    text: str
    page_start: int
    page_end: int
    n_words: int
    part: int
    n_parts: int
    text_sha1: str = ""

    def finalise(self) -> Passage:
        self.n_words = len(self.text.split())
        self.text_sha1 = hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:12]
        return self


RE_BAD_BULLET = re.compile(r"[�]")


def clean_lines(text: str) -> list[str]:
    """Drop template boilerplate and repair bullets, line by line.

    Bullet repair has to happen here rather than in normalise(), because
    split_paragraphs() inspects these raw lines to decide where paragraphs
    begin. If it still sees U+FFFD it misses the list-item boundary and welds
    every bullet into one run-on sentence.
    """
    out = []
    for raw in text.split("\n"):
        line = RE_BAD_BULLET.sub("•", raw).strip()
        if not line or line in BOILERPLATE or RE_PAGE_MARK.match(line):
            continue
        out.append(line)
    return out


def normalise(text: str) -> str:
    """Minimal, for the same reason as the clause corpus: this is ground truth."""
    text = text.replace("­", "").replace(" ", " ")
    # The CER uses a symbol font for list bullets, which pdfplumber cannot map
    # and surfaces as U+FFFD. Left alone these corrupt the passage text and
    # collapse list items into a run-on sentence ("Urologists Surgeons trained
    # in endourology ..."). Normalise to a real bullet so the list survives and
    # split_paragraphs() still recognises the item boundary.
    text = re.sub(r"[�]", "•", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1-\2", text)
    return re.sub(r"\s+", " ", text).strip()


# The document template renders its header as a TABLE on every page, so table
# extraction picks it up 91 times unless it is recognised and dropped.
TEMPLATE_TABLE_MARKERS = ("Document No.", "Revision. No", "BIORAD MEDISYS")

# A "table" of one column is a layout artefact, not data.
MIN_TABLE_COLS = 2


def _clean_cell(value: object) -> str:
    """Flatten one cell. Cells wrap internally, so they arrive full of newlines."""
    if value is None:
        return ""
    return normalise(RE_BAD_BULLET.sub("•", str(value)).replace("\n", " "))


def render_table(rows: list[list], caption: str = "") -> str:
    """Turn a table into text a retriever and an LLM can both use.

    Rendered as "header: value" pairs per row rather than as a grid:

        Category: Technical; Characteristic: Design and Operating Principle;
        Device 1: Passive ureteral stent...

    A grid loses its meaning the moment it is flattened into an embedding --
    which is exactly what the previous version did, producing queries like
    "Categor Characteris Device 1 Device 2 Identified Scientific C". Binding
    each value to its column keeps the row readable as a statement, which is
    what both the embedder and the auditing LLM actually need.

    Rows whose cells are all empty, and columns with no header, are dropped.
    """
    cleaned = [[_clean_cell(c) for c in row] for row in rows if row]
    cleaned = [r for r in cleaned if any(c for c in r)]
    if len(cleaned) < 2:
        # A single row carries no header/value relationship; emit it plainly.
        return " ".join(c for r in cleaned for c in r if c).strip()

    header = cleaned[0]
    out: list[str] = []
    if caption:
        out.append(caption)
    for row in cleaned[1:]:
        parts = []
        for idx, cell in enumerate(row):
            if not cell:
                continue
            label = header[idx] if idx < len(header) and header[idx] else ""
            parts.append(f"{label}: {cell}" if label else cell)
        if parts:
            out.append("; ".join(parts) + ".")
    return " ".join(out).strip()


def _is_template_table(rows: list[list]) -> bool:
    flat = " ".join(_clean_cell(c) for row in rows[:3] for c in row)
    return any(marker in flat for marker in TEMPLATE_TABLE_MARKERS)


def page_items(page) -> list[tuple[float, str]]:
    """Text lines and rendered tables for one page, in vertical order.

    Table regions are cut out of the prose before it is read, then re-inserted
    as rendered text at their original vertical position. Without the cut, every
    table's contents appear twice: once as flattened garbage inside the prose
    and once as a rendered block.

    Ordering by position matters because a heading can follow a table on the
    same page; appending tables at the end of a page would file them under the
    wrong section.
    """
    tables = [t for t in page.find_tables() if not _is_template_table(t.extract())]
    boxes = [t.bbox for t in tables]

    def inside_a_table(line: dict) -> bool:
        cx = (line["x0"] + line["x1"]) / 2
        cy = (line["top"] + line["bottom"]) / 2
        return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)

    items: list[tuple[float, str]] = []
    try:
        lines = page.extract_text_lines()
    except Exception:  # noqa: BLE001 - fall back to flat text on odd pages
        lines = []
    if lines:
        for line in lines:
            if inside_a_table(line):
                continue
            items.append((float(line["top"]), line["text"]))
    else:
        for offset, text in enumerate(page.extract_text().split("\n") if page.extract_text() else []):
            items.append((float(offset), text))

    for table in tables:
        rows = table.extract()
        if not rows or max((len(r) for r in rows), default=0) < MIN_TABLE_COLS:
            continue
        rendered = render_table(rows)
        if rendered:
            items.append((float(table.bbox[1]), rendered))

    items.sort(key=lambda item: item[0])
    return items


def collect_sections(pdf_path: Path) -> list[dict]:
    """Walk the document and group content under the dotted heading that owns it."""
    sections: list[dict] = []
    current: dict | None = None
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for _, raw in page_items(page):
                for line in clean_lines(raw):
                    m = RE_HEADING.match(line)
                    if m:
                        if current:
                            sections.append(current)
                        current = {
                            "section": m.group(1),
                            "title": normalise(m.group(2)),
                            "lines": [],
                            "page_start": page_no,
                            "page_end": page_no,
                        }
                        continue
                    if current is None:
                        continue  # front matter, before the first heading
                    current["lines"].append(line)
                    current["page_end"] = page_no
    if current:
        sections.append(current)
    return sections


def split_paragraphs(lines: list[str]) -> list[str]:
    """Regroup hard-wrapped lines into paragraphs.

    A new paragraph starts at a bullet, an enumerator, or a line that follows
    one ending in terminal punctuation. Crude, but it only decides where a long
    section is cut, and cutting at a sentence end is the property that matters.
    """
    paras: list[list[str]] = []
    buf: list[str] = []
    for line in lines:
        starts_new = bool(re.match(r"^([•\-\*]|\(?[a-z0-9]{1,3}[.)])\s+", line))
        if buf and (starts_new or buf[-1].rstrip().endswith((".", ":", ";"))):
            paras.append(buf)
            buf = []
        buf.append(line)
    if buf:
        paras.append(buf)
    return [normalise(" ".join(p)) for p in paras if " ".join(p).strip()]


def chunk_section(sec: dict) -> list[str]:
    """Pack paragraphs up to TARGET_WORDS, never exceeding MAX_WORDS."""
    paras = split_paragraphs(sec["lines"])
    if not paras:
        return []
    out: list[str] = []
    buf: list[str] = []
    count = 0
    for para in paras:
        words = len(para.split())
        if buf and count + words > MAX_WORDS:
            out.append(" ".join(buf))
            buf, count = [], 0
        buf.append(para)
        count += words
        if count >= TARGET_WORDS:
            out.append(" ".join(buf))
            buf, count = [], 0
    if buf:
        tail = " ".join(buf)
        # Fold a stub tail back into the previous passage rather than emitting
        # a fragment that is too small to carry an obligation.
        if out and len(tail.split()) < MIN_TAIL_WORDS:
            out[-1] = out[-1] + " " + tail
        else:
            out.append(tail)
    return out


def disambiguate_sections(sections: list[dict]) -> list[dict]:
    """Give reused section numbers distinct keys.

    The CER numbers two different sections 4.5.1 ("Appraisal method and
    criteria" on p64 and "Requirement on safety" on p75) and likewise 4.5.2.
    That is a defect in the source document, not in this parser, but it still
    has to be handled: without it both sections emit CER.4.5.1#1, the IDs
    collide, and a gold label silently addresses two unrelated passages.

    The first occurrence keeps the plain number so existing IDs stay stable;
    later ones get a "~2", "~3" suffix in document order.
    """
    seen: dict[str, int] = {}
    for sec in sections:
        number = sec["section"]
        seen[number] = seen.get(number, 0) + 1
        sec["section_key"] = number if seen[number] == 1 else f"{number}~{seen[number]}"
    return sections


def build(pdf_path: Path) -> list[Passage]:
    passages: list[Passage] = []
    for sec in disambiguate_sections(collect_sections(pdf_path)):
        parts = chunk_section(sec)
        parts = [p for p in parts if len(p.split()) >= MIN_PASSAGE_WORDS]
        for idx, text in enumerate(parts, start=1):
            pid = f"CER.{sec['section_key']}" + (f"#{idx}" if len(parts) > 1 else "")
            passages.append(
                Passage(
                    passage_id=pid,
                    section=sec["section_key"],
                    section_title=sec["title"],
                    text=text,
                    page_start=sec["page_start"],
                    page_end=sec["page_end"],
                    n_words=0,
                    part=idx,
                    n_parts=len(parts),
                ).finalise()
            )
    return passages
