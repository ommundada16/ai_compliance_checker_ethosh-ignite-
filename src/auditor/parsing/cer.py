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


def collect_sections(pdf_path: Path) -> list[dict]:
    """Walk the document and group lines under the dotted heading that owns them."""
    sections: list[dict] = []
    current: dict | None = None
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for line in clean_lines(page.extract_text() or ""):
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


def build(pdf_path: Path) -> list[Passage]:
    passages: list[Passage] = []
    for sec in collect_sections(pdf_path):
        parts = chunk_section(sec)
        parts = [p for p in parts if len(p.split()) >= MIN_PASSAGE_WORDS]
        for idx, text in enumerate(parts, start=1):
            pid = f"CER.{sec['section']}" + (f"#{idx}" if len(parts) > 1 else "")
            passages.append(
                Passage(
                    passage_id=pid,
                    section=sec["section"],
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
