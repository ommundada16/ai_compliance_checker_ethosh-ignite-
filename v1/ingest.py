import re
import pdfplumber

_FRONT_MATTER_MARKERS = (
    "table of contents", "list of tables", "list of figures",
    "list of abbreviation", "abbreviations and definitions",
    "revision history", "document history", "glossary of terms",
    "list of acronym",
)

# Matches glossary/abbreviation-table lines like "AE Adverse Events" or
# "BfArm The Federal Institute for Drugs and Medical Devices" -- a short
# acronym-like first token followed by a capitalized definition.
_ACRONYM_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9&/]{1,9}\s{1,4}[A-Z(]")

def looks_like_front_matter(text: str) -> bool:
    """Heuristic: is this page a cover/TOC/index/abbreviations page rather than
    substantive report content? These pages have no real compliance content to
    audit, and sending them to the LLM anyway tends to produce hallucinated
    "violations" that can't be located in the text.
    """
    if not text or not text.strip():
        return True
    words = text.split()
    if len(words) < 40:
        return True
    lower = text.lower()
    if any(marker in lower for marker in _FRONT_MATTER_MARKERS):
        return True
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        # TOC pages are dominated by lines like "Section Name ..... 12"
        leader_lines = sum(
            1 for l in lines
            if re.search(r"\.{4,}\s*\d{1,4}\s*$", l) or re.search(r"\s{3,}\d{1,4}\s*$", l)
        )
        if leader_lines / len(lines) > 0.35:
            return True
        # Abbreviation/glossary tables are dominated by "ACRONYM  Definition" lines
        acronym_lines = sum(1 for l in lines if _ACRONYM_LINE_RE.match(l))
        if acronym_lines / len(lines) > 0.5:
            return True
    return False

def extract_text(pdf_path: str) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)

def extract_pages(pdf_path: str, max_pages: int = 2, skip_front_matter: bool = True) -> tuple:
    """Extract page-level text for the document annotation view.

    Scans through the PDF and, when skip_front_matter is True, skips over
    cover/TOC/abbreviations pages entirely rather than counting them toward
    max_pages -- so "N pages" means N pages of actual content, starting
    wherever the real content begins.

    Returns (pages, skipped_page_numbers) where pages is a list of
    {page_num: int, text: str} (page_num is the true 1-indexed PDF page).
    """
    pages = []
    skipped = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            if skip_front_matter and looks_like_front_matter(text):
                skipped.append(i + 1)
                continue
            pages.append({"page_num": i + 1, "text": text})
            if len(pages) >= max_pages:
                break
    return pages, skipped

def quote_locatable(page_text: str, quote: str) -> bool:
    """Can this quote actually be found (exactly or by prefix) in the page text?
    Used to drop findings that can't be anchored to visible text instead of
    showing a "phantom" issue with no highlight to point at.
    """
    if not quote or not quote.strip():
        return False
    quote = quote.strip()
    if quote in page_text:
        return True
    return quote[:40] in page_text


def chunk_text(text: str, words_per_chunk: int = 800, overlap: int = 100):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + words_per_chunk]))
        i += words_per_chunk - overlap
    return chunks