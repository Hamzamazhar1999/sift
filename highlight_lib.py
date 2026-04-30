"""Robust PDF highlighting that survives multi-line wraps, hyphenation, and
mild paraphrases.
"""
import difflib
import json
import re
from pathlib import Path

import pymupdf


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _NON_ALNUM.sub("", s.lower())


def find_quote_rects(page, quote: str, min_words: int = 3):
    """Find rects on `page` covering `quote`. Returns (rects, matched_text).

    matched_text is the actual page text (with original casing/punct) that
    was highlighted — caller should store this as the citation's "quote" so
    the UI shows what's actually highlighted in the PDF.
    """
    words = page.get_text("words")  # (x0,y0,x1,y1, text, block, line, word)
    if not words:
        return [], None

    page_words = [(_norm(w[4]), w) for w in words]
    page_words = [(n, w) for n, w in page_words if n]
    if not page_words:
        return [], None

    quote_norm = [w for w in (_norm(t) for t in quote.split()) if w]
    if not quote_norm:
        return [], None

    page_strs = [n for n, _ in page_words]
    qlen = len(quote_norm)

    # Fast path: exact contiguous match on normalized words.
    for i in range(len(page_strs) - qlen + 1):
        if page_strs[i : i + qlen] == quote_norm:
            picked = [page_words[i + j][1] for j in range(qlen)]
            return _rects(picked), _join(picked)

    # Fallback: longest contiguous matching block (handles paraphrase / partial).
    matcher = difflib.SequenceMatcher(None, page_strs, quote_norm, autojunk=False)
    match = matcher.find_longest_match(0, len(page_strs), 0, qlen)
    if match.size < min(min_words, qlen):
        return [], None

    picked = [page_words[match.a + j][1] for j in range(match.size)]
    return _rects(picked), _join(picked)


def _rects(picked):
    return [pymupdf.Rect(w[0], w[1], w[2], w[3]) for w in picked]


def _join(picked):
    return " ".join(w[4] for w in picked)


def highlight_pdf(
    input_pdf: str,
    output_pdf: str,
    citations_path: str,
    passages: list,
):
    """passages: list of {"page": int (1-indexed), "quote": str}.

    Writes the highlighted PDF to output_pdf and a citations JSON to
    citations_path: [{"page": int, "quote": str, "found": bool}], where
    "quote" is updated to the text actually highlighted in the PDF.
    """
    doc = pymupdf.open(input_pdf)
    citations = []
    for p in passages:
        page_num = int(p["page"])
        original = p["quote"]
        if page_num < 1 or page_num > len(doc):
            citations.append({"page": page_num, "quote": original, "found": False})
            continue
        page = doc[page_num - 1]
        rects, matched = find_quote_rects(page, original)
        if rects:
            for r in rects:
                page.add_highlight_annot(r)
            citations.append(
                {"page": page_num, "quote": matched or original, "found": True}
            )
        else:
            citations.append({"page": page_num, "quote": original, "found": False})
    doc.save(output_pdf)
    doc.close()
    Path(citations_path).write_text(json.dumps(citations, indent=2))
    return citations
