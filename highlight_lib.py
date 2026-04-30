"""Robust PDF highlighting that survives multi-line wraps, hyphenation, and
mild paraphrases.
"""
import colorsys
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


def _palette(n: int):
    """Return n visually-distinct pastel colors as ((r,g,b)_0to1, "#rrggbb")
    pairs. Uses golden-angle hue stepping so colors stay distinguishable as
    n grows."""
    phi = 0.6180339887498949
    out = []
    for i in range(max(n, 1)):
        h = ((i * phi) + 0.13) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, 0.78, 0.85)
        hex_str = f"#{int(round(r*255)):02x}{int(round(g*255)):02x}{int(round(b*255)):02x}"
        out.append(((r, g, b), hex_str))
    return out


def _best_match_across_pages(doc, quote: str, prefer_page: int | None = None):
    """Scan every page for `quote`, return (rects, matched, page_num) for
    whichever yields the longest matching word run. The cited `prefer_page`
    gets a small score boost so it wins ties — but a substantively longer
    match on another page always wins, which is the whole point: the model
    sometimes mis-cites the page number."""
    best = None  # (rects, matched, page_num, score)
    for j in range(len(doc)):
        rects, matched = find_quote_rects(doc[j], quote)
        if not rects:
            continue
        score = len(rects) + (0.5 if (j + 1) == prefer_page else 0.0)
        if best is None or score > best[3]:
            best = (rects, matched, j + 1, score)
    return best


def highlight_pdf(
    input_pdf: str,
    output_pdf: str,
    citations_path: str,
    passages: list,
):
    """passages: list of {"page": int (1-indexed), "quote": str}.

    For each passage:
      1. Try the cited page.
      2. If nothing matches, scan every other page for the longest match —
         this catches off-by-one or wrong-section page numbers without
         dropping the citation.
    Each citation gets a distinct pastel color used both on the PDF
    highlight and exposed via citations.json["color"] for the UI to mirror,
    so the same yellow/peach/etc. shade ties a chip to its highlight.

    Citations JSON: [{"page", "quote", "found", "color"}]. "page" is the
    page actually highlighted on (may differ from the input if fallback
    found it elsewhere); "quote" is the text actually highlighted.
    """
    doc = pymupdf.open(input_pdf)
    palette = _palette(len(passages))
    citations = []
    for i, p in enumerate(passages):
        rgb, hex_color = palette[i]
        original = p["quote"]
        cited_page = int(p["page"])

        best = _best_match_across_pages(doc, original, prefer_page=cited_page)
        if best:
            rects, matched, used_page, _ = best
            page = doc[used_page - 1]
            for r in rects:
                annot = page.add_highlight_annot(r)
                annot.set_colors(stroke=rgb)
                annot.update()
            citations.append(
                {
                    "page": used_page,
                    "quote": matched or original,
                    "found": True,
                    "color": hex_color,
                }
            )
        else:
            citations.append(
                {"page": cited_page, "quote": original, "found": False}
            )
    doc.save(output_pdf)
    doc.close()
    Path(citations_path).write_text(json.dumps(citations, indent=2))
    return citations
