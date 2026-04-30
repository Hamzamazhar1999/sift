"""Shared agent logic for the CLI and the web app.
"""
import json
import re
from pathlib import Path

import pymupdf

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)


PROMPT_TEMPLATE = """\
You are answering a question about a PDF and producing a highlighted copy.

INPUT PDF:         {pdf_path}
PAGES TEXT FILE:   {pages_text_path}  ({page_count} pages, {char_count} chars)
OUTPUT PDF:        {output_path}
CITATIONS JSON:    {citations_path}
HIGHLIGHT LIB DIR: {highlight_lib_dir}
QUESTION:          {question}
MODE:              {mode}

{mode_guidance}
{conversation_history}
Do this end-to-end without asking for confirmation:

1. Read the ENTIRE pages text file ({pages_text_path}). It is split by page
   markers like "=== PAGE N ===". The Read tool defaults to 2000 lines per
   call — if the file is longer, call Read again with offset/limit (or use
   `wc -l` first to size it) until you have read every page through page
   {page_count}. Do NOT use the Read tool on the input PDF directly.
2. Find the passages that ground the answer. For each, capture the **exact
   verbatim wording** from the PDF (same casing, same punctuation) and the
   1-indexed page number. CITE GENEROUSLY: every distinct claim in your
   answer should have its own citation. Do not bundle multiple claims into
   one citation. There is no upper limit — long answers commonly cite 8–20
   passages. Under-citing is a worse failure than over-citing.
3. Write a small Python script (e.g. /tmp/highlight.py) that delegates ALL
   highlighting to the project library at {highlight_lib_dir}/highlight_lib.py.
   The library uses word-level coordinate matching, so it handles multi-line
   wraps, hyphenation, ligatures, and minor paraphrases automatically — you
   do NOT write matching code yourself.

   The script should be exactly this shape (fill in `passages`):

       import sys
       sys.path.insert(0, "{highlight_lib_dir}")
       from highlight_lib import highlight_pdf

       passages = [
           {{"page": <int>, "quote": "<verbatim text from PDF>"}},
           # ... one dict per passage ...
       ]
       highlight_pdf(
           input_pdf="{pdf_path}",
           output_pdf="{output_path}",
           citations_path="{citations_path}",
           passages=passages,
       )

   highlight_pdf writes the highlighted PDF and CITATIONS JSON. It also
   updates each citation's "quote" field to the actual text that was
   highlighted (in case it shrunk to a partial match), so the UI's chip and
   the yellow PDF highlight always agree.
4. Run the script with bash.
5. Verify both OUTPUT PDF and CITATIONS JSON exist.
6. Output the final answer.

SOURCE RESTRICTION — abstract is OFF-LIMITS (every mode):
- The pages text file marks the abstract with literal sentinel lines:
      [BEGIN ABSTRACT — DO NOT CITE FROM HERE]
      <abstract paragraphs>
      [END ABSTRACT]
- Any verbatim quote in your `passages` list MUST come from outside that
  region. Do not extract a passage whose words sit between those markers,
  even if it perfectly answers the question.
- The abstract exists for your context only. When a relevant claim appears
  in it, locate the same claim in the body — introduction, methodology,
  results, discussion, conclusion — and cite the body passage instead.
  Authors restate abstract claims in the body almost without exception.
- If you genuinely cannot locate the same claim outside the abstract, cite
  the closest related body passage rather than reaching into the abstract.
- If the markers are absent (rare — abstract not detectable), apply the
  same rule by inspection: skip whatever block of text serves as the
  paper's opening summary and cite the body version of any claim.

FORMATTING:
- The answer is rendered as Markdown. Use **bold** for key terms, bullet
  lists ("- item") when enumerating, and short headers ("## Heading") only
  when the answer has clearly distinct sections. Do not wrap the whole
  answer in a code block.
- Cite inline with "(p. N)" immediately after each claim. Multiple pages:
  "(pp. 4, 7)".
- No preamble ("Based on the PDF…", "The authors…"), no recap of the
  question, no closing summary.
- Do NOT print the citations list or the highlighted-PDF path — the
  harness surfaces those separately.

Work autonomously. Do not ask permission.
"""


MODE_GUIDANCE = {
    "auto": """\
MODE GUIDANCE — auto (adaptive, default):
Let the question dictate the answer's shape. If asked for "examples," give
a short list of examples. If asked "what is X," give the definition. If
asked to "compare X and Y," give a comparison. If asked a yes/no question,
lead with the answer. Don't impose a length — match the scope of what was
asked. Use markdown lists or short headers when the answer is naturally
enumerative or has distinct sections.

Cite every distinct claim with "(p. N)". No upper limit on citations.
Don't pad. Don't preface ("Based on the PDF…"). Don't summarize what you
just said.
""",
    "strict": """\
MODE GUIDANCE — strict (extractive):
Answer using ONLY what the PDF explicitly states. Do not add inference,
implication, commentary, or analysis beyond what the text directly says.
If the PDF doesn't answer the question, say so in one sentence and stop.

Length: as concise as possible while covering the question completely
(typically 2–5 sentences, or a short bullet list when enumerating items).
""",
    "brainstorm": """\
MODE GUIDANCE — brainstorm (analytical):
Answer using both the explicit content of the PDF AND your own analysis.
You may identify gaps, weaknesses, methodological issues, implicit
assumptions, missing comparisons, alternative interpretations, scope
limitations, and counterpoints the authors did not state outright.

GROUND EVERY CLAIM. For every inferred or analytical point, cite the
passage that PROMPTED the inference — e.g. "the study only evaluates
on Java projects (p. 4), which limits generalization to dynamic
languages." Make the inferential leap visible: when you go beyond the
text, say so ("the paper does not address X, but…", "this implies…").

Length: thorough. Use markdown bullets / sub-headers for structure.
Cite generously: most brainstorming answers cite 10+ distinct passages
covering both explicit support and the passages that anchor inferences.
""",
}


MODE_CHOICES = tuple(MODE_GUIDANCE.keys())
DEFAULT_MODE = "auto"
MAX_HISTORY_TURNS_IN_PROMPT = 10


def _format_history(history) -> str:
    """Render prior turns as a context block. Empty if no history."""
    if not history:
        return ""
    recent = history[-MAX_HISTORY_TURNS_IN_PROMPT:]
    parts = []
    for i, turn in enumerate(recent, 1):
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        parts.append(f"[Turn {i}]\nUser: {q}\nAssistant: {a}")
    body = "\n\n".join(parts)
    return (
        "\nPRIOR CONVERSATION (for context — the user may refer back to it):\n\n"
        + body
        + "\n"
    )


def _join_chunk(buf: str, chunk: str) -> str:
    """Append `chunk` to `buf`, inserting a space when streamed model chunks
    meet at a word boundary without whitespace (mirrors the JS heuristic so
    stored answers read naturally)."""
    if not buf or not chunk:
        return buf + chunk
    last, first = buf[-1], chunk[0]
    if last.isspace() or first.isspace():
        return buf + chunk
    if first in '.,;:!?)]"\'':
        return buf + chunk
    return buf + " " + chunk


_ABSTRACT_START = re.compile(r"^\s*(Abstract|ABSTRACT)\s*\.?\s*$", re.M)

# Patterns that mark the end of the abstract. We stop at whichever appears
# first — footnotes typically follow the abstract on the same page, before
# the actual Introduction section appears further down (often on page 2).
_ABSTRACT_END_PATTERNS = [
    re.compile(r"^\s*[∗*†‡§]", re.M),                                    # footnote markers
    re.compile(r"^(?:\s*\d+\.?\s+)?(?:Introduction|INTRODUCTION)\b", re.M),
    re.compile(r"^\s*\d+(?:st|nd|rd|th)\s+", re.M),                       # "31st Conference…"
    re.compile(r"^\s*arXiv:", re.M),
    re.compile(r"^=== PAGE ", re.M),                                      # page boundary fallback
]


def _mark_abstract(text: str) -> str:
    """Wrap the abstract section in BEGIN/END ABSTRACT markers so the agent
    can be instructed to never extract verbatim quotes from inside them.
    Returns the text unchanged if the abstract can't be located."""
    m_start = _ABSTRACT_START.search(text)
    if not m_start:
        return text
    body_start = m_start.end()
    end_pos = None
    for pat in _ABSTRACT_END_PATTERNS:
        m = pat.search(text, pos=body_start)
        if m and (end_pos is None or m.start() < end_pos):
            end_pos = m.start()
    if end_pos is None:
        return text
    abstract = text[body_start:end_pos].strip()
    # Sanity check — abstracts are typically 500–3000 chars.
    if len(abstract) < 100 or len(abstract) > 6000:
        return text
    return (
        text[:body_start]
        + "\n\n[BEGIN ABSTRACT — DO NOT CITE FROM HERE]\n"
        + abstract
        + "\n[END ABSTRACT]\n\n"
        + text[end_pos:]
    )


def extract_pages(pdf_path: Path) -> tuple[Path, str, int]:
    """Returns (path, full_text, page_count). Also writes the text to a
    sibling .pages.txt file for inspection."""
    out = pdf_path.with_suffix(".pages.txt")
    doc = pymupdf.open(pdf_path)
    parts = [f"=== PAGE {i+1} ===\n{page.get_text()}" for i, page in enumerate(doc)]
    page_count = len(parts)
    doc.close()
    text = _mark_abstract("\n\n".join(parts))
    # Force UTF-8: PDF text contains ligatures (ﬁ ﬂ), em-dashes, smart
    # quotes, etc. Windows' default cp1252 can't encode them.
    out.write_text(text, encoding="utf-8")
    return out, text, page_count


def _tool_label(block: ToolUseBlock) -> str:
    name = block.name
    inp = block.input or {}
    if name == "Bash":
        cmd = (inp.get("command") or "").splitlines()[0][:80]
        return f"bash: {cmd}"
    if name == "Read":
        return f"read: {inp.get('file_path', '?')}"
    if name == "Write":
        return f"write: {inp.get('file_path', '?')}"
    return name


# Aliases the Claude Code CLI accepts. "inherit" = use whatever the CLI is
# currently set to (e.g. via /model). Full model IDs also work.
MODEL_CHOICES = ("haiku", "sonnet", "opus", "inherit")
DEFAULT_MODEL = "haiku"


async def run_agent(
    pdf_path: Path,
    question: str,
    model: str = DEFAULT_MODEL,
    mode: str = DEFAULT_MODE,
    history: list | None = None,
):
    """Async generator yielding ('text', str) | ('tool', str) | ('done', dict).

    `history` is a list of {"question", "answer"} dicts from prior turns on
    this same PDF. It is rendered into a context block in the prompt so the
    model can answer follow-ups like "what about table 3?" or "remember what
    I asked first?".
    """
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages_text_path, pages_text, page_count = extract_pages(pdf_path)
    output_path = pdf_path.with_name(f"{pdf_path.stem}_highlighted.pdf")
    citations_path = pdf_path.with_name(f"{pdf_path.stem}_citations.json")

    yield (
        "stats",
        {
            "page_count": page_count,
            "char_count": len(pages_text),
            "pages_text_path": str(pages_text_path),
        },
    )

    if mode not in MODE_GUIDANCE:
        raise ValueError(f"unknown mode: {mode}")

    prompt = PROMPT_TEMPLATE.format(
        pdf_path=pdf_path,
        pages_text_path=pages_text_path,
        output_path=output_path,
        citations_path=citations_path,
        question=question,
        page_count=page_count,
        char_count=len(pages_text),
        mode=mode,
        mode_guidance=MODE_GUIDANCE[mode],
        highlight_lib_dir=str(Path(__file__).parent.resolve()),
        conversation_history=_format_history(history),
    )

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Bash", "Write"],
        permission_mode="bypassPermissions",
        cwd=str(pdf_path.parent),
        max_buffer_size=20 * 1024 * 1024,
        model=None if model == "inherit" else model,
    )

    # Mirror the client's "text resets on tool" logic so we can capture the
    # final post-last-tool text and return it on the done event for storage.
    final_answer = ""

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    final_answer = _join_chunk(final_answer, block.text)
                    yield ("text", block.text)
                elif isinstance(block, ToolUseBlock):
                    final_answer = ""  # was thinking-out-loud
                    yield ("tool", _tool_label(block))
        elif isinstance(message, ResultMessage):
            citations = []
            if citations_path.exists():
                try:
                    # Force UTF-8: citations.json is written with raw
                    # Unicode (ensure_ascii=False) so smart quotes / em-dashes
                    # / ligatures stay human-readable. Without an explicit
                    # encoding, Path.read_text falls back to cp1252 on
                    # Windows and throws on any non-ASCII char — which
                    # silently empties the citation list.
                    citations = json.loads(
                        citations_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    citations = []
            yield (
                "done",
                {
                    "cost_usd": getattr(message, "total_cost_usd", None),
                    "highlighted_pdf": str(output_path) if output_path.exists() else None,
                    "citations": citations,
                    "answer": final_answer,
                },
            )
