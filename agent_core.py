"""Shared agent logic for the CLI and the web app.
"""
import json
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
DEFAULT_MODE = "strict"


def extract_pages(pdf_path: Path) -> tuple[Path, str, int]:
    """Returns (path, full_text, page_count). Also writes the text to a
    sibling .pages.txt file for inspection."""
    out = pdf_path.with_suffix(".pages.txt")
    doc = pymupdf.open(pdf_path)
    parts = [f"=== PAGE {i+1} ===\n{page.get_text()}" for i, page in enumerate(doc)]
    page_count = len(parts)
    doc.close()
    text = "\n\n".join(parts)
    out.write_text(text)
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
):
    """Async generator yielding ('text', str) | ('tool', str) | ('done', dict)."""
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
    )

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Bash", "Write"],
        permission_mode="bypassPermissions",
        cwd=str(pdf_path.parent),
        max_buffer_size=20 * 1024 * 1024,
        model=None if model == "inherit" else model,
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            # Emit every text/tool block. The client treats text as ephemeral
            # until no more tools fire — any text shown before a tool event
            # gets discarded and the "thinking" indicator returns. Only the
            # text after the final tool sticks as the answer.
            for block in message.content:
                if isinstance(block, TextBlock):
                    yield ("text", block.text)
                elif isinstance(block, ToolUseBlock):
                    yield ("tool", _tool_label(block))
        elif isinstance(message, ResultMessage):
            citations = []
            if citations_path.exists():
                try:
                    citations = json.loads(citations_path.read_text())
                except Exception:
                    citations = []
            yield (
                "done",
                {
                    "cost_usd": getattr(message, "total_cost_usd", None),
                    "highlighted_pdf": str(output_path) if output_path.exists() else None,
                    "citations": citations,
                },
            )
