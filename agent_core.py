"""Shared agent logic for the CLI and the web app.
"""
import json
import re
import shutil
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

SOURCE RESTRICTION — what to do when the answer is NOT in the paper:
- If the paper does not address the question, lead with one short sentence
  saying so explicitly — phrasings like "Not addressed in this paper",
  "The paper does not discuss X", or "X is not covered." Then stop.
- Return an EMPTY `passages` list in that case. Do not cite tangentially
  related sentences as if they answered the question. Citing tangential
  content is worse than saying the paper doesn't cover the topic.
- The user prefers a clean "no" over a fabricated "yes."

SECTION ROUTING — cite from the right part of the paper:
- The pages text file marks each section with sentinel lines:
      [SECTION: <name> BEGIN]
      ... body ...
      [SECTION: <name> END]
- Use the section a passage lives in to decide whether it is appropriate
  for the question. In particular:
  · Empirical questions ("what results / accuracy / performance / findings"):
    cite from Results / Evaluation / Experiments / Findings / Discussion.
    DO NOT cite
    from Related Work or Background — those sentences describe what OTHER
    authors reported, not what THIS paper found.
  · Methodology questions ("how does it work / what approach"):
    cite from Methods / Approach / Implementation / Experimental Setup.
  · Prior-work questions ("what does prior work do / how does this differ"):
    cite from Related Work / Background — these are appropriate here.
  · Threats / limitations questions: cite from Threats to Validity /
    Limitations / Discussion.
- If a section is unrecognised (kept as a literal title rather than a
  canonical label), use the title and your judgement.

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
- INLINE REFS ↔ PASSAGES LIST MUST CORRESPOND BIDIRECTIONALLY:
  · Every inline "(p. N)" must have a matching entry in `passages` with
    that page number. Don't write "(p. 14)" unless `passages` includes
    a passage with page=14.
  · Every entry in `passages` must be referenced inline at least once
    via "(p. N)". Don't include a passage in the list if you don't
    intend to cite it in the prose — that produces a citation chip in
    the UI that makes no sense to the reader.
  Orphans either direction (inline ref with no passage, or passage with
  no inline ref) are wrong. If you can't find a passage to back a claim,
  drop the cite from the prose. If you have a passage you don't end up
  citing, drop it from `passages`.
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
    "freehand": """\
MODE GUIDANCE — freehand (collaborative, user-driven):

The user's prompt IS the spec. Treat them as a colleague handing you a
task and a paper. Match whatever shape, length, format, and tone the
user's request implies — fill a rubric, draft a section, sketch a
hypothesis, write a critique, generate a comparison table, free-form
discussion, anything. Do not impose your own format.

THINK FOR YOURSELF. Synthesize across passages. Draw connections the
authors didn't make explicit. Apply the paper's framework to a new
setting if the user asks. Identify what the paper implies even when
not stated outright. Generate hypotheses and infer consequences. Be
generous with analytical commentary, framing, and meta-discussion.

THE ONE HARD RULE — grounding:
Every factual claim ABOUT THE PAPER'S CONTENT must be anchored to a
citable passage. Inferences are not only allowed but encouraged — just
make the inferential step visible:
  "the authors evaluate only on Java (p. 4), which suggests their
   findings may not generalize to dynamic typing"
  "the paper does not address X directly, but the framework in §3
   (pp. 5–6) implies…"
  "extending their argument, …"
Analytical commentary, structural framing ("here's a draft you could
use:"), and meta-discussion do NOT need citations — they are yours,
not the paper's.

NEVER state as fact what the paper does not establish; mark inferences
with "suggests", "implies", "we might expect", "extending this",
"the paper does not say so, but…".

OVERRIDES:
This mode supersedes the FORMATTING block below for length, preamble,
closing remarks, headers, and structure — the user's request governs
all of those. The only formatting constraints that still hold:
- cite inline with "(p. N)" after each claim grounded in a passage
- do NOT manually print the citations list (the UI surfaces it)
- do NOT cite the abstract (per SOURCE RESTRICTION above)
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


# Match the "Abstract" header in either form papers use:
#   Abstract            (alone on its own line, followed by the body below)
#   Abstract—Large...   (inline, with the body starting after a separator)
# Separators we tolerate: whitespace, em-dash, en-dash, hyphen, colon, period.
_ABSTRACT_START = re.compile(r"^\s*(?:Abstract|ABSTRACT)\b[\s.:\-–—]*", re.M)

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


# Section-header detection. Papers use several shapes; we try each, collect
# all (pos, title) candidates, then carve the document into labelled sections.
# Detection runs AFTER abstract marking so the abstract block isn't re-wrapped.
_CANONICAL_NAMES = (
    r"Introduction|Background|Related Work|Related Works?|Prior Work|"
    r"Literature Review|Methodology|Methods?|Method|Approach|Implementation|"
    r"Experimental Setup|Experiments?|Evaluation|Results?|Findings?|Analysis|"
    r"Discussion|Threats to Validity|Limitations?|Future Work|Conclusions?|"
    r"Acknowledg(?:e)?ments?|References"
)
_SECTION_PATTERNS = [
    # "II.\nLARGE LANGUAGE MODELS" or "4\nOUR METHOD"  (Roman/Arabic prefix,
    # ALL-CAPS title on the next line, must be multi-word so single-token
    # table cells like "GPT-3" / "MAWPS" don't trip it).
    re.compile(
        r"^(?:[IVXLCDM]+|\d+)\.?\s*\n"
        r"(?P<title>[A-Z][A-Z0-9,&'’\-]*\s+[A-Z][A-Z 0-9,&'’\-]{1,60})\s*$",
        re.M,
    ),
    # "II. LARGE LANGUAGE MODELS"  (Roman/Arabic + ALL-CAPS title same line)
    re.compile(
        r"^(?:[IVXLCDM]+|\d+)\.?\s+"
        r"(?P<title>[A-Z][A-Z0-9,&'’\-]*\s+[A-Z][A-Z 0-9,&'’\-]{1,60})\s*$",
        re.M,
    ),
    # "1 Introduction" / "2. Related Work"  (numbered, must be a canonical name)
    re.compile(
        r"^\s*\d+\.?\s+(?P<title>" + _CANONICAL_NAMES + r")\s*$",
        re.M | re.I,
    ),
    # Standalone canonical name on its own line
    re.compile(
        r"^\s*(?P<title>" + _CANONICAL_NAMES + r")\s*$",
        re.M | re.I,
    ),
]

# A few canonical names ALSO appear as table column headers in papers
# (LoRA's "Method" column is the motivating case). For those — and only
# those — we drop the match unless the following line is real prose.
_AMBIGUOUS_HEADERS = {"method", "methods", "methodology", "approach"}


def _looks_like_table_cell(title: str) -> bool:
    """Reject multi-word titles that are obviously benchmark-table content
    ('ROC AUC', 'OOM OOM', 'BLOOM-175B A100-80GB', 'IN IN-V2'). Two cheap
    rules cover them all without losing legitimate titles like
    'OUR METHOD' or 'DECOUPLING THE WEIGHT DECAY …':
    1. fewer than 8 alphabetic characters overall, or
    2. any word with ≥30% digit characters (i.e. model/GPU names)."""
    words = title.split()
    if len(words) < 2:
        return False  # single-word already filtered upstream
    if sum(c.isalpha() for c in title) < 8:
        return True
    for w in words:
        alnum = sum(c.isalnum() for c in w)
        digits = sum(c.isdigit() for c in w)
        if alnum and digits / alnum > 0.3:
            return True
    return False


def _has_prose_followup(text: str, pos: int, min_chars: int = 40) -> bool:
    """Return True if the next non-blank line after pos is substantial prose.
    Table cells (column headers' siblings) tend to be short; section bodies
    open with a paragraph."""
    for line in text[pos : pos + 2000].splitlines():
        s = line.strip()
        if not s:
            continue
        return len(s) >= min_chars
    return False

# Map detected raw titles to a small set of canonical labels so the prompt
# rule can route by intent. Anything else is preserved verbatim.
_CANONICAL = {
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "related works": "Related Work",
    "prior work": "Related Work",
    "literature review": "Related Work",
    "methodology": "Methods",
    "methods": "Methods",
    "method": "Methods",
    "approach": "Methods",
    "implementation": "Methods",
    "experimental setup": "Methods",
    "experiments": "Experiments",
    "experiment": "Experiments",
    "evaluation": "Evaluation",
    "results": "Results",
    "result": "Results",
    "findings": "Results",
    "finding": "Results",
    "analysis": "Results",
    "discussion": "Discussion",
    "threats to validity": "Threats to Validity",
    "limitations": "Limitations",
    "limitation": "Limitations",
    "future work": "Future Work",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "acknowledgments": "Acknowledgments",
    "acknowledgements": "Acknowledgments",
    "references": "References",
}


def _canonical_section(title: str) -> str:
    return _CANONICAL.get(title.strip().lower(), title.strip())


def _mark_sections(text: str) -> str:
    """Wrap each detected section's body in BEGIN/END SECTION sentinels so
    the prompt rule can route citations by section. Returns text unchanged
    if no section starts are found."""
    starts = []  # list of (header_end_pos, canonical_name)
    for pat in _SECTION_PATTERNS:
        for m in pat.finditer(text):
            title = m.group("title").strip()
            if _looks_like_table_cell(title):
                continue
            raw = title.lower()
            if raw in _AMBIGUOUS_HEADERS and not _has_prose_followup(text, m.end()):
                continue
            starts.append((m.end(), _canonical_section(title)))
    if not starts:
        return text
    starts.sort()
    # Dedupe: skip adjacent positions and skip repeats of the same canonical
    # name (a paper's roadmap paragraph re-mentioning section titles, or
    # near-identical regex matches on the same header).
    deduped = []
    seen_names = set()
    for pos, name in starts:
        if deduped and pos - deduped[-1][0] < 5:
            continue
        if name in seen_names:
            continue
        deduped.append((pos, name))
        seen_names.add(name)
    pieces = []
    last_pos = 0
    for i, (pos, name) in enumerate(deduped):
        body_start = pos
        body_end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        # Find header start so we can preserve the prologue verbatim
        # (everything before THIS section header's body_start that came after
        # the previous section's body_end).
        prologue_end = body_start
        # Heuristic: the header is on the line(s) immediately before body_start.
        # We want to keep it visible, so wrap from body_start (just after header).
        if i == 0:
            pieces.append(text[last_pos:body_start])
        body = text[body_start:body_end]
        # Skip very short sections — likely TOC/list items, not real bodies
        if len(body.strip()) < 200:
            pieces.append(body)
            last_pos = body_end
            continue
        pieces.append(
            "\n[SECTION: " + name + " BEGIN]\n"
            + body.strip("\n")
            + "\n[SECTION: " + name + " END]\n"
        )
        last_pos = body_end
    pieces.append(text[last_pos:])
    return "".join(pieces)


def extract_pages(pdf_path: Path) -> tuple[Path, str, int]:
    """Returns (path, full_text, page_count). Also writes the text to a
    sibling .pages.txt file for inspection."""
    out = pdf_path.with_suffix(".pages.txt")
    doc = pymupdf.open(pdf_path)
    parts = [f"=== PAGE {i+1} ===\n{page.get_text()}" for i, page in enumerate(doc)]
    page_count = len(parts)
    doc.close()
    text = _mark_sections(_mark_abstract("\n\n".join(parts)))
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


# Backend registry. Each entry describes a CLI that implements Claude
# Code's stream-json transport. The Agent SDK spawns whichever one is
# selected per turn; the rest of sift stays backend-agnostic. To add a
# new backend, add an entry here — no branching elsewhere in the code.
#
# Fields:
#   display        — human-readable label for the UI dropdown
#   binary         — name of the CLI to spawn (looked up via shutil.which).
#                    None lets the Agent SDK do its own discovery,
#                    useful when the binary lives in an installer-managed
#                    location not on PATH (e.g. Anthropic's claude).
#   models         — model aliases this backend's CLI accepts
#   default_model  — sift's default when this backend is picked
#
# `inherit` means: let the CLI use whatever default it was configured with.
BACKENDS = {
    "claude": {
        "display": "Claude Code (Anthropic)",
        "binary": None,
        "models": ["haiku", "sonnet", "opus", "inherit"],
        "default_model": "haiku",
    },
    "openclaude": {
        "display": "OpenClaude (OpenAI / Gemini / OpenRouter / Ollama)",
        "binary": "openclaude",
        "models": ["inherit"],
        "default_model": "inherit",
    },
}
BACKEND_CHOICES = tuple(BACKENDS.keys())
DEFAULT_BACKEND = "claude"

# Union of every backend's accepted model aliases — used purely for
# /ask body validation. The chosen CLI does the real check at spawn time.
MODEL_CHOICES = tuple(
    dict.fromkeys(m for b in BACKENDS.values() for m in b["models"])
)
DEFAULT_MODEL = BACKENDS[DEFAULT_BACKEND]["default_model"]


def _resolve_cli_path(backend: str) -> str | None:
    """Look up the CLI binary for a backend. Returns None when the entry
    has no explicit binary (the SDK handles its own discovery in that
    case)."""
    binary = BACKENDS[backend].get("binary")
    if binary is None:
        return None
    path = shutil.which(binary)
    if not path:
        raise FileNotFoundError(
            f"backend `{backend}` expects the `{binary}` CLI on PATH "
            f"but couldn't find it. Install it and ensure `{binary}` "
            f"runs in your shell."
        )
    return path


async def run_agent(
    pdf_path: Path,
    question: str,
    model: str = DEFAULT_MODEL,
    mode: str = DEFAULT_MODE,
    history: list | None = None,
    backend: str = DEFAULT_BACKEND,
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

    cli_path = _resolve_cli_path(backend)
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Bash", "Write"],
        permission_mode="bypassPermissions",
        cwd=str(pdf_path.parent),
        max_buffer_size=20 * 1024 * 1024,
        model=None if model == "inherit" else model,
        cli_path=cli_path,
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
