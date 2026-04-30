<img src="static/icon.png">

# sift

Ask questions about a PDF, get answers with **clickable, page-anchored
citations**, and see the source passages **highlighted** in a side-by-side
PDF viewer.

Built on the [Claude Agent SDK][sdk] and [PyMuPDF][pymupdf]. Authentication
piggybacks on your local [Claude Code][cc] session — no API keys to manage.

## Features

- **Two-pane web UI** — PDF on the left, chat on the right
- **Color-matched citations** — each citation gets a distinct pastel
  color, applied to the PDF highlight, the citation chip's left border,
  and the inline `(p. N)` pill in the answer text. When several citations
  land on the same page, the colors tell you which highlight goes with
  which chip at a glance.
- **Click-to-jump** — both inline `(p. 4)` references and the citation
  chips scroll the PDF to the right page on click.
- **Three answer modes**
  - `auto` (default): adapts to whatever the question asks for — list,
    table, paragraph, single sentence. No imposed length.
  - `strict`: only what the PDF explicitly says, no inference.
  - `brainstorm`: surface gaps, weaknesses, and inferences — every claim
    still anchored to a real passage.
- **Per-PDF chat memory** — follow-up questions like "make that more
  concise" or "what did I ask first?" work because each turn sees the
  prior conversation. A clear button wipes history; turns rehydrate when
  you switch back to a PDF.
- **Abstract is off-limits** — `extract_pages` wraps the abstract in
  explicit `[BEGIN ABSTRACT]` / `[END ABSTRACT]` markers and the prompt
  forbids citing inside them, forcing the model to anchor claims in the
  body where they're elaborated.
- **Markdown answers** with bold, lists, headings, and inline code.
- **Activity trail** — collapsed `Thought for Ns · K steps` per turn
  shows the agent's reasoning and tool calls.
- **Robust word-coordinate highlighting** — multi-line wraps, hyphenated
  breaks, and minor paraphrases all match. Cross-page fallback: if the
  cited page misses, every other page is scanned and the longest match
  wins, so off-by-one page numbers from the model self-heal.
- **Model selector** — Haiku 4.5 (fast/cheap default), Sonnet 4.6,
  Opus 4.7, or whatever your CLI's `/model` is set to (`inherit`).
- **CLI** — same agent, headless:
  `python pdf_qa.py paper.pdf "question" [--mode auto|strict|brainstorm]`.

## Prerequisites

- **Python 3.10 or newer.** The Claude Agent SDK requires it. If
  `pip install` reports `No matching distribution found for
  claude-agent-sdk`, your Python is too old — check with
  `python --version` and install a newer one
  ([python.org](https://www.python.org/downloads/)).
- [Claude Code CLI][cc] installed and signed in once interactively
  (`claude` from a terminal). The Agent SDK spawns the `claude` CLI as a
  subprocess and inherits whatever it's authenticated with — works with a
  Claude Pro/Max subscription (OAuth) or an `ANTHROPIC_API_KEY` configured
  via the CLI.

  Where Claude Code stores its credentials per OS:

  | OS | Path |
  |---|---|
  | macOS / Linux | `~/.claude/` |
  | Windows | `%USERPROFILE%\.claude\` (`C:\Users\<you>\.claude\`) |

  Nothing in this repo touches that directory; auth lives entirely outside
  the project, so cloning is enough — no `.env` files, no key configuration.

## Setup

### macOS / Linux

```bash
git clone https://github.com/<you>/sift.git
cd sift

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If `python3` resolves to an older interpreter, run the venv step with the
specific binary, e.g. `python3.12 -m venv venv`.

### Windows (PowerShell)

```powershell
git clone https://github.com/<you>/sift.git
cd sift

python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script with an execution-policy error,
run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
Use `venv\Scripts\activate.bat` from cmd.exe instead of PowerShell.

## Run

### Web UI

```bash
uvicorn app:app --port 8000
```

Open http://localhost:8000. Upload a PDF, ask a question, click any
`(p. N)` reference or citation chip to jump.

### CLI

```bash
python pdf_qa.py path/to/paper.pdf "What is the main contribution?"
python pdf_qa.py --model sonnet --mode brainstorm paper.pdf "Limitations?"
```

Outputs: `paper_highlighted.pdf` and `paper_citations.json` next to the
input PDF.

## How it works

1. PyMuPDF extracts per-page text into `<paper>.pages.txt`.
2. The agent reads that file, identifies passages that ground each claim,
   and writes a tiny script that calls
   `highlight_lib.highlight_pdf(input, output, citations, passages)`.
3. `highlight_lib` matches each passage at the **word-coordinate level**
   (it pulls every word's bounding box via `page.get_text("words")` and
   finds the longest contiguous matching run vs the quote, normalized
   lowercase + alphanumeric). Word-level matching survives anything
   `search_for` chokes on: line wraps, hyphenated breaks, ligatures.
4. Citations JSON records the **actual highlighted text** plus a per-citation
   pastel color, so the chip's border, the inline `(p. N)` pill, and the
   yellow PDF region always agree.
5. **Per-PDF chat memory** lives in an in-process dict (`CHATS`) keyed by
   filename. Each `/ask` prepends the last 10 turns to the prompt as a
   `PRIOR CONVERSATION` block, so the agent can answer follow-ups that
   reference earlier turns. Memory is volatile (lost on uvicorn restart);
   add a JSON dump in `app.py` if you want persistence.

## Project layout

```
.
├── app.py             FastAPI server (web UI + SSE streaming)
├── agent_core.py      Shared agent setup and prompt for CLI + web
├── highlight_lib.py   Word-coordinate highlighting library
├── pdf_qa.py          CLI entrypoint
├── static/index.html  Two-pane UI (vanilla JS, no build step)
├── pdfs/              User PDFs and generated artifacts (gitignored)
└── requirements.txt
```

## Configuration

Endpoints (`app.py`):

- `GET  /`                  static UI
- `GET  /config`            model + mode choices
- `POST /upload`            multipart PDF upload
- `GET  /pdfs`              list uploaded PDFs
- `GET  /pdf/{id}`          serve PDF (`?highlighted=true` for annotated copy)
- `GET  /history/{id}`      per-PDF chat history for rehydration on reload
- `POST /clear/{id}`        wipe chat history for a PDF
- `POST /ask`               SSE stream: `stats`, `text`, `tool`, `done`, `error`

Per-turn options sent to `/ask`:
`{ file_id, question, model: "haiku|sonnet|opus|inherit", mode: "auto|strict|brainstorm" }`.

The defaults (`DEFAULT_MODEL`, `DEFAULT_MODE`) and per-PDF turn cap
(`MAX_TURNS_KEPT`) live in `agent_core.py` and `app.py` respectively.

## Notes

- `max_buffer_size` is set to 20 MB on the SDK transport because the `Read`
  tool returns large JSON payloads when invoked on big files. Don't lower it.
- The `Read` tool is intentionally pointed at the extracted `.pages.txt`,
  never at the PDF directly — invoking `Read` on a PDF returns each page as
  base64 image data and immediately blows past any sane buffer.

[sdk]: https://github.com/anthropics/claude-agent-sdk-python
[pymupdf]: https://pymupdf.io/
[cc]: https://docs.claude.com/en/docs/claude-code
