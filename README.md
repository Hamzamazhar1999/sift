# sift

Ask questions about a PDF, get answers with **clickable, page-anchored
citations**, and see the source passages **highlighted** in a side-by-side
PDF viewer.

Built on the [Claude Agent SDK][sdk] and [PyMuPDF][pymupdf]. Authentication
piggybacks on your local [Claude Code][cc] session — no API keys to manage.

## Features

- **Two-pane web UI** — PDF on the left, chat on the right
- **Click-to-jump citations** — both inline `(p. 4)` references and the
  citation chips below the answer; the PDF viewer scrolls to the page and
  the cited passages are highlighted in yellow
- **Two answer modes**
  - `strict` (default): only what the PDF explicitly says
  - `brainstorm`: surface gaps, weaknesses, and inferences — every claim
    still anchored to a real passage
- **Markdown answers** with bold, lists, headings, and inline code
- **Activity trail** — collapsed `Thought for Ns · K steps` per turn shows
  the agent's reasoning and tool calls
- **Robust word-coordinate highlighting** — multi-line wraps, hyphenation
  across line breaks, and minor paraphrases all match
- **Model selector** — Haiku 4.5 (fast/cheap default), Sonnet 4.6, Opus 4.7,
  or whatever your CLI's `/model` is set to (`inherit`)
- **CLI** — same agent, headless: `python pdf_qa.py paper.pdf "question"`

## Prerequisites

- Python **3.10+**
- [Claude Code CLI][cc] installed and signed in once interactively
  (`claude` from a terminal). The Agent SDK piggybacks on this auth — works
  with a Claude Pro/Max subscription or an API key configured via the CLI.

## Setup

```bash
git clone <your-fork-url> sift
cd sift

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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
4. Citations JSON records the **actual highlighted text** so the chip and
   the yellow PDF region always agree.

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

- `GET  /`           static UI
- `GET  /config`     model + mode choices
- `POST /upload`     multipart PDF upload
- `GET  /pdfs`       list uploaded PDFs
- `GET  /pdf/{id}`   serve PDF (`?highlighted=true` for annotated copy)
- `POST /ask`        SSE stream: `stats`, `text`, `tool`, `done`, `error`

Per-turn options sent to `/ask`:
`{ file_id, question, model: "haiku|sonnet|opus|inherit", mode: "strict|brainstorm" }`.

The default model and mode are defined in `agent_core.py` (`DEFAULT_MODEL`,
`DEFAULT_MODE`).

## Notes

- `max_buffer_size` is set to 20 MB on the SDK transport because the `Read`
  tool returns large JSON payloads when invoked on big files. Don't lower it.
- The `Read` tool is intentionally pointed at the extracted `.pages.txt`,
  never at the PDF directly — invoking `Read` on a PDF returns each page as
  base64 image data and immediately blows past any sane buffer.

[sdk]: https://github.com/anthropics/claude-agent-sdk-python
[pymupdf]: https://pymupdf.io/
[cc]: https://docs.claude.com/en/docs/claude-code
