"""FastAPI server: PDF on the left, Claude Q&A with citations on the right.

Run:
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000.
"""
import hashlib
import json
import re
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_core import (
    DEFAULT_MODE,
    DEFAULT_MODEL,
    MODE_CHOICES,
    MODEL_CHOICES,
    run_agent,
)

ROOT = Path(__file__).parent.resolve()
PDF_DIR = ROOT / "pdfs"
PDF_DIR.mkdir(exist_ok=True)
STATIC_DIR = ROOT / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-process per-PDF chat memory: file_id -> [{question, answer, citations}].
# Volatile (lost on restart) — fine for v1; add disk persistence later if
# desired by writing to pdfs/<stem>_history.json.
CHATS: dict[str, list[dict]] = {}
MAX_TURNS_KEPT = 20  # cap stored turns per file to keep prompts bounded


def _safe_id(file_id: str) -> str:
    return Path(file_id).name


@app.get("/")
async def index():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Must be a .pdf file")
    safe = Path(file.filename).name
    dest = PDF_DIR / safe
    dest.write_bytes(await file.read())
    return {"file_id": safe}


_HIGHLIGHTED_SUFFIX = re.compile(r"_highlighted(?:_t\d+(?:_[a-f0-9]+)?)?$")


def _citations_hash(citations: list) -> str:
    """Stable short hash of a turn's citations. Used as a cache key so the
    per-turn highlighted PDF on disk is naturally invalidated whenever the
    underlying citations change (e.g. across uvicorn restarts where CHATS
    is wiped but old cache files survive)."""
    blob = json.dumps(citations, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]


@app.get("/pdfs")
async def list_pdfs():
    return {
        "files": sorted(
            p.name
            for p in PDF_DIR.glob("*.pdf")
            if not _HIGHLIGHTED_SUFFIX.search(p.stem)
        )
    }


@app.get("/pdf/{file_id}")
async def get_pdf(file_id: str, highlighted: bool = False, turn: int | None = None):
    """Serve the original PDF, the latest highlighted PDF, or a per-turn
    re-render of an arbitrary past turn's highlights.

    `turn` (optional) — index into CHATS[file_id]. When set, regenerate
    the highlighted PDF using THAT turn's stored citations and serve it.
    Cached on disk as `<stem>_highlighted_t{turn}_{hash}.pdf` where the
    hash is derived from the citations content — so the cache key
    naturally invalidates whenever content changes (across uvicorn
    restarts, CHATS is wiped but disk caches survive; without a content
    hash a new turn at index 0 would inherit a stale t0 file).
    """
    safe = Path(file_id).name
    src = PDF_DIR / safe
    if not src.exists():
        raise HTTPException(404, f"Not found: {safe}")

    if turn is not None:
        chats = CHATS.get(safe, [])
        if turn < 0 or turn >= len(chats):
            raise HTTPException(404, f"Turn {turn} not found")
        cits = chats[turn].get("citations") or []
        h = _citations_hash(cits)
        out = PDF_DIR / f"{Path(safe).stem}_highlighted_t{turn}_{h}.pdf"
        if not out.exists():
            from highlight_lib import re_highlight_from_citations
            re_highlight_from_citations(str(src), str(out), cits)
        return FileResponse(out, media_type="application/pdf")

    if highlighted:
        path = PDF_DIR / f"{Path(safe).stem}_highlighted.pdf"
        if not path.exists():
            raise HTTPException(404, f"Not found: {path.name}")
        return FileResponse(path, media_type="application/pdf")

    return FileResponse(src, media_type="application/pdf")


class AskBody(BaseModel):
    file_id: str
    question: str
    model: str = DEFAULT_MODEL
    mode: str = DEFAULT_MODE


@app.get("/config")
async def config():
    return {
        "models": {"choices": list(MODEL_CHOICES), "default": DEFAULT_MODEL},
        "modes": {"choices": list(MODE_CHOICES), "default": DEFAULT_MODE},
    }


# kept for backwards compat with the older client
@app.get("/models")
async def models():
    return {"choices": list(MODEL_CHOICES), "default": DEFAULT_MODEL}


@app.get("/history/{file_id}")
async def get_history(file_id: str):
    """Return the stored chat for a PDF so the UI can rehydrate after a
    page reload."""
    return {"turns": CHATS.get(_safe_id(file_id), [])}


@app.post("/clear/{file_id}")
async def clear_history(file_id: str):
    """Wipe stored chat for a PDF and remove any per-turn cached PDFs."""
    safe = _safe_id(file_id)
    CHATS.pop(safe, None)
    stem = Path(safe).stem
    for cached in PDF_DIR.glob(f"{stem}_highlighted_t*.pdf"):
        try:
            cached.unlink()
        except OSError:
            pass
    return {"ok": True}


@app.post("/ask")
async def ask(body: AskBody):
    safe = _safe_id(body.file_id)
    pdf_path = PDF_DIR / safe
    if not pdf_path.exists():
        raise HTTPException(404)
    if body.model not in MODEL_CHOICES:
        raise HTTPException(400, f"Unknown model: {body.model}")
    if body.mode not in MODE_CHOICES:
        raise HTTPException(400, f"Unknown mode: {body.mode}")

    history = list(CHATS.get(safe, []))

    async def stream():
        final_answer = ""
        final_citations: list = []
        try:
            async for kind, payload in run_agent(
                pdf_path,
                body.question,
                model=body.model,
                mode=body.mode,
                history=history,
            ):
                if kind == "done":
                    final_answer = payload.get("answer") or ""
                    final_citations = payload.get("citations") or []
                yield f"data: {json.dumps({'type': kind, 'data': payload})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            return

        # Persist the turn only if the run produced an answer.
        if final_answer.strip():
            turns = CHATS.setdefault(safe, [])
            turns.append(
                {
                    "question": body.question,
                    "answer": final_answer,
                    "citations": final_citations,
                }
            )
            del turns[:-MAX_TURNS_KEPT]

    return StreamingResponse(stream(), media_type="text/event-stream")
