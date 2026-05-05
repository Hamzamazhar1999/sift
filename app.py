"""FastAPI server: PDF on the left, Claude Q&A with citations on the right.

Run:
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000.
"""
import json
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


@app.get("/pdfs")
async def list_pdfs():
    return {
        "files": sorted(
            p.name for p in PDF_DIR.glob("*.pdf") if not p.stem.endswith("_highlighted")
        )
    }


@app.get("/pdf/{file_id}")
async def get_pdf(file_id: str, highlighted: bool = False):
    safe = Path(file_id).name
    path = (
        PDF_DIR / f"{Path(safe).stem}_highlighted.pdf" if highlighted else PDF_DIR / safe
    )
    if not path.exists():
        raise HTTPException(404, f"Not found: {path.name}")
    return FileResponse(path, media_type="application/pdf")


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
    """Wipe stored chat for a PDF."""
    CHATS.pop(_safe_id(file_id), None)
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
