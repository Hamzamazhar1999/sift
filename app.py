"""FastAPI server: PDF on the left, Claude Q&A with citations on the right.

Run:
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000.
"""
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
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


@app.post("/ask")
async def ask(body: AskBody):
    safe = Path(body.file_id).name
    pdf_path = PDF_DIR / safe
    if not pdf_path.exists():
        raise HTTPException(404)
    if body.model not in MODEL_CHOICES:
        raise HTTPException(400, f"Unknown model: {body.model}")
    if body.mode not in MODE_CHOICES:
        raise HTTPException(400, f"Unknown mode: {body.mode}")

    async def stream():
        try:
            async for kind, payload in run_agent(
                pdf_path, body.question, model=body.model, mode=body.mode
            ):
                yield f"data: {json.dumps({'type': kind, 'data': payload})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
