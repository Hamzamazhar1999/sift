"""FastAPI server: PDF on the left, Claude Q&A with citations on the right.

Run:
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000.
"""
import asyncio
import hashlib
import json
import re
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_core import (
    BACKEND_CHOICES,
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_MODE,
    DEFAULT_MODEL,
    MODE_CHOICES,
    MODEL_CHOICES,
    _build_global_citations,
    aggregate_answers,
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

# Server-side answer cache keyed by an exact-input hash. Hits replay the
# stats/text/done events without spawning the agent — that's where the
# token savings come from. Cache key changes whenever any of the inputs
# the agent actually saw change (PDF content via mtime, question wording,
# mode, model, backend, conversation history). Volatile, capped FIFO.
ASK_CACHE: dict[str, dict] = {}
ASK_CACHE_MAX = 256

# /ask-multi runs (one per "Ask selected" click in the UI). Each run owns
# a short id and per-file citation lists, so the client can later request
# /pdf/<file>?cmp=<run_id> to view that specific card's highlights — the
# answers themselves aren't persisted to CHATS, but their highlights still
# need a stable server-side source to render from.
COMPARE_RUNS: dict[str, dict[str, list]] = {}
COMPARE_RUNS_MAX = 32

# Cache for the cross-paper synthesis so repeating an identical comparison
# (same question + same per-PDF answers + model/backend) doesn't pay for the
# aggregation LLM call again. Keyed by a hash of those inputs.
AGG_CACHE: dict[str, dict] = {}
AGG_CACHE_MAX = 128


def _safe_id(file_id: str) -> str:
    return Path(file_id).name


def _new_run_id() -> str:
    import secrets
    return secrets.token_urlsafe(8)


def _compare_runs_put(run_id: str, file_id: str, citations: list) -> None:
    bucket = COMPARE_RUNS.setdefault(run_id, {})
    bucket[file_id] = citations
    # FIFO eviction if too many runs accumulate.
    while len(COMPARE_RUNS) > COMPARE_RUNS_MAX:
        oldest = next(iter(COMPARE_RUNS))
        if oldest == run_id:  # don't evict the run we just added to
            break
        COMPARE_RUNS.pop(oldest)


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
async def get_pdf(
    file_id: str, highlighted: bool = False,
    turn: int | None = None, cmp: str | None = None,
):
    """Serve the original PDF, the latest highlighted PDF, a per-turn
    re-render of past CHATS citations, OR a per-compare-run re-render
    of /ask-multi citations.

    `turn` (optional) — index into CHATS[file_id]. Regenerates highlights
    from that turn's stored citations.
    `cmp`  (optional) — run id from /ask-multi. Regenerates highlights
    from COMPARE_RUNS[cmp][file_id]. Compare answers are deliberately
    NOT persisted into CHATS (so they don't pollute conversation history),
    so this is a parallel path with its own on-disk cache.

    Both paths cache the rendered PDF on disk keyed by a content hash, so
    repeated views are instant and cache files survive uvicorn restarts.
    """
    safe = Path(file_id).name
    src = PDF_DIR / safe
    if not src.exists():
        raise HTTPException(404, f"Not found: {safe}")

    if cmp is not None:
        run = COMPARE_RUNS.get(cmp)
        if run is None or safe not in run:
            raise HTTPException(404, f"Compare run {cmp} for {safe} not found")
        cits = run[safe] or []
        h = _citations_hash(cits)
        # Sanitize run id for the filename — only allow url-safe chars.
        safe_run = re.sub(r"[^A-Za-z0-9_\-]", "", cmp)[:16] or "x"
        out = PDF_DIR / f"{Path(safe).stem}_highlighted_cmp_{safe_run}_{h}.pdf"
        if not out.exists():
            from highlight_lib import re_highlight_from_citations
            re_highlight_from_citations(str(src), str(out), cits)
        return FileResponse(out, media_type="application/pdf")

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
    backend: str = DEFAULT_BACKEND


@app.get("/config")
async def config():
    return {
        "models": {"choices": list(MODEL_CHOICES), "default": DEFAULT_MODEL},
        "modes": {"choices": list(MODE_CHOICES), "default": DEFAULT_MODE},
        "backends": {
            "choices": list(BACKEND_CHOICES),
            "default": DEFAULT_BACKEND,
            "details": BACKENDS,  # per-backend model list + display name
        },
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


def _validate_ask(model: str, mode: str, backend: str) -> None:
    if model not in MODEL_CHOICES:
        raise HTTPException(400, f"Unknown model: {model}")
    if mode not in MODE_CHOICES:
        raise HTTPException(400, f"Unknown mode: {mode}")
    if backend not in BACKEND_CHOICES:
        raise HTTPException(400, f"Unknown backend: {backend}")


def _history_hash(history: list) -> str:
    """Hash of the conversation history that would be sent to the agent.
    Including this in the cache key means we don't return a stale answer
    when the user has had more turns since the cached run."""
    if not history:
        return "none"
    blob = json.dumps(
        [{"q": t.get("question", ""), "a": t.get("answer", "")} for t in history],
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _ask_cache_key(
    file_id: str, question: str, mode: str, model: str, backend: str,
    history: list,
) -> str:
    pdf = PDF_DIR / file_id
    pdf_mtime = pdf.stat().st_mtime_ns if pdf.exists() else 0
    q_norm = " ".join((question or "").strip().lower().split())
    blob = (
        f"{file_id}|{pdf_mtime}|{q_norm}|{mode}|{model}|{backend}|"
        f"{_history_hash(history)}"
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _ask_cache_put(key: str, stats: dict, answer: str, citations: list,
                   done_payload: dict) -> None:
    ASK_CACHE[key] = {
        "stats": stats,
        "answer": answer,
        "citations": citations,
        "done": done_payload,
    }
    # FIFO eviction — dict insertion order is preserved in Py3.7+.
    while len(ASK_CACHE) > ASK_CACHE_MAX:
        ASK_CACHE.pop(next(iter(ASK_CACHE)))


def _agg_cache_key(question: str, model: str, backend: str,
                   ordered: list) -> str:
    """Hash the aggregation inputs: question + model/backend + each PDF's
    answer text and citations (in display order). Changes whenever any
    upstream answer changes, so the synthesis never goes stale."""
    payload = {
        "q": " ".join((question or "").strip().lower().split()),
        "model": model, "backend": backend,
        "items": [
            {"f": o["file_id"], "a": o.get("answer", ""),
             "c": o.get("citations", [])}
            for o in ordered
        ],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _agg_cache_put(key: str, payload: dict) -> None:
    AGG_CACHE[key] = payload
    while len(AGG_CACHE) > AGG_CACHE_MAX:
        AGG_CACHE.pop(next(iter(AGG_CACHE)))


def _persist_turn(file_id: str, question: str, answer: str,
                  citations: list) -> None:
    turns = CHATS.setdefault(file_id, [])
    turns.append({"question": question, "answer": answer,
                  "citations": citations})
    del turns[:-MAX_TURNS_KEPT]


async def _run_ask(
    file_id: str, question: str, mode: str, model: str, backend: str,
    *, history: list | None = None, persist: bool = True,
    use_cache: bool = True,
):
    """Shared backbone for /ask and /ask-multi. Yields (type, payload)
    tuples in the same shape run_agent does, plus handles cache lookup,
    cache storage, and (optionally) persistence into CHATS.

    On a cache hit, replays the stats + final text + done events without
    spawning the agent — that's where the token savings live."""
    safe = _safe_id(file_id)
    pdf_path = PDF_DIR / safe
    if history is None:
        history = list(CHATS.get(safe, []))

    if use_cache:
        key = _ask_cache_key(safe, question, mode, model, backend, history)
        hit = ASK_CACHE.get(key)
        if hit:
            yield ("stats", hit["stats"])
            yield ("text", hit["answer"])
            yield ("done", {**hit["done"], "cached": True, "cost_usd": 0.0})
            if persist and hit["answer"].strip():
                _persist_turn(safe, question, hit["answer"], hit["citations"])
            return
    else:
        key = None

    final_answer = ""
    final_citations: list = []
    final_stats: dict = {}
    final_done: dict = {}
    try:
        async for kind, payload in run_agent(
            pdf_path, question, model=model, mode=mode,
            history=history, backend=backend,
        ):
            if kind == "stats":
                final_stats = payload
            elif kind == "done":
                final_answer = payload.get("answer") or ""
                final_citations = payload.get("citations") or []
                final_done = payload
            yield (kind, payload)
    except Exception as e:
        yield ("error", str(e))
        return

    if final_answer.strip():
        if key is not None:
            _ask_cache_put(key, final_stats, final_answer, final_citations,
                           final_done)
        if persist:
            _persist_turn(safe, question, final_answer, final_citations)


@app.post("/ask")
async def ask(body: AskBody):
    safe = _safe_id(body.file_id)
    if not (PDF_DIR / safe).exists():
        raise HTTPException(404)
    _validate_ask(body.model, body.mode, body.backend)

    async def stream():
        async for kind, payload in _run_ask(
            safe, body.question, body.mode, body.model, body.backend,
        ):
            yield f"data: {json.dumps({'type': kind, 'data': payload})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


class AskMultiBody(BaseModel):
    file_ids: list[str]
    question: str
    model: str = DEFAULT_MODEL
    mode: str = DEFAULT_MODE
    backend: str = DEFAULT_BACKEND


@app.post("/ask-multi")
async def ask_multi(body: AskMultiBody):
    """Fan one question out across N PDFs in parallel. Events are
    multiplexed into a single SSE stream, each tagged with the file_id
    they belong to. Each PDF runs with empty history (treated as a
    standalone query, not a turn in that PDF's chat) and answers are NOT
    persisted into CHATS — multi-PDF answers live only in the response
    stream, so the per-file conversations stay clean."""
    if not body.file_ids:
        raise HTTPException(400, "file_ids must be non-empty")
    _validate_ask(body.model, body.mode, body.backend)

    safe_ids: list[str] = []
    for fid in body.file_ids:
        safe = _safe_id(fid)
        if not (PDF_DIR / safe).exists():
            raise HTTPException(404, f"Not found: {safe}")
        safe_ids.append(safe)

    run_id = _new_run_id()
    # Pre-create the bucket so /pdf?cmp=<id> can 404 cleanly until the
    # first done event lands rather than racing on a missing key.
    COMPARE_RUNS.setdefault(run_id, {})

    async def stream():
        queue: asyncio.Queue = asyncio.Queue()
        results: dict[str, dict] = {}  # fid -> {"answer", "citations"}

        async def runner(fid: str):
            try:
                async for kind, payload in _run_ask(
                    fid, body.question, body.mode, body.model, body.backend,
                    history=[], persist=False,
                ):
                    if kind == "done":
                        cits = payload.get("citations") or []
                        # Save citations so /pdf?cmp= can render this PDF's
                        # highlights, and stash the answer for aggregation.
                        _compare_runs_put(run_id, fid, cits)
                        results[fid] = {
                            "answer": payload.get("answer") or "",
                            "citations": cits,
                        }
                    await queue.put({"file_id": fid, "type": kind,
                                     "data": payload})
            except Exception as e:
                await queue.put({"file_id": fid, "type": "error",
                                 "data": str(e)})
            await queue.put({"file_id": fid, "type": "_runner_done"})

        tasks = [asyncio.create_task(runner(fid)) for fid in safe_ids]
        outstanding = set(safe_ids)
        # Tell the client which run id to use for chip/pill clicks, what
        # files to render slots for, and in what order.
        yield (
            f"data: {json.dumps({'type': 'multi_start', 'data': {'run_id': run_id, 'file_ids': safe_ids}})}\n\n"
        )
        try:
            while outstanding:
                evt = await queue.get()
                if evt["type"] == "_runner_done":
                    outstanding.discard(evt["file_id"])
                    continue
                yield f"data: {json.dumps(evt)}\n\n"
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

        # ── Cross-paper synthesis ──────────────────────────────────────
        # Once every PDF has answered, hand all answers to one aggregation
        # agent that writes a single unified answer citing across papers.
        ordered = [
            {"file_id": fid, **results[fid]}
            for fid in safe_ids if fid in results and results[fid]["answer"].strip()
        ]
        if ordered:
            yield f"data: {json.dumps({'type': 'aggregate_start', 'data': {}})}\n\n"
            if len(ordered) == 1:
                # Nothing to synthesize across — replay the single paper's
                # answer as the "aggregate" (its global numbering == local),
                # with no extra LLM call.
                gcites, _ = _build_global_citations(ordered)
                only = ordered[0]["answer"]
                yield f"data: {json.dumps({'type': 'aggregate_text', 'data': only})}\n\n"
                yield f"data: {json.dumps({'type': 'aggregate_done', 'data': {'answer': only, 'citations': gcites, 'single': True}})}\n\n"
            else:
                agg_key = _agg_cache_key(body.question, body.model, body.backend, ordered)
                cached = AGG_CACHE.get(agg_key)
                if cached:
                    yield f"data: {json.dumps({'type': 'aggregate_text', 'data': cached['answer']})}\n\n"
                    yield f"data: {json.dumps({'type': 'aggregate_done', 'data': {**cached, 'cached': True}})}\n\n"
                else:
                    agg_payload = None
                    try:
                        async for kind, payload in aggregate_answers(
                            body.question, ordered,
                            model=body.model, backend=body.backend,
                        ):
                            if kind == "agg_text":
                                yield f"data: {json.dumps({'type': 'aggregate_text', 'data': payload})}\n\n"
                            elif kind == "agg_done":
                                agg_payload = payload
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'aggregate_error', 'data': str(e)})}\n\n"
                    # Only cache + finalize a NON-empty synthesis; an empty
                    # one is a silent failure, not a result to memoize.
                    if agg_payload is not None and (agg_payload.get("answer") or "").strip():
                        _agg_cache_put(agg_key, agg_payload)
                        yield f"data: {json.dumps({'type': 'aggregate_done', 'data': agg_payload})}\n\n"
                    elif agg_payload is not None:
                        yield f"data: {json.dumps({'type': 'aggregate_error', 'data': 'synthesis produced no text'})}\n\n"

        yield f"data: {json.dumps({'type': 'multi_done', 'data': {'run_id': run_id}})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
