"""Plenarsuche: FastAPI retrieval backend for semantic search over Bundestag speeches (GermaParl).

Serves one Jina-v4 FAISS index (flat, 8-bit scalar quantisation) and its row-aligned metadata table.
Results are the original speeches ranked by similarity; nothing is generated. Endpoints are sync (`def`),
so FastAPI runs them in a threadpool and concurrent queries run in parallel.

Query encoding has two backends (PARLIAMENT_EMBED_BACKEND):
  local: jina-embeddings-v4 through sentence-transformers on this machine (no API key).
  api:   the Jina embeddings API (for a keyed host without a GPU, such as the public VM).
Default: api if JINA_API_KEY is set, else local.
"""
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger("uvicorn.error")   # goes to the journal; never log queries or visitor addresses

DATA_DIR = Path(os.environ.get("PARLIAMENT_DATA_DIR", "./data"))
JINA_API_KEY = os.environ.get("JINA_API_KEY", "").strip()
JINA_API_URL = os.environ.get("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
JINA_API_MODEL = os.environ.get("JINA_API_MODEL", "jina-embeddings-v4")
EMBED_BACKEND = os.environ.get("PARLIAMENT_EMBED_BACKEND", "api" if JINA_API_KEY else "local").strip().lower()
EMBED_MODEL = os.environ.get("PARLIAMENT_EMBED_MODEL", "jinaai/jina-embeddings-v4")
EMBED_DEVICE = os.environ.get("PARLIAMENT_EMBED_DEVICE", "cuda").strip()
EMBED_MAX_SEQ = int(os.environ.get("PARLIAMENT_EMBED_MAX_SEQ", "512"))

CORPUS_ID = "germaparl"
CORPUS_LABEL = "Deutscher Bundestag (GermaParl)"
INDEX_PATH = DATA_DIR / "indexes" / "de_sq8.faiss"
META_PATH = DATA_DIR / "de_metadata.parquet"

MAX_QUERY_CHARS = int(os.environ.get("PARLIAMENT_MAX_QUERY_CHARS", "500"))
QUERY_CACHE_SIZE = int(os.environ.get("PARLIAMENT_QUERY_CACHE", "512"))
RATE_PER_MIN = int(os.environ.get("PARLIAMENT_RATE_PER_MIN", "30"))        # uncached searches per visitor
DAILY_EMBED_CAP = int(os.environ.get("PARLIAMENT_DAILY_EMBED_CAP", "3000"))  # uncached searches per day, all visitors
JINA_CONCURRENCY = int(os.environ.get("PARLIAMENT_JINA_CONCURRENCY", "6"))

app = FastAPI(title="Plenarsuche API")
# The shipped frontend is same-origin and needs no CORS. The list only lets a local dev server
# (vite on :5173) and the public address talk to the API from a browser.
_CORS = [o.strip() for o in os.environ.get(
    "PARLIAMENT_CORS_ORIGINS",
    "https://germaparl.geolab.soz.uni-bielefeld.de,http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_CORS, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.exception_handler(RequestValidationError)
async def _invalid_request(request: Request, exc: RequestValidationError):
    """A short German message. FastAPI's default echoes the offending input, which breaks the response
    for a lone surrogate and gives the German interface an array where it expects a sentence."""
    if any(e.get("type") == "string_too_long" and "query" in e.get("loc", ()) for e in exc.errors()):
        msg = f"Die Suchanfrage ist zu lang (höchstens {MAX_QUERY_CHARS} Zeichen)."
    else:
        msg = "Ungültige Anfrage. Bitte die Eingaben prüfen."
    return JSONResponse(status_code=422, content={"detail": msg})


# ---------------------------------------------------------------------------------------------
# Data: loaded once, lazily or at start-up
# ---------------------------------------------------------------------------------------------
_state = None
_load_lock = threading.Lock()


def _norm_year(df):
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce").fillna(-1).astype(int)
    return pd.to_datetime(df.get("date"), errors="coerce").dt.year.fillna(-1).astype(int)


def _load():
    """Index and metadata, matched by row position. Loaded on first use if start-up loading failed."""
    global _state
    if _state is not None:
        return _state
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise HTTPException(503, "Der Datensatz ist momentan nicht verfügbar.")
    with _load_lock:
        if _state is not None:
            return _state
        t = time.time()
        index = faiss.read_index(str(INDEX_PATH))
        df = pd.read_parquet(META_PATH)
        # Rows are matched by position only, so a length mismatch would show the wrong speaker,
        # date and text with no error. Refuse to serve instead.
        if int(index.ntotal) != len(df):
            raise RuntimeError(f"index has {index.ntotal} vectors but the metadata has {len(df)} rows")
        df["text"] = df["text"].fillna("").astype(str) if "text" in df.columns else ""
        df["year"] = _norm_year(df)
        df["date"] = df["date"].astype(str) if "date" in df.columns else ""
        spk = next((k for k in ("speaker_name", "speaker") if k in df.columns), None)
        df["speaker"] = df[spk].fillna("").astype(str).str.replace("﻿", "", regex=False) if spk else ""
        df["party"] = df["party"].fillna("").astype(str) if "party" in df.columns else ""
        years = df["year"].to_numpy()
        valid = years[years > 0]
        _state = {"index": index, "df": df, "ntotal": int(index.ntotal), "years": years,
                  "min_year": int(valid.min()) if len(valid) else 0,
                  "max_year": int(valid.max()) if len(valid) else 0}
        log.info("data loaded: %d speeches in %.0f s", len(df), time.time() - t)
        return _state


@app.on_event("startup")
def _preload():
    """Load in the background so the first visitor after a restart does not wait for the data."""
    def run():
        try:
            _load()
        except Exception as e:   # the first request will try again and report the problem
            log.warning("start-up load failed: %s: %s", type(e).__name__, e)
    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------------------------------------
# Query encoding
# ---------------------------------------------------------------------------------------------
class EmbedUnavailable(Exception):
    """The query could not be turned into a vector (reason in args[0])."""


_models = {}
_model_lock = threading.Lock()


def _get_model(name: str):
    m = _models.get(name)
    if m is not None:
        return m
    with _model_lock:
        if name in _models:
            return _models[name]
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(name, trust_remote_code=True, device=EMBED_DEVICE)
        try:
            m.max_seq_length = EMBED_MAX_SEQ
        except Exception:
            pass
        if EMBED_DEVICE == "cuda":
            try:
                m.half()  # match the fp16 document embeddings; negligible for one short query
            except Exception:
                pass
        _models[name] = m
        return m


def _encode_local(query: str) -> np.ndarray:
    model = _get_model(EMBED_MODEL)
    for kw in (dict(task="retrieval", prompt_name="query"), dict(task="retrieval.query"), dict()):
        try:
            v = model.encode([query], convert_to_numpy=True, normalize_embeddings=True,
                             batch_size=1, show_progress_bar=False, **kw)
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 2 and v.shape[0] == 1:
                return v
        except (TypeError, ValueError):
            continue
    raise RuntimeError("Local query encoding failed (unexpected encode signature).")


def _encode_api(query: str, retries: int = 2) -> np.ndarray:
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                JINA_API_URL, timeout=(5, 15),
                headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
                json={"model": JINA_API_MODEL, "task": "retrieval.query", "input": [{"text": query}]},
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            continue
        if resp.status_code >= 500:
            log.warning("Jina API answered %s", resp.status_code)
            last_err = f"Jina API {resp.status_code}"
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            continue
        if resp.status_code >= 400:
            log.warning("Jina API answered %s (key, quota or input problem)", resp.status_code)
            raise EmbedUnavailable("quota" if resp.status_code == 429 else "rejected")
        v = np.asarray(resp.json()["data"][0]["embedding"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v.reshape(1, -1)
    raise RuntimeError(f"Jina embedding API failed after {retries} attempts: {last_err}")


# A query vector depends only on the query text, so it is cached: a repeated query, a filter change or a
# refined phrase then costs no external call. Bounded by entry count, guarded by a lock.
_QUERY_VEC_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_JINA_SLOTS = threading.BoundedSemaphore(JINA_CONCURRENCY)


def _cache_get(key):
    with _CACHE_LOCK:
        hit = _QUERY_VEC_CACHE.get(key)
        if hit is not None:
            _QUERY_VEC_CACHE.move_to_end(key)
        return hit


def _encode(query: str, count_call=None) -> np.ndarray:
    hit = _cache_get(query)
    if hit is not None:
        return hit
    if count_call is not None:
        count_call()        # only vectors that cost an external call count against an allowance
    if EMBED_BACKEND == "api":
        if not _JINA_SLOTS.acquire(timeout=5):
            raise EmbedUnavailable("busy")
        try:
            vec = _encode_api(query)
        finally:
            _JINA_SLOTS.release()
    else:
        vec = _encode_local(query)
    with _CACHE_LOCK:
        _QUERY_VEC_CACHE[query] = vec
        while len(_QUERY_VEC_CACHE) > QUERY_CACHE_SIZE:
            _QUERY_VEC_CACHE.popitem(last=False)
    return vec


# ---------------------------------------------------------------------------------------------
# Allowances for searches that cost an external embedding call. Held in memory only and keyed on the
# address Caddy forwards; nothing is written to disk or to the log.
# ---------------------------------------------------------------------------------------------
_rate = {}
_rate_lock = threading.Lock()
_daily = {"day": None, "n": 0}


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")) or "?"


def _allowance_check(ip: str):
    now = time.time()
    today = datetime.now(timezone.utc).date()
    with _rate_lock:
        if _daily["day"] != today:
            _daily["day"], _daily["n"] = today, 0
        if DAILY_EMBED_CAP and _daily["n"] >= DAILY_EMBED_CAP:
            raise HTTPException(503, "Das Tageslimit für neue Suchanfragen ist erreicht. Wiederholte Suchen und "
                                     "Filteränderungen funktionieren weiterhin; neue Suchbegriffe morgen wieder.")
        if len(_rate) > 20000:                      # bound the table
            for k in [k for k, d in _rate.items() if not d or now - d[-1] > 60]:
                _rate.pop(k, None)
        d = _rate.setdefault(ip, deque())
        while d and now - d[0] > 60:
            d.popleft()
        if len(d) >= RATE_PER_MIN:
            raise HTTPException(429, "Zu viele Suchanfragen in kurzer Zeit. Bitte in einer Minute erneut versuchen.")
        d.append(now)
        _daily["n"] += 1


# ---------------------------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------------------------
def _check_corpus(corpus):
    if corpus not in (None, "", CORPUS_ID):
        raise HTTPException(404, "Unbekannter Datensatz.")


def _project(row, rank: int, score: float) -> dict:
    return {"rank": rank, "score": round(float(score), 4), "date": str(row["date"])[:10],
            "year": int(row["year"]), "text": row["text"],
            "speaker": row.get("speaker", ""), "party": row.get("party", "")}


class SearchReq(BaseModel):
    query: str = Field(max_length=MAX_QUERY_CHARS)
    corpus: str | None = Field(default=None, max_length=40)   # accepted for older clients, only germaparl exists
    top_k: int = 10
    year_start: int | None = Field(default=None, ge=0, le=3000)
    year_end: int | None = Field(default=None, ge=0, le=3000)


@app.get("/api/health")
def health():
    return {"ok": True, "backend": EMBED_BACKEND, "corpora": [CORPUS_ID], "loaded": _state is not None}


@app.get("/api/corpora")
def corpora():
    return {"corpora": [{"id": CORPUS_ID, "kind": "parliament", "label": CORPUS_LABEL}], "default": CORPUS_ID}


@app.get("/api/meta")
def meta(corpus: str | None = None):
    _check_corpus(corpus)
    st = _load()
    return {"corpus": CORPUS_ID, "kind": "parliament", "label": CORPUS_LABEL,
            "min_year": st["min_year"], "max_year": st["max_year"],
            "count": int(len(st["df"])), "n_speeches": int(len(st["df"]))}   # n_speeches kept as alias


@app.post("/api/search")
def search(req: SearchReq, request: Request):
    _check_corpus(req.corpus)
    st = _load()
    df, nt = st["df"], st["ntotal"]
    t0 = time.time()
    # Drop NUL bytes and lone surrogates (they survive JSON parsing but break the response encoder).
    q = (req.query or "").replace("\x00", " ").encode("utf-8", "ignore").decode("utf-8").strip()
    ys, ye = req.year_start, req.year_end
    if ys is not None and ye is not None and ys > ye:
        raise HTTPException(422, "Das Startjahr liegt nach dem Endjahr.")

    def reply(results, embed_ms=0):
        return {"results": results, "took_ms": int((time.time() - t0) * 1000), "embed_ms": embed_ms,
                "query": q, "total": len(results), "corpus": CORPUS_ID, "kind": "parliament"}

    if not q:
        return reply([])
    t_embed = time.time()
    try:
        qv = _encode(q, count_call=lambda: _allowance_check(_client_ip(request)))
        embed_ms = int((time.time() - t_embed) * 1000)
    except HTTPException:
        raise
    except EmbedUnavailable as e:
        if e.args and e.args[0] in ("quota", "busy"):
            raise HTTPException(503, "Der Embedding-Dienst ist gerade stark ausgelastet. Bitte in einigen Sekunden erneut versuchen.")
        raise HTTPException(503, "Die Suche ist momentan nicht erreichbar (Embedding-Dienst). Bitte gleich erneut versuchen.")
    except Exception as e:
        log.warning("query encoding failed: %s", type(e).__name__)
        raise HTTPException(503, "Die Suche ist momentan nicht erreichbar (Embedding-Dienst). Bitte gleich erneut versuchen.")

    k = max(1, min(int(req.top_k or 10), 50))
    # Year filter on a numpy array. A range that matches nothing returns at once.
    yrs = st["years"]
    mask = np.ones(len(yrs), dtype=bool)
    if ys is not None:
        mask &= yrs >= ys
    if ye is not None:
        mask &= yrs <= ye
    if not mask.any():
        return reply([], embed_ms)
    filtered = ys is not None or ye is not None
    sel_ids = np.flatnonzero(mask)
    if filtered and len(sel_ids) < 0.5 * len(yrs) and hasattr(faiss, "SearchParameters"):
        # A narrow filter: search exactly among the matching rows in one pass. Widening an unfiltered
        # search until enough rows match would take many passes for a single year.
        params = faiss.SearchParameters()
        params.sel = faiss.IDSelectorBatch(sel_ids.astype("int64"))
        scores, ids = st["index"].search(qv, k, params=params)
        scores, ids = scores[0], ids[0]
        keep = ids >= 0
    else:
        search_k = min(nt, k * 10 if filtered else k)
        while True:
            scores, ids = st["index"].search(qv, search_k)
            scores, ids = scores[0], ids[0]
            ok = ids >= 0
            scores, ids = scores[ok], ids[ok]
            keep = mask[ids]
            if keep.sum() >= k or search_k >= nt:
                break
            search_k = min(nt, search_k * 4)
    scores, ids = scores[keep][:k], ids[keep][:k]
    return reply([_project(df.iloc[int(i)], r + 1, float(s)) for r, (s, i) in enumerate(zip(scores, ids))], embed_ms)


# Optionally serve the built React UI at / (same origin, so the frontend's "/api" works with no CORS or
# proxy). Mounted last, after the /api routes, so those take precedence.
_UI_DIR = os.environ.get("PARLIAMENT_UI_DIR", "").strip()
if _UI_DIR and Path(_UI_DIR).is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="ui")
