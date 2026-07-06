"""GermaParl Speech Finder — FastAPI retrieval backend.

Thin wrapper over the FAISS SQ8 index + Jina v4 query embeddings. Retrieval-only:
returns similarity-ranked speeches, no model-generated summary. Endpoints are sync
(`def`) so FastAPI runs them in a threadpool -> concurrent queries run in parallel.
"""
import os
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATA_DIR = Path(os.environ.get("PARLIAMENT_DATA_DIR", "./data"))
INDEX_PATH = DATA_DIR / "indexes" / "de_sq8.faiss"
META_PATH = DATA_DIR / "de_metadata.parquet"
JINA_API_KEY = os.environ.get("JINA_API_KEY", "").strip()
JINA_API_URL = os.environ.get("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
JINA_API_MODEL = os.environ.get("JINA_API_MODEL", "jina-embeddings-v4")
CORPUS_LABEL = os.environ.get("PARLIAMENT_CORPUS_LABEL", "Deutscher Bundestag (GermaParl)")

app = FastAPI(title="GermaParl Speech Finder API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_index = None
_df = None


def _load():
    global _index, _df
    if _index is not None:
        return
    idx = faiss.read_index(str(INDEX_PATH))
    df = pd.read_parquet(META_PATH)
    df["text"] = df["text"].fillna("").astype(str)
    spk = "speaker_name" if "speaker_name" in df.columns else ("speaker" if "speaker" in df.columns else None)
    df["speaker"] = (df[spk].fillna("").astype(str) if spk else "")
    df["party"] = df["party"].fillna("").astype(str) if "party" in df.columns else ""
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(-1).astype(int)
    else:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year.fillna(-1).astype(int)
    df["date"] = df["date"].astype(str) if "date" in df.columns else ""
    _index, _df = idx, df


def _encode(query: str, retries: int = 3) -> np.ndarray:
    # The Jina API occasionally has slow/timeout hiccups (esp. on the free tier). Retry
    # transient failures (network timeout / 5xx) with a short backoff; let 4xx (auth/quota)
    # raise immediately. `search` converts any final failure to a clean 503, not a 500.
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                JINA_API_URL, timeout=(10, 45),
                headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
                json={"model": JINA_API_MODEL, "task": "retrieval.query", "input": [{"text": query}]},
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            continue
        if resp.status_code >= 500:  # transient server-side → retry
            last_err = f"Jina API {resp.status_code}"
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()  # 4xx (auth/quota) → propagate immediately
        v = np.asarray(resp.json()["data"][0]["embedding"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n  # match the L2-normalized docs / inner-product index
        return v.reshape(1, -1)
    raise RuntimeError(f"Jina embedding API failed after {retries} attempts: {last_err}")


class SearchReq(BaseModel):
    query: str
    top_k: int = 10
    year_start: int | None = None
    year_end: int | None = None


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta():
    _load()
    yrs = _df.loc[_df["year"] > 0, "year"]
    return {
        "min_year": int(yrs.min()), "max_year": int(yrs.max()),
        "n_speeches": int(len(_df)), "corpus_label": CORPUS_LABEL,
    }


@app.post("/api/search")
def search(req: SearchReq):
    _load()
    t0 = time.time()
    q = (req.query or "").strip()
    if not q:
        return {"results": [], "took_ms": 0, "query": q, "total": 0}
    try:
        qv = _encode(q)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Die Suche ist momentan nicht erreichbar (Embedding-Dienst). Bitte in einem Moment erneut versuchen.",
        )
    k = max(1, min(int(req.top_k or 10), 50))
    ys, ye = req.year_start, req.year_end
    filtered = ys is not None or ye is not None
    search_k = min(_index.ntotal, k * 10 if filtered else k)
    out = []
    while True:
        D, I = _index.search(qv, search_k)
        out = []
        for score, ri in zip(D[0], I[0]):
            ri = int(ri)
            if ri < 0:
                continue
            row = _df.iloc[ri]
            yr = int(row["year"])
            if ys is not None and yr < ys:
                continue
            if ye is not None and yr > ye:
                continue
            out.append({
                "rank": len(out) + 1, "score": round(float(score), 4),
                "speaker": row["speaker"], "party": row["party"],
                "date": str(row["date"])[:10], "year": yr, "text": row["text"],
            })
            if len(out) >= k:
                break
        if len(out) >= k or search_k >= _index.ntotal:
            break
        search_k = min(_index.ntotal, search_k * 4)
    return {"results": out, "took_ms": int((time.time() - t0) * 1000), "query": q, "total": len(out)}
