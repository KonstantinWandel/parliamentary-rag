"""Speech / article Finder — FastAPI retrieval backend (multi-corpus).

Serves semantic search over one or more Jina-v4 FAISS SQ8 indexes. Retrieval-only:
similarity-ranked passages, no model-generated summary. Endpoints are sync (`def`) so
FastAPI runs them in a threadpool -> concurrent queries parallelize.

Corpora are auto-discovered: any entry in CORPORA whose index + metadata files exist
under PARLIAMENT_DATA_DIR is served. So the same code serves GermaParl alone today and
gains RegioPress the moment its index is built — no config change.

Query encoding has two backends (PARLIAMENT_EMBED_BACKEND):
  local  — jina-embeddings-v4 via sentence-transformers on this box's GPU (no API key).
  api    — the Jina embeddings API (for a keyed, GPU-less host like the VM).
Default: api if JINA_API_KEY is set, else local.
"""
import os
import threading
import time
from pathlib import Path

from collections import OrderedDict

import faiss
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATA_DIR = Path(os.environ.get("PARLIAMENT_DATA_DIR", "./data"))
JINA_API_KEY = os.environ.get("JINA_API_KEY", "").strip()
JINA_API_URL = os.environ.get("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
JINA_API_MODEL = os.environ.get("JINA_API_MODEL", "jina-embeddings-v4")
EMBED_BACKEND = os.environ.get("PARLIAMENT_EMBED_BACKEND", "api" if JINA_API_KEY else "local").strip().lower()
EMBED_MODEL = os.environ.get("PARLIAMENT_EMBED_MODEL", "jinaai/jina-embeddings-v4")
EMBED_DEVICE = os.environ.get("PARLIAMENT_EMBED_DEVICE", "cuda").strip()
EMBED_MAX_SEQ = int(os.environ.get("PARLIAMENT_EMBED_MAX_SEQ", "512"))
NPROBE = int(os.environ.get("PARLIAMENT_NPROBE", "64"))  # IVF cells scanned per query (recall<->speed)
# Move the FAISS index onto the GPU. At RegioPress scale (62.7M) a flat CPU index searches a
# single query single-threaded (~56s); on the H200 a flat SQ8 clone is EXACT and ~tens of ms.
# IVF was rejected here: the corpus is near-duplicate-heavy, so clustering is lopsided (~0.4 recall).
FAISS_GPU = os.environ.get("PARLIAMENT_FAISS_GPU", "0").strip() not in ("", "0", "false", "False")
GPU_ID = int(os.environ.get("PARLIAMENT_GPU_ID", "0"))
HNSW_EF = int(os.environ.get("PARLIAMENT_HNSW_EF", "512"))  # recall<->speed AND the filtered-query candidate pool
MAX_EXACT = int(os.environ.get("PARLIAMENT_MAX_EXACT", "5000000"))  # exact filtered search cap when vectors are in RAM
# (RAM fancy-index + matmul is ~1-2s even at 5M; covers every paper. Above this -> ANN pool.)

# Static corpus registry. `kind` drives the field projection and the UI card layout;
# `embed` pins each corpus to the model + query-formatting its index was BUILT with, so a
# query is always encoded into the same space as its documents (GermaParl=Jina v4;
# RegioPress=e5-large-instruct, whose queries take the instruct prefix and whose passages
# are raw — see tools/embed_regiopress.py).
CORPORA = {
    "germaparl": {
        "kind": "parliament",
        "label": "Deutscher Bundestag (GermaParl)",
        "index": "indexes/de_sq8.faiss",
        "meta": "de_metadata.parquet",
        "embed": {"model": "jinaai/jina-embeddings-v4", "family": "jina"},
    },
    "regiopress": {
        "kind": "press",
        "label": "RegioPress (Regionalzeitungen)",
        "index": "indexes/rp_near_hnsw.bin",
        "index_type": "hnswlib",
        "dim": 1024,
        "meta": "rp_near.parquet",
        "vectors": "regiopress/rp_near_embeddings.npy",  # loaded to RAM for exact filtered search (any paper size)
        "embed": {"model": "intfloat/multilingual-e5-large-instruct", "family": "e5",
                  "query_instruct": "Instruct: Given a search query, retrieve relevant German news articles\nQuery: "},
    },
}

# Optional allowlist: PARLIAMENT_CORPORA="regiopress" serves ONLY that corpus. This box
# serves RegioPress; GermaParl is served separately from the GeoLAB VM, so we exclude it
# here even though its index files happen to be staged on disk. Empty = every built corpus.
_ALLOW = [c.strip() for c in os.environ.get("PARLIAMENT_CORPORA", "").split(",") if c.strip()]


def _available():
    """Corpora that are allowed AND whose index + metadata files exist, in registry order."""
    out = []
    for cid, c in CORPORA.items():
        if _ALLOW and cid not in _ALLOW:
            continue
        if (DATA_DIR / c["index"]).exists() and (DATA_DIR / c["meta"]).exists():
            out.append(cid)
    return out


DEFAULT_CORPUS = os.environ.get("PARLIAMENT_DEFAULT_CORPUS", "").strip() or (
    _available()[0] if _available() else "germaparl"
)

app = FastAPI(title="Semantic Finder API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_loaded = {}          # cid -> {"index":..., "df":...}
_load_lock = threading.Lock()
_models = {}          # model_name -> SentenceTransformer (a corpus may use its own encoder)
_model_lock = threading.Lock()
_gpu_res = []         # StandardGpuResources must outlive the GPU indexes that use them


def _norm_year(df):
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce").fillna(-1).astype(int)
    return pd.to_datetime(df.get("date"), errors="coerce").dt.year.fillna(-1).astype(int)


def _load_corpus(cid: str):
    if cid in _loaded:
        return _loaded[cid]
    if cid not in CORPORA:
        raise HTTPException(404, f"Unknown corpus '{cid}'")
    if _ALLOW and cid not in _ALLOW:
        raise HTTPException(404, f"Corpus '{cid}' is not served here.")
    c = CORPORA[cid]
    ipath, mpath = DATA_DIR / c["index"], DATA_DIR / c["meta"]
    if not ipath.exists() or not mpath.exists():
        raise HTTPException(503, f"Corpus '{cid}' is not built yet.")
    with _load_lock:
        if cid in _loaded:
            return _loaded[cid]
        itype = c.get("index_type", "faiss")
        if itype == "hnswlib":
            import hnswlib
            idx = hnswlib.Index(space="ip", dim=int(c["dim"]))
            idx.load_index(str(ipath))
            idx.set_ef(HNSW_EF)
            ntotal = idx.get_current_count()
        else:
            idx = faiss.read_index(str(ipath))
            try:  # IVF indexes need nprobe set; no-op for flat indexes
                faiss.extract_index_ivf(idx).nprobe = NPROBE
            except Exception:
                pass
            if FAISS_GPU:
                try:
                    res = faiss.StandardGpuResources()
                    idx = faiss.index_cpu_to_gpu(res, GPU_ID, idx)  # CPU copy is dropped -> RAM freed
                    _gpu_res.append(res)                            # keep resources alive
                except Exception as e:
                    print(f"[warn] '{cid}' index stays on CPU (GPU load failed): {e}", flush=True)
            ntotal = idx.ntotal
        df = pd.read_parquet(mpath)
        df["text"] = df["text"].fillna("").astype(str) if "text" in df.columns else ""
        df["year"] = _norm_year(df)
        df["date"] = df["date"].astype(str) if "date" in df.columns else ""
        extra = {}
        if c["kind"] == "parliament":
            spk = next((k for k in ("speaker_name", "speaker") if k in df.columns), None)
            df["speaker"] = df[spk].fillna("").astype(str) if spk else ""
            df["party"] = df["party"].fillna("").astype(str) if "party" in df.columns else ""
        else:  # press: RegioPress metadata uses source/section/page (no source_id/ressort/title).
            # Normalize section casing ("Lokales"/"LOKALES"/"lokales" -> "Lokales") so the ressort
            # filter/dropdown merges the variants; the raw `section` in the data stays untouched.
            df["ressort"] = (df["section"].fillna("").astype(str).str.strip().str.title()
                             if "section" in df.columns else "")
            df["source_id"] = df["source"].fillna("").astype(str) if "source" in df.columns else ""
            for col in ("source", "title", "subtitle", "page"):
                df[col] = df[col].fillna("").astype(str) if col in df.columns else ""
            # filter arrays for the exhaustive selective-filter path (exact search within a paper/ressort)
            sc = df["source_id"].astype("category"); rc = df["ressort"].astype("category")
            extra = {"yr_arr": df["year"].to_numpy(),
                     "src_codes": sc.cat.codes.to_numpy(), "src_map": {v: i for i, v in enumerate(sc.cat.categories)},
                     "res_codes": rc.cat.codes.to_numpy(), "res_map": {v: i for i, v in enumerate(rc.cat.categories)}}
        vecs = None
        if c.get("vectors"):
            vpath = DATA_DIR / c["vectors"]
            if vpath.exists():
                t = time.time()
                vecs = np.load(vpath)   # full load into RAM (fancy-index needs RAM, not a CephFS memmap)
                print(f"[{cid}] loaded {vecs.shape} vectors into RAM for exact filtering ({time.time()-t:.0f}s)", flush=True)
        _loaded[cid] = {"index": idx, "df": df, "ntotal": ntotal, "type": itype, "vecs": vecs, **extra}
        return _loaded[cid]


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
                m.half()  # match the fp16 doc embeddings; negligible for a single-query encode
            except Exception:
                pass
        _models[name] = m
        return m


def _encode_local(query: str, c: dict) -> np.ndarray:
    """Encode a query in the SAME space as corpus `c`'s documents."""
    e = c["embed"]
    model = _get_model(e["model"])
    if e.get("family") == "e5":
        # e5-large-instruct: queries get the instruct prefix, passages are raw (built that way).
        v = model.encode([e.get("query_instruct", "") + query], convert_to_numpy=True,
                         normalize_embeddings=True, batch_size=1, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)
    # jina (or unknown): try the task/prompt kwargs, fall back to a raw encode.
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


def _encode_api(query: str, retries: int = 3) -> np.ndarray:
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
        if resp.status_code >= 500:
            last_err = f"Jina API {resp.status_code}"
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        v = np.asarray(resp.json()["data"][0]["embedding"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v.reshape(1, -1)
    raise RuntimeError(f"Jina embedding API failed after {retries} attempts: {last_err}")


# A query vector depends only on the query text and the corpus's embedding space, so it is
# cached. On the VM the vector comes from the Jina API over the network, measured at 0.3 to 1.0 s
# and the single largest part of a search; a repeated query (a demo shown twice, a filter change,
# someone refining a phrase) then costs nothing. Small, bounded, per process.
_QUERY_VEC_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
QUERY_CACHE_SIZE = int(os.environ.get("PARLIAMENT_QUERY_CACHE", "512"))


def _encode(query: str, cid: str) -> np.ndarray:
    c = CORPORA[cid]
    key = (c["embed"].get("model"), c["embed"].get("query_instruct", ""), query)
    hit = _QUERY_VEC_CACHE.get(key)
    if hit is not None:
        _QUERY_VEC_CACHE.move_to_end(key)
        return hit
    # The Jina API path is only valid for the jina family (the GPU-less VM serving GermaParl);
    # everything else (incl. e5 RegioPress on this GPU box) encodes locally.
    if EMBED_BACKEND == "api" and c["embed"].get("family") == "jina":
        vec = _encode_api(query)
    else:
        vec = _encode_local(query, c)
    _QUERY_VEC_CACHE[key] = vec
    while len(_QUERY_VEC_CACHE) > QUERY_CACHE_SIZE:
        _QUERY_VEC_CACHE.popitem(last=False)
    return vec


def _project(kind: str, row, rank: int, score: float) -> dict:
    base = {"rank": rank, "score": round(float(score), 4),
            "date": str(row["date"])[:10], "year": int(row["year"]), "text": row["text"]}
    if kind == "parliament":
        base.update(speaker=row.get("speaker", ""), party=row.get("party", ""))
    else:
        base.update(source=row.get("source", ""), source_id=row.get("source_id", ""),
                    ressort=row.get("ressort", ""), title=row.get("title", ""),
                    subtitle=row.get("subtitle", ""), page=row.get("page", ""))
    return base


class SearchReq(BaseModel):
    query: str
    corpus: str | None = None
    top_k: int = 10
    year_start: int | None = None
    year_end: int | None = None
    source_id: str | None = None   # press only
    ressort: str | None = None     # press only


@app.get("/api/health")
def health():
    return {"ok": True, "backend": EMBED_BACKEND, "corpora": _available()}


@app.get("/api/corpora")
def corpora():
    return {"corpora": [{"id": c, "kind": CORPORA[c]["kind"], "label": CORPORA[c]["label"]}
                        for c in _available()], "default": DEFAULT_CORPUS}


@app.get("/api/meta")
def meta(corpus: str | None = None):
    cid = corpus or DEFAULT_CORPUS
    st = _load_corpus(cid)
    df = st["df"]
    yrs = df.loc[df["year"] > 0, "year"]
    out = {
        "corpus": cid, "kind": CORPORA[cid]["kind"], "label": CORPORA[cid]["label"],
        "min_year": int(yrs.min()) if len(yrs) else 0,
        "max_year": int(yrs.max()) if len(yrs) else 0,
        "count": int(len(df)), "n_speeches": int(len(df)),  # n_speeches kept as alias
    }
    if CORPORA[cid]["kind"] == "press":
        pairs = df[["source_id", "source"]].drop_duplicates().sort_values("source")
        out["sources"] = [{"id": a, "name": b} for a, b in pairs.itertuples(index=False) if b]
        out["ressorts"] = [r for r in df["ressort"].value_counts().head(40).index.tolist() if r]
    return out


def _knn(st, qv, k):
    """Top-k (scores, ids) from either a faiss (.search) or hnswlib (.knn_query) index."""
    if st["type"] == "hnswlib":
        k = min(k, HNSW_EF)                       # hnswlib needs k <= ef (fixed at load)
        labels, dists = st["index"].knn_query(qv, k=k)
        return 1.0 - dists[0], labels[0]          # ip space: cosine = 1 - distance
    D, I = st["index"].search(qv, k)
    return D[0], I[0]


def _filter_ids(st, ys, ye, src, res):
    """Row ids matching the press filters, via the precomputed arrays (fast, exact)."""
    m = np.ones(st["ntotal"], dtype=bool)
    if ys is not None:
        m &= st["yr_arr"] >= ys
    if ye is not None:
        m &= st["yr_arr"] <= ye
    if src:
        m &= st["src_codes"] == st["src_map"].get(src, -2)
    if res:
        m &= st["res_codes"] == st["res_map"].get(res, -2)
    return np.flatnonzero(m)


@app.post("/api/search")
def search(req: SearchReq):
    cid = req.corpus or DEFAULT_CORPUS
    st = _load_corpus(cid)
    df, kind = st["df"], CORPORA[cid]["kind"]
    t0 = time.time()
    q = (req.query or "").strip()
    if not q:
        return {"results": [], "took_ms": 0, "query": q, "total": 0, "corpus": cid, "kind": kind}
    t_embed = time.time()
    try:
        qv = _encode(q, cid)
        embed_ms = int((time.time() - t_embed) * 1000)
    except Exception:
        raise HTTPException(503, "Die Suche ist momentan nicht erreichbar (Embedding-Dienst). Bitte gleich erneut versuchen.")
    k = max(1, min(int(req.top_k or 10), 50))
    ys, ye, src, res = req.year_start, req.year_end, req.source_id, req.ressort
    filtered = any(v is not None and v != "" for v in (ys, ye, src, res))
    nt = st["ntotal"]
    # filtered: pull a big candidate pool then post-filter. faiss escalates search_k in the loop;
    # hnswlib is ef-capped, so grab the whole ef pool (HNSW_EF) up front for decent filter coverage.
    # Exhaustive path for a SELECTIVE press filter (specific paper/ressort): restrict to the matching
    # rows and do an EXACT search among them (vectors fetched from the in-RAM HNSW store), so a rare
    # paper+topic is never missed. Broad/year-only filters fall through to the ANN pool below.
    if st["type"] == "hnswlib" and kind == "press" and (src or res) and "src_codes" in st:
        ids = _filter_ids(st, ys, ye, src, res)
        if len(ids) == 0:
            return {"results": [], "took_ms": int((time.time() - t0) * 1000), "embed_ms": embed_ms,
                    "query": q, "total": 0, "corpus": cid, "kind": kind}
        if len(ids) <= MAX_EXACT:
            if st.get("vecs") is not None:
                vecs = st["vecs"][ids]                                          # RAM fancy-index, fast at any size
            else:
                vecs = np.asarray(st["index"].get_items(ids), dtype=np.float32)  # fallback if vectors not in RAM
            sims = (qv @ vecs.T)[0]
            top = np.argsort(-sims)[:k]
            out = [_project(kind, df.iloc[int(ids[t])], r + 1, float(sims[t])) for r, t in enumerate(top)]
            return {"results": out, "took_ms": int((time.time() - t0) * 1000), "embed_ms": embed_ms,
                    "query": q, "total": len(out), "corpus": cid, "kind": kind}
    search_k = min(nt, (HNSW_EF if st["type"] == "hnswlib" else k * 10) if filtered else k)
    out = []
    while True:
        scores, ids = _knn(st, qv, search_k)
        out = []
        for score, ri in zip(scores, ids):
            ri = int(ri)
            if ri < 0:
                continue
            row = df.iloc[ri]
            yr = int(row["year"])
            if ys is not None and yr < ys:
                continue
            if ye is not None and yr > ye:
                continue
            if kind == "press":
                if src and row.get("source_id", "") != src:
                    continue
                if res and row.get("ressort", "") != res:
                    continue
            out.append(_project(kind, row, len(out) + 1, score))
            if len(out) >= k:
                break
        if len(out) >= k or search_k >= nt or st["type"] == "hnswlib":
            break  # hnswlib is ef-capped, so no escalation pass
        search_k = min(nt, search_k * 4)
    return {"results": out, "took_ms": int((time.time() - t0) * 1000), "embed_ms": embed_ms,
            "query": q, "total": len(out), "corpus": cid, "kind": kind}


# Optionally serve the built React UI at / (same-origin, so the frontend's "/api" works
# with no CORS/proxy). Mounted last, after the /api routes, so those take precedence.
_UI_DIR = os.environ.get("PARLIAMENT_UI_DIR", "").strip()
if _UI_DIR and Path(_UI_DIR).is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="ui")
