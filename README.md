# Parliamentary Speech Finder (GermaParl)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21194707.svg)](https://doi.org/10.5281/zenodo.21194707)

Semantic search over **parliamentary speeches** of the German Bundestag (GermaParl, 1949–2025).
The public deployment serves GermaParl only; other corpora the code supports are licensed
material and are run locally, never published. You ask in plain language
("Deutsche Wiedervereinigung", "Klimawandel und Energiepolitik", "Rente mit 63") and get the
most similar speeches back, ranked, with speaker, party, date and the passage — filterable by year.

> Status: retrieval-only research prototype. It returns similarity-ranked speeches and **no
> model-generated summary** — deliberately, to keep it lightweight and dependency-light for
> external hosting. Verify hits against the official records before use.

## How it works

- **Embeddings:** speeches are encoded with [`jinaai/jina-embeddings-v4`](https://huggingface.co/jinaai/jina-embeddings-v4)
  (multilingual, 2048-d, up to 32k-token context — so full speeches, not truncated) into a shared
  cross-lingual space (`task=retrieval`, passage adapter, L2-normalized).
- **Index:** a **FAISS flat SQ8** index (8-bit scalar quantization; inner product = cosine on the
  normalized vectors) — **near-exact** (recall@10 ≈ 1.0 vs a brute-force baseline), ~2 GB per ~1M German
  speeches, ~150 ms/query on CPU (negligible next to the query-embedding call). (An IVF-PQ build is
  ~30× smaller but far too lossy on these 2048-d vectors — recall@10 ≈ 0.32 — so SQ8 is the default.)
- **Query encoding — two backends** (`PARLIAMENT_EMBED_BACKEND`):
  - **`api`** (default when `JINA_API_KEY` is set): encodes the query via the
    [Jina embeddings API](https://jina.ai/embeddings/) (`task=retrieval.query`). **No local model,
    no GPU, no torch** — the serving box only needs `requirements-api.txt`. Ideal for a small CPU host.
  - **`local`**: loads `jina-embeddings-v4` via `sentence-transformers` on CPU (needs the full
    `requirements.txt`; heavier). Good for fully offline / air-gapped use.
- **Retrieval-only:** no LLM/summary; the ranked original passages are the output.

## Data — **not included in this repository**

This repo is **code only**. The corpora, embeddings and indexes are git-ignored and must be supplied
separately (they are large and/or license-bound):

- **GermaParl** (German Bundestag plenary protocols) — from the [PolMine project](https://polmine.github.io/)
  (Blätte et al.). Cite and observe their terms.
- Build the deployment artifacts from your embeddings with the scripts in `tools/`:
  ```bash
  # produces data/de_metadata.parquet + data/indexes/de_ivfpq.faiss (row-aligned)
  python tools/build_metadata_parquet.py --data-dir /path/with/DE_Master_Chunks.csv --corpus DE
  python tools/build_sq8_indexes.py      --data-dir /path/with/embeddings_jina_v4 --corpus DE  # near-exact, ~2 GB
  ```

## Run it

**Web app (FastAPI + React) — the deployed version, matching the sibling SOEP/INKAR finders.**
```bash
cp .env.example .env                      # put your JINA_API_KEY in it (never commit .env)
# backend (retrieval API)
pip install -r backend/requirements.txt
cd backend && PARLIAMENT_DATA_DIR=../data uvicorn main:app --port 18004
# frontend (separate shell)
cd frontend && npm install && VITE_API_URL=http://localhost:18004/api npm run dev
```
In production: `uvicorn main:app` behind a reverse proxy that routes `/api/*` to the backend and serves
the static `frontend/` build at `/` (a Caddy `germaparl` vhost reusing the wildcard cert — same pattern
as the SOEP/INKAR finders).

**Streamlit (single-file, portable) — the simple/offline alternative.**
```bash
pip install -r requirements-api.txt
PORTABLE_PARLIAMENT_DATA_DIR=./data streamlit run app.py
```
Or the macOS double-click launcher `deploy/run_peer_app.command`.

## Citing

If you use this software, please cite it via the archived release — see `CITATION.cff`
(a Zenodo DOI is minted per GitHub release). Please also cite the underlying corpus (GermaParl, PolMine).

## License

[MIT](LICENSE) © 2026 Konstantin Wandel. Built at Universität Bielefeld. Uses the Jina v4 embeddings
(model © Jina AI) and GermaParl data under its own terms.

## What the public deployment serves

`PARLIAMENT_CORPORA` is an allowlist, and the public deployment sets it to `germaparl`. Any other
corpus is refused with a 404 at every endpoint, even if its index files are present on the host.
This is deliberate: **RegioPress is licensed material (Genios) and must never be exposed publicly.**
It is run locally only, and its data has no place on a public host.

