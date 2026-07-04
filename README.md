# Parliamentary Speech Finder (GermaParl / Hansard)

Semantic search over **parliamentary speeches** — the German Bundestag (GermaParl, 1949–2025)
and, optionally, the UK House of Commons (Hansard). You ask in plain language
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
  (Blätte et al.). Cite/observe their terms. **Hansard** (UK) similarly from its source.
- Build the deployment artifacts from your embeddings with the scripts in `tools/`:
  ```bash
  # produces data/de_metadata.parquet + data/indexes/de_ivfpq.faiss (row-aligned)
  python tools/build_metadata_parquet.py --data-dir /path/with/DE_Master_Chunks.csv --corpus DE
  python tools/build_sq8_indexes.py      --data-dir /path/with/embeddings_jina_v4 --corpus DE  # near-exact, ~2 GB
  ```

## Run it

```bash
cp .env.example .env          # put your JINA_API_KEY in it (query-side only; never commit .env)
pip install -r requirements-api.txt
PORTABLE_PARLIAMENT_DATA_DIR=./data streamlit run app.py
```

- **Server (subdomain) deployment:** see `deploy/systemd/germaparl-finder.service` (Streamlit on
  127.0.0.1:18003) + `deploy/Caddyfile.snippet` (reverse-proxy a subdomain, reusing an existing cert).
- **Laptop / on-prem:** `deploy/run_local.sh`, or the macOS double-click launcher `deploy/run_peer_app.command`.

## Citing

If you use this software, please cite it via the archived release — see `CITATION.cff`
(a Zenodo DOI is minted per GitHub release). Please also cite the underlying corpora (GermaParl/PolMine, Hansard).

## License

[MIT](LICENSE) © 2026 Konstantin Wandel. Built at Universität Bielefeld. Uses the Jina v4 embeddings
(model © Jina AI) and GermaParl/Hansard data under their respective terms.
