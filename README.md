# Plenarsuche

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21194707.svg)](https://doi.org/10.5281/zenodo.21194707)

Plenarsuche is a web application for semantic search over the plenary speeches of the German Bundestag
from 1949 to 2025. It works on GermaParl, the corpus of the [PolMine project](https://polmine.github.io/).
You describe a topic in plain language ("Deutsche Wiedervereinigung", "Klimawandel und Energiepolitik",
"Rente mit 63") and receive the most similar speeches, ranked, with speaker, party, date and the passage.
Results can be limited to a range of years, and selected speeches can be downloaded as PDF files, one file
per speech.

A public instance is running at <https://germaparl.geolab.soz.uni-bielefeld.de>.

> Status: research prototype. The search returns the original speeches ranked by similarity. An optional
> answer mode lets a language model summarise the most similar speeches with numbered source references;
> it can be wrong, and hits and answers should be checked against the official records before use.

## How it works

- **Embeddings:** every speech is encoded as a whole with
  [`jinaai/jina-embeddings-v4`](https://huggingface.co/jinaai/jina-embeddings-v4) (multilingual, 2048
  dimensions, context of up to 32k tokens), using the retrieval task with the passage adapter and
  L2-normalised vectors. Speeches of more than 24,000 words (three in the corpus) are split into
  overlapping chunks, so such a speech can appear twice in a result list.
- **Index:** a FAISS flat index with 8-bit scalar quantisation (SQ8), searched by inner product, which equals
  cosine similarity on normalised vectors. On the German speeches it reaches a recall@10 of about 1.0
  against a brute-force search, needs about 2 GB per million speeches and answers in about 150 ms on a CPU.
  An IVF-PQ build is roughly thirty times smaller but too lossy on these vectors (recall@10 of about 0.32),
  so SQ8 is the default.
- **Query encoding**, two backends (`PARLIAMENT_EMBED_BACKEND`):
  - `api` (the default when `JINA_API_KEY` is set) encodes the query through the
    [Jina embeddings API](https://jina.ai/embeddings/) with `task=retrieval.query`. The serving machine
    needs no GPU and no torch, only `requirements-api.txt`.
  - `local` loads `jina-embeddings-v4` through `sentence-transformers` and needs the full
    `requirements.txt` and a machine with enough memory (about 8 GB for the weights alone).
  - A third option is a small embedding service on a GPU machine, reached over an SSH tunnel
    (`PARLIAMENT_EMBED_URL`, see `deploy/llm/`). It is tried first; if it does not answer, the backend above
    takes over.
- **Answers (optional):** with `PARLIAMENT_LLM_URL` set, the interface offers a switch that lets a language
  model summarise the most similar speeches (as many as the visitor chooses, up to 50). The model sees the question and excerpts of those speeches
  and nothing else, must cite them by number, and the sources are shown directly below the answer. The model
  runs on a separate GPU machine behind a reverse tunnel (`deploy/llm/`), so the public host needs no GPU. If
  the tunnel is down the switch disappears and the search keeps working.
- **Results:** the ranked original passages. The interface is a React single-page application on top of a
  FastAPI service; a single-file Streamlit version is kept for offline use.

## Data

The repository contains the software. The speeches come from GermaParl, which the PolMine project
(Blätte et al.) publishes under its own terms (the prepared corpus under CLARIN PUB+BY+NC+SA, see the
[GermaParlTEI repository](https://github.com/PolMine/GermaParlTEI)); please obtain the corpus there and cite it.
The page footer and every exported PDF name the source and the licence. Embeddings and
index are built from the speeches with the scripts in `tools/`; they are large and therefore not stored in
the repository.

```bash
# produces data/de_metadata.parquet and data/indexes/de_sq8.faiss (row-aligned)
python tools/build_metadata_parquet.py --data-dir /path/with/DE_Master_Chunks.csv --corpus DE
python tools/build_sq8_indexes.py      --data-dir /path/with/embeddings_jina_v4 --corpus DE
```

## Run it

**Web app (FastAPI and React).**
```bash
cp .env.example .env                      # put your JINA_API_KEY in it, and never commit .env
# backend (retrieval API)
pip install -r backend/requirements.txt
cd backend && PARLIAMENT_DATA_DIR=../data uvicorn main:app --port 18004
# frontend (separate shell)
cd frontend && npm install && VITE_API_URL=http://localhost:18004/api npm run dev
```
In production, `uvicorn main:app` runs behind a reverse proxy that routes `/api/*` to the backend and serves
the static frontend build at `/`; `deploy/Caddyfile.snippet` shows the Caddy configuration used for the
public instance.

**Streamlit (single file, portable).**
```bash
pip install -r requirements-api.txt
PORTABLE_PARLIAMENT_DATA_DIR=./data streamlit run app.py
```
or the macOS double-click launcher `deploy/run_peer_app.command`.

## Configuration

Environment variables of the backend (all optional except the key for the `api` backend):

| Variable | Default | Meaning |
|---|---|---|
| `JINA_API_KEY` | none | key for query encoding through the Jina API |
| `PARLIAMENT_EMBED_BACKEND` | `api` with a key, else `local` | where queries are encoded |
| `PARLIAMENT_DATA_DIR` | `./data` | folder with `de_metadata.parquet` and `indexes/de_sq8.faiss` |
| `PARLIAMENT_MAX_QUERY_CHARS` | 500 | longest accepted query |
| `PARLIAMENT_RATE_PER_MIN` | 30 | uncached searches per visitor and minute (held in memory) |
| `PARLIAMENT_DAILY_EMBED_CAP` | 3000 | uncached searches per day across all visitors (0 switches the cap off) |
| `PARLIAMENT_EMBED_URL` | unset | query-embedding service (`deploy/llm/embed_service.py`), tried before the backend above |
| `PARLIAMENT_LLM_URL` | unset | OpenAI-compatible endpoint of the answer model, for example `http://127.0.0.1:18100/v1`; unset switches answers off |
| `PARLIAMENT_LLM_KEY` | unset | key the endpoint requires (also used for the embedding service) |
| `PARLIAMENT_ANSWER_PER_HOUR` | 12 | answers per visitor and hour |
| `PARLIAMENT_ANSWER_DAILY_CAP` | 400 | answers per day across all visitors |
| `PARLIAMENT_ANSWER_CONCURRENCY` | 2 | answers generated at the same time; a third request is refused politely |
| `PARLIAMENT_CORS_ORIGINS` | the public address and local dev servers | origins allowed to call the API from a browser |

The data are loaded when the service starts. A search that is repeated, or only changes the year filter,
reuses the cached query vector and costs no external call.

## Feedback

The page header links to a short feedback form at
<https://geolab.soz.uni-bielefeld.de/feedback/?app=germaparl>. The form stores a message and an optional
email address on the research server for 90 days and does not attach the query, the selected speeches or
any browser identity.

## Author

Konstantin Wandel, Universität Bielefeld (SOEP-RegioHub): <https://konstantinwandel.github.io/>

## Citing

Please cite the software through the archived release (see `CITATION.cff`; a Zenodo DOI is minted for each
GitHub release), and cite the underlying corpus, GermaParl (PolMine), as well.

## License

[MIT](LICENSE) © 2026 Konstantin Wandel. Built at Universität Bielefeld. The Jina v4 model is © Jina AI,
and GermaParl is used under its own terms.
