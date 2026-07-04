#!/bin/bash
# Run the finder locally (laptop / on-prem). Needs data/de_metadata.parquet and
# data/indexes/de_ivfpq.faiss present, and a JINA_API_KEY in .env (or EMBED_BACKEND=local).
set -e
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export PORTABLE_PARLIAMENT_DATA_DIR="${PORTABLE_PARLIAMENT_DATA_DIR:-./data}"
exec streamlit run app.py --server.port "${PORT:-8501}"
