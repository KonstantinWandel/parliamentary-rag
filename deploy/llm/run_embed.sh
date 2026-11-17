#!/usr/bin/env bash
# Start the query-embedding service (loopback :18101). Needs an env with sentence-transformers and peft.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${EMBED_PY:-$HOME/miniconda3/envs/bert-politics-new/bin/python}"
echo $$ > "${PIDFILE:-/tmp/plenarsuche_embed.pid}"
exec "$PY" -I "$(dirname "$0")/embed_service.py"
