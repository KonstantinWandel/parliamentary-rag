#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

./run_peer_app.sh

sleep 3
open "http://localhost:8501"
