#!/usr/bin/env bash
# Plenarsuche, answer model: a plain OpenAI-compatible vLLM server on loopback only.
# Reached from the public VM through the reverse SSH tunnel (tunnel.sh). It knows nothing about
# the corpus; the VM sends it the finished prompt.
#   MODEL         Hugging Face id (default Qwen/Qwen3.8-27B-FP8)
#   PORT          loopback port (default 18100)
#   KEY_FILE      file holding the API key the VM presents (default ~/.config/secrets/plenarsuche_llm.key)
#   GPU_FRACTION  share of GPU memory vLLM may reserve (default 0.40, the GPU is shared)
# Quantization is deliberately not passed: vLLM reads it from the checkpoint and picks the fast kernel.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
PORT="${PORT:-18100}"
KEY_FILE="${KEY_FILE:-$HOME/.config/secrets/plenarsuche_llm.key}"
GPU_FRACTION="${GPU_FRACTION:-0.40}"
VLLM_BIN="${VLLM_BIN:-$HOME/miniconda3/envs/vllm/bin/vllm}"
echo $$ > "${PIDFILE:-/tmp/plenarsuche_vllm.pid}"
exec "$VLLM_BIN" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name plenarsuche \
  --api-key "$(cat "$KEY_FILE")" \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_FRACTION" \
  --max-num-seqs 8
