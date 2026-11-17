#!/usr/bin/env bash
# Start (or restart what has died of) the three processes behind the answer feature on the GPU machine:
# the answer model (vLLM, :18100), the query-embedding service (:18101) and the reverse tunnel to the VM.
# Each runs detached in its own session with a pidfile; running this again only starts what is missing.
#   LOGDIR   where the logs go (default ~/plenarsuche_logs)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Local settings (TUNNEL_HOST=user@host, optional MODEL, GPU_FRACTION) live outside the repository.
[ -f "$HOME/.config/plenarsuche.env" ] && set -a && . "$HOME/.config/plenarsuche.env" && set +a
LOGDIR="${LOGDIR:-$HOME/plenarsuche_logs}"; mkdir -p "$LOGDIR"
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
start() {   # name script pidfile
  if alive "$3"; then echo "$1: running (pid $(cat "$3"))"; return; fi
  setsid bash "$HERE/$2" </dev/null >>"$LOGDIR/$1.log" 2>&1 &
  echo "$1: started"
}
start vllm   run_vllm.sh   /tmp/plenarsuche_vllm.pid
start embed  run_embed.sh  /tmp/plenarsuche_embed.pid
start tunnel tunnel.sh     /tmp/plenarsuche_tunnel.pid
