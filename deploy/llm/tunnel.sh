#!/usr/bin/env bash
# Reverse tunnel from the GPU machine to the public VM: the VM's 127.0.0.1:18100 reaches the local vLLM.
# The connection is opened from this side, so nothing on this machine listens to the outside. The key is
# restricted on the VM to forwarding exactly these loopback ports (no shell, no other forwards).
#   TUNNEL_KEY  private key (default ~/.ssh/id_ed25519_plenarsuche_tunnel)
#   TUNNEL_HOST user@host of the public host (required)
# Runs in a restart loop; start it detached: setsid bash tunnel.sh </dev/null >>tunnel.log 2>&1 &
set -u
KEY="${TUNNEL_KEY:-$HOME/.ssh/id_ed25519_plenarsuche_tunnel}"
HOST="${TUNNEL_HOST:?set TUNNEL_HOST=user@host of the public host}"
PORTS="${PORTS:-18100 18101}"     # 18100 answer model, 18101 query embeddings
FWD=""
for p in $PORTS; do FWD="$FWD -R 127.0.0.1:${p}:127.0.0.1:${p}"; done
echo $$ > "${PIDFILE:-/tmp/plenarsuche_tunnel.pid}"
while true; do
  ssh -N -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ConnectTimeout=10 \
      $FWD "$HOST"
  echo "$(date -Is) tunnel ended (exit $?), retrying in 10 s"
  sleep 10
done
