#!/usr/bin/env bash
# SSH tunnels to the two vLLM endpoints. Ports in configs/models/*.yaml are local tunnel ports.
#
#   scripts/tunnel.sh up      # start both (idempotent)
#   scripts/tunnel.sh down    # stop
#   scripts/tunnel.sh status  # health of both endpoints
#
# The GPU nodes' 20049/20050 are SSH ports, not HTTP: each instance serves vLLM on 127.0.0.1.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .build/tunnel
HOST=${HYPOSPACE_SSH_HOST:-10.1.3.100}
USER_=${HYPOSPACE_SSH_USER:-isaac}
KEY=${HYPOSPACE_SSH_KEY:-}
JUMP=${HYPOSPACE_SSH_JUMP:-}   # e.g. user@jumphost when 10.1.3.100 is not directly reachable

# name ssh_port local_port remote_port
LINKS=(
  "27b ${HYPOSPACE_PORT_27B:-20049} 18000 8000"
  "9b  ${HYPOSPACE_PORT_9B:-20050} 18001 8001"
)

up_one() {
  local name=$1 sport=$2 lport=$3 rport=$4
  local pidf=.build/tunnel/$name.pid
  if curl -s -m 2 "http://127.0.0.1:$lport/v1/models" >/dev/null 2>&1; then
    echo "$name: 127.0.0.1:$lport already answering"
    return
  fi
  if [[ -f $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "$name: already up (pid $(cat "$pidf"))"
    return
  fi
  local keyopt=()
  [[ -n $KEY ]] && keyopt=(-i "$KEY")
  [[ -n $JUMP ]] && keyopt+=(-J "$JUMP")
  ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 \
      ${keyopt[@]+"${keyopt[@]}"} -p "$sport" -L "$lport:127.0.0.1:$rport" "$USER_@$HOST"
  local pid
  pid=$(pgrep -f -- "-L $lport:127.0.0.1:$rport" | tail -1 || true)
  [[ -n $pid ]] && echo "$pid" > "$pidf"
  echo "$name: 127.0.0.1:$lport -> $HOST:$sport (guest 127.0.0.1:$rport)"
}

down_one() {
  local name=$1 pidf=.build/tunnel/$name.pid
  if [[ -f $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    kill "$(cat "$pidf")" && echo "$name: down"
  else
    echo "$name: not running"
  fi
  rm -f "$pidf"
}

case "${1:-up}" in
  up)
    for l in "${LINKS[@]}"; do read -r n s lp rp <<<"$l"; up_one "$n" "$s" "$lp" "$rp"; done
    ;;
  down)
    for l in "${LINKS[@]}"; do read -r n _ _ _ <<<"$l"; down_one "$n"; done
    ;;
  status)
    for l in "${LINKS[@]}"; do
      read -r n _ lp _ <<<"$l"
      out=$(curl -s -m 4 "http://127.0.0.1:$lp/v1/models" || true)
      if [[ -n $out ]]; then
        echo "$n: up  $(echo "$out" | python3 -c 'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || echo '?')"
      else
        echo "$n: unreachable on 127.0.0.1:$lp (run: scripts/tunnel.sh up)"
      fi
    done
    ;;
  *)
    echo "usage: $0 up|down|status" >&2; exit 2
    ;;
esac
