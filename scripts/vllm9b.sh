#!/usr/bin/env bash
# Control the 9B vLLM server (the fine-tuning track). Training and serving share one GPU and
# cannot coexist: bf16 weights are 16.7 GiB and the server reserves ~27.7 GB at 0.85.
#
#   scripts/vllm9b.sh start|stop|restart|status
#
# Stop it before a training run, start it again before evaluation. The 27B server belongs to
# the harness track and is deliberately not touched here.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${HYPOSPACE_SSH_HOST:-10.1.3.100}
USER_=${HYPOSPACE_SSH_USER:-isaac}
PORT=${HYPOSPACE_PORT_9B:-20050}
DIR=${EC4300_9B_DIR:-/opt/data/private/isaaclab/ec4300-9b}
KEY=${HYPOSPACE_SSH_KEY:-}

keyopt=()
[[ -n $KEY ]] && keyopt=(-i "$KEY")
ssh9() { ssh -o ConnectTimeout=10 ${keyopt[@]+"${keyopt[@]}"} -p "$PORT" "$USER_@$HOST" "$@"; }

cmd=${1:-status}
case "$cmd" in
  start)
    ssh9 "cd $DIR && nohup ./scripts/start_vllm_9b.sh > logs/vllm_9b_\$(date +%Y%m%d_%H%M).log 2>&1 & echo started"
    ;;
  stop)
    # [v]llm keeps pkill from matching the shell that runs it (and killing this SSH session).
    ssh9 "pkill -f '[v]llm serve' && echo stopped || echo 'not running'"
    ;;
  restart)
    "$0" stop
    sleep 3
    "$0" start
    ;;
  status)
    ssh9 "curl -s -m 3 http://127.0.0.1:8001/v1/models | head -c 200; echo; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"
    ;;
  *)
    echo "usage: $0 start|stop|restart|status" >&2
    exit 2
    ;;
esac
