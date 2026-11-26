#!/usr/bin/env bash
# Evaluate the local autoresearch/policy.py on a GPU node, where vLLM is on localhost.
#
#   autoresearch/remote_run.sh [--smoke] > run.log 2>&1        # 9B (default)
#   AR_RUN=27b autoresearch/remote_run.sh > run.log 2>&1       # 27B
#   AR_FINAL=1 autoresearch/remote_run.sh > final.log 2>&1     # once, after a loop: final_eval.py
#
# Pushes policy.py, refuses to run if the node's locked files differ from local ones,
# runs the harness detached on the node (survives SSH drops), polls until it finishes,
# prints the run output and copies the run directory into runs/autoresearch/.
set -euo pipefail
cd "$(dirname "$0")/.."

MINI=${AR_MINI:-marcoskk7@100.81.134.74}
NODE=${AR_NODE:-isaac@10.1.3.100}
case ${AR_RUN:-9b} in
  9b)  DEFAULT_PORT=20050; DEFAULT_DIR=/opt/data/private/isaaclab/ec4300-9b/autoresearch-repo
       DEFAULT_MODEL=configs/models/qwen3.5-9b.yaml;  DEFAULT_WORKERS=32; DEFAULT_RUN_TIMEOUT=1500; DEFAULT_FINAL_TIMEOUT=5400 ;;
  27b) DEFAULT_PORT=20049; DEFAULT_DIR=/opt/data/private/isaaclab/ec4300/autoresearch-repo
       DEFAULT_MODEL=configs/models/qwen3.8-27b.yaml; DEFAULT_WORKERS=8;  DEFAULT_RUN_TIMEOUT=5400; DEFAULT_FINAL_TIMEOUT=10800 ;;
  *) echo "AR_RUN must be 9b or 27b" >&2; exit 2 ;;
esac
NODE_PORT=${AR_NODE_PORT:-$DEFAULT_PORT}
DIR=${AR_DIR:-$DEFAULT_DIR}
MODEL=${AR_MODEL:-$DEFAULT_MODEL}
WORKERS=${AR_WORKERS:-$DEFAULT_WORKERS}
PY=${AR_PY:-../repo/.venv/bin/python}
if [[ ${AR_FINAL:-0} == 1 ]]; then
  SCRIPT=autoresearch/final_eval.py; ENVV="AUTORESEARCH_FINAL=1"; TIMEOUT=${AR_TIMEOUT:-$DEFAULT_FINAL_TIMEOUT}
else
  SCRIPT=autoresearch/harness.py; ENVV=""; TIMEOUT=${AR_TIMEOUT:-$DEFAULT_RUN_TIMEOUT}
fi
POLL=${AR_POLL:-30}
LOCKED=(autoresearch/harness.py autoresearch/final_eval.py tools/adaptive_causal.py tools/cross_domain.py "$MODEL")
SSH_OPTS=(-o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

on_node() {  # run stdin as a bash script on the node
  ssh "${SSH_OPTS[@]}" "$MINI" \
    "ssh -o ConnectTimeout=10 -o ServerAliveInterval=15 -p $NODE_PORT $NODE bash -s"
}

# On the node the config points at each node's own vLLM port; locally it is the tunnel port.
node_config() { sed -e 's#127.0.0.1:18001#127.0.0.1:8001#' -e 's#127.0.0.1:18000#127.0.0.1:8000#' "$1"; }
local_sums=$(for f in "${LOCKED[@]}"; do
  if [[ $f == configs/* ]]; then node_config "$f"; else cat "$f"; fi | shasum -a 256 | cut -c1-64
done)
policy_b64=$(base64 < autoresearch/policy.py | tr -d '\n')
id=$(date +%Y%m%d%H%M%S)-$$

on_node <<EOF
set -euo pipefail
cd $DIR
remote_sums=\$(for f in ${LOCKED[*]}; do sha256sum "\$f" | cut -c1-64; done)
if [[ "\$remote_sums" != "$local_sums" ]]; then
  echo "LOCKED FILES DIFFER between local repo and node ($DIR); ask the human to resync." >&2; exit 3
fi
exec 9>/tmp/ar_run.lock
flock -n 9 || { echo "another autoresearch run is active on the node" >&2; exit 4; }
echo "$policy_b64" | base64 -d > autoresearch/policy.py
nohup setsid bash -c 'flock 9; $ENVV timeout $TIMEOUT $PY $SCRIPT --model $MODEL --workers $WORKERS $* > /tmp/ar_$id.log 2>&1; echo \$? > /tmp/ar_$id.rc' 9>/tmp/ar_run.lock >/dev/null 2>&1 &
EOF
echo "started on $NODE_PORT: id=$id model=$MODEL workers=$WORKERS"

while :; do
  sleep "$POLL"
  rc=$(echo "cat /tmp/ar_$id.rc 2>/dev/null || true" | on_node 2>/dev/null || true)
  [[ -n $rc ]] && break
done

echo "cat /tmp/ar_$id.log" | on_node
run_dir=$(echo "grep '^run_dir:' /tmp/ar_$id.log | awk '{print \$2}'" | on_node || true)
if [[ -n $run_dir ]]; then
  mkdir -p runs/autoresearch
  echo "tar czf - -C $DIR $run_dir" | on_node | tar xzf - -C .
  echo "fetched:          $run_dir"
fi
echo "exit_code:        $rc"
exit "$rc"
