#!/usr/bin/env bash
# Benchmark venv only.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv -p 3.11 .venv
uv pip install -p .venv -r env/bench.txt
echo "ok: .venv ($(.venv/bin/python -V))"
