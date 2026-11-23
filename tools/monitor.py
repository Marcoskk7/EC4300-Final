"""One-screen status: GPU, vLLM queue, per-run progress and parse-failure rate.

Reads run logs (runs/*/*.log) and checkpoints, so it never touches the runner process.
vLLM stats come from /metrics on the endpoint in the model configs; skipped when offline.

Usage:
  python tools/monitor.py                # print once
  python tools/monitor.py --watch 30     # refresh every 30 s
"""

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SAMPLE_RE = re.compile(r"^Sample (\d+)/(\d+)$", re.M)
PARSE_RE = re.compile(r"Parse success rate: ([\d.]+)%")
METRIC_KEYS = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:gpu_cache_usage_perc", "vllm:avg_generation_throughput_toks_per_s")


def expand_env(s):
    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), s)


def metrics_url(base_url):
    b = expand_env(base_url).rstrip("/")
    return b[: -len("/v1")] + "/metrics" if b.endswith("/v1") else b + "/metrics"


def vllm_stats(host_filter=None):
    out = []
    for cfg in sorted((ROOT / "configs/models").glob("*.yaml")):
        model = yaml.safe_load(cfg.read_text())
        url = metrics_url(model["base_url"])
        if host_filter and host_filter not in url:
            continue
        try:
            text = urllib.request.urlopen(url, timeout=2).read().decode()
        except Exception:
            out.append((model["name"], url, None))
            continue
        vals = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for k in METRIC_KEYS:
                if line.startswith(k + "{"):
                    vals[k] = float(line.rsplit(" ", 1)[1])
        out.append((model["name"], url, vals))
    return out


def gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def run_progress():
    rows = []
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        meta_p = run / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        for log in sorted(run.glob("*.log")):
            text = log.read_text(errors="replace")
            m = SAMPLE_RE.findall(text)
            done, total = (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)
            parses = [float(x) / 100 for x in PARSE_RE.findall(text)]
            finished = (run / f"{log.stem}.json").exists()
            rows.append({
                "run": run.name,
                "task": log.stem,
                "done": total if finished else done,
                "total": total,
                "parse": (sum(parses) / len(parses) if parses else None),
                "mtime": log.stat().st_mtime,
            })
    return rows


def render():
    lines = []
    g = gpu()
    lines.append(f"GPU: {g}" if g else "GPU: nvidia-smi unavailable (local machine?)")
    for name, url, vals in vllm_stats():
        if vals is None:
            lines.append(f"vLLM {name}: unreachable ({url})")
        elif not vals:
            lines.append(f"vLLM {name}: up, no traffic yet")
        else:
            lines.append(
                f"vLLM {name}: running={vals.get('vllm:num_requests_running', 0):g} "
                f"waiting={vals.get('vllm:num_requests_waiting', 0):g} "
                f"kv={100 * vals.get('vllm:gpu_cache_usage_perc', 0):.0f}% "
                f"gen={vals.get('vllm:avg_generation_throughput_toks_per_s', 0):.0f} tok/s"
            )
    rows = run_progress()
    if not rows:
        lines.append("runs: none yet")
    for r in rows:
        age = time.time() - r["mtime"]
        pct = 100 * r["done"] / r["total"] if r["total"] else 0
        parse = "-" if r["parse"] is None else f"{100 * r['parse']:.1f}%"
        flag = ""
        if r["parse"] is not None and r["parse"] < 0.9:
            flag = "  << parse < 90%: check prompt/format"
        lines.append(
            f"{r['run']}/{r['task']}: {r['done']}/{r['total']} ({pct:.0f}%) "
            f"parse={parse} updated {age / 60:.0f}m ago{flag}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", type=float, help="refresh interval in seconds")
    args = p.parse_args()
    if not args.watch:
        print(render())
        return
    while True:
        print("\033[2J\033[H" + time.strftime("%H:%M:%S ") + render(), flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
