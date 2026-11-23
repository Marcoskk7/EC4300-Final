"""Run a HypoSpace evaluation suite against one model endpoint.

Each invocation creates runs/<YYYYMMDD>_<model>_<suite>[_<tag>]/ containing
  meta.json            model/suite/budget/git provenance
  <task>.json          upstream runner output (statistics + per_sample_results)
  <task>.log           runner stdout/stderr (tailed by tools/monitor.py)
Re-running with the same --run-dir resumes: finished tasks are skipped.

Usage:
  python tools/run_eval.py --model configs/models/qwen3.5-9b.yaml --suite configs/suites/main.yaml
  python tools/run_eval.py --model ... --suite ... --served-model my-lora --tag sft-v1
  python tools/run_eval.py --model ... --suite ... --tasks causal_n4,3d
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from bench import ensure  # noqa: E402
from merge_shards import merge  # noqa: E402

HS = ensure()

RUNNERS = {
    "causal": HS / "causal/run_causal_benchmark.py",
    "3d": HS / "3d/run_3d_benchmark.py",
    "boolean": HS / "boolean/boolean_benchmark.py",
}
# 3D must forbid prose: Qwen otherwise reasons past max_tokens and parse rate drops to 0.
SYSTEM_PROMPTS = {
    "causal": "You are an expert in causal inference and graph theory.",
    "3d": "You are an expert in 3D spatial reasoning. Answer with the requested Structure/Layer format only, no explanation.",
    "boolean": "You are an expert in Boolean logic and expression synthesis.",
}


def expand_env(s):
    """Expand ${VAR} and ${VAR:-default}."""
    return re.sub(
        r"\$\{(\w+)(?::-([^}]*))?\}",
        lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""),
        s,
    )


def git_state():
    def git(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def runner_config(model, domain):
    return {
        "llm": {
            "type": "local",
            "models": {"local": model["served_model"]},
            "api_keys": {},  # key comes from HYPOSPACE_API_KEY, never from files
            "temperature": model["temperature"],
            "base_url": model["base_url"],
            "max_tokens": model["max_tokens"],
            "system_prompt": model.get("system_prompts", {}).get(domain, SYSTEM_PROMPTS[domain]),
            "extra_body": model.get("extra_body", {}),
        },
        "benchmark": {"verbose": True},
    }


def runner_args(task, seed, out, cfg, run_dir):
    a = [
        sys.executable, "-u", str(RUNNERS[task["domain"]]),
        "--dataset", str(ROOT / task["dataset"]),
        "--config", str(cfg),
        "--n-samples", str(task["n_samples"]),
        "--seed", str(seed),
        "--output", str(out),
    ]
    if "n_queries" in task:
        a += ["--n-queries", str(task["n_queries"])]
    else:
        a += ["--query-multiplier", str(task["multiplier"])]
    if "gt_filter" in task:
        a += ["--gt-filter", task["gt_filter"]]
    if task["domain"] != "boolean":  # boolean runner has no checkpoint flag
        a += ["--checkpoint-dir", str(run_dir / "checkpoints")]
    return a


def run_sharded(name, task, argv, out, run_dir, n):
    """Run n runner processes on round-robin shards in parallel, then merge into `out`.

    Finished shards (<task>.shardI.json) are kept, so a re-run only redoes the missing ones.
    """
    shard_out = [run_dir / f"{name}.shard{i}.json" for i in range(n)]
    procs = []
    for i, so in enumerate(shard_out):
        if so.exists():
            print(f"[skip] {name} shard {i}/{n}: {so.name} exists")
            continue
        a = list(argv)
        a[a.index("--output") + 1] = str(so)
        a += ["--shard", f"{i}/{n}"]
        log = open(run_dir / f"{name}.shard{i}.log", "w")
        procs.append((i, log, subprocess.Popen(a, cwd=RUNNERS[task["domain"]].parent, stdout=log, stderr=subprocess.STDOUT)))
    print(f"[run ] {name}: {len(procs)} shard processes (logs: {name}.shard*.log)", flush=True)
    rc = 0
    for i, log, proc in procs:
        code = proc.wait()
        log.close()
        if code != 0 or not shard_out[i].exists():
            print(f"[FAIL] {name} shard {i}/{n}: exit {code}")
            rc = rc or code or 1
            continue
        res = json.loads(shard_out[i].read_text())
        if res.get("error_summary", {}).get("error_types", {}).get("connection_error", 0):
            shard_out[i].rename(shard_out[i].with_suffix(".connfail.json"))  # so a re-run redoes it
            print(f"[FAIL] {name} shard {i}/{n}: connection errors")
            rc = rc or 1
    if rc:
        return rc
    out.write_text(json.dumps(merge([json.loads(p.read_text()) for p in shard_out]), indent=2))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="configs/models/*.yaml")
    p.add_argument("--suite", required=True, help="configs/suites/*.yaml")
    p.add_argument("--tasks", help="comma-separated subset of suite tasks")
    p.add_argument("--served-model", help="override served_model (e.g. a vLLM LoRA name)")
    p.add_argument("--base-url", help="override base_url")
    p.add_argument("--tag", help="suffix for the run directory, e.g. sft-v1")
    p.add_argument("--run-dir", help="resume into an existing run directory")
    p.add_argument("--shards", type=int, default=1,
                   help="causal only: split instances round-robin over N parallel runner processes "
                        "(needs vLLM --max-num-seqs >= N); shards are merged into <task>.json")
    args = p.parse_args()

    model = yaml.safe_load(Path(args.model).read_text())
    suite = yaml.safe_load(Path(args.suite).read_text())
    if args.served_model:
        model["served_model"] = args.served_model
    if args.base_url:
        model["base_url"] = args.base_url
    model["base_url"] = expand_env(model["base_url"])

    tasks = suite["tasks"]
    if args.tasks:
        wanted = args.tasks.split(",")
        unknown = set(wanted) - set(tasks)
        if unknown:
            sys.exit(f"unknown tasks {sorted(unknown)}; suite has {sorted(tasks)}")
        tasks = {k: tasks[k] for k in wanted}

    suite_name = Path(args.suite).stem
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        name = f"{datetime.now():%Y%m%d}_{model['served_model']}_{suite_name}"
        run_dir = ROOT / "runs" / (f"{name}_{args.tag}" if args.tag else name)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": model["name"],
        "served_model": model["served_model"],
        "base_url": model["base_url"],
        "suite": suite_name,
        "tag": args.tag,
        "seed": suite["seed"],
        "tasks": tasks,
        "sampling": {k: model[k] for k in ("temperature", "max_tokens")} | {"extra_body": model.get("extra_body")},
        "git": git_state(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "shards": args.shards,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"run dir: {run_dir}")

    failed = []
    for name, task in tasks.items():
        out = run_dir / f"{name}.json"
        if out.exists():
            print(f"[skip] {name}: {out.name} exists")
            continue
        cfg = run_dir / f"config.{task['domain']}.yaml"
        cfg.write_text(yaml.safe_dump(runner_config(model, task["domain"]), sort_keys=False))
        argv = runner_args(task, suite["seed"], out, cfg, run_dir)
        if args.shards > 1 and task["domain"] == "causal":
            rc = run_sharded(name, task, argv, out, run_dir, args.shards)
        else:
            print(f"[run ] {name} -> {out.name} (log: {name}.log)", flush=True)
            with open(run_dir / f"{name}.log", "w") as log:
                rc = subprocess.run(
                    argv, cwd=RUNNERS[task["domain"]].parent, stdout=log, stderr=subprocess.STDOUT,
                ).returncode
        if rc != 0 or not out.exists():
            failed.append(name)
            print(f"[FAIL] {name}: exit {rc}, see {run_dir / (name + '.log')}")
            continue
        res = json.loads(out.read_text())
        # Upstream runners record endpoint outages as per-query errors and still write a result;
        # such a file must not count as finished, or a resume would skip the task.
        conn_errs = res.get("error_summary", {}).get("error_types", {}).get("connection_error", 0)
        if conn_errs:
            out.rename(out.with_suffix(".connfail.json"))
            failed.append(name)
            print(f"[FAIL] {name}: {conn_errs} connection errors; moved to {out.stem}.connfail.json")
            continue
        s = res["statistics"]
        print(
            f"[done] {name}: VR={s['valid_rate']['mean']:.3f} "
            f"NR={s['novelty_rate']['mean']:.3f} RR={s['recovery_rate']['mean']:.3f}"
        )

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
