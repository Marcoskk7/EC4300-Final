"""Merge round-robin shards of one causal run into a single upstream-shaped result.

`run_causal_benchmark.py --shard i/n` evaluates sampled_sets[i::n] of the same seeded sample, so
interleaving the shards restores the unsharded order. Statistics are recomputed exactly as
run_causal_benchmark.run_benchmark does (np.mean/std/var/min/max + one-sample t-test vs 0).

Usage: python tools/merge_shards.py OUT.json SHARD0.json SHARD1.json ...
"""

import json
import sys
from datetime import datetime
from itertools import chain, zip_longest
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = ("parse_success_rate", "valid_rate", "novelty_rate", "recovery_rate")


def _stat(rates: list[float]) -> dict:
    if not rates:
        return {"mean": 0, "std": 0, "var": 0, "min": 0, "max": 0, "p_value": None}
    p = stats.ttest_1samp(rates, 0)[1] if len(rates) >= 2 else None
    return {
        "mean": float(np.mean(rates)),
        "std": float(np.std(rates)),
        "var": float(np.var(rates)),
        "min": float(np.min(rates)),
        "max": float(np.max(rates)),
        "p_value": None if p is None else float(p),
    }


def merge(shards: list[dict]) -> dict:
    """Shards must be given in shard-index order (0..n-1)."""
    gap = object()
    per_sample = [
        r for r in chain.from_iterable(zip_longest(*(s["per_sample_results"] for s in shards), fillvalue=gap))
        if r is not gap
    ]
    n = len(per_sample)
    nq = shards[0]["n_queries_per_sample"]
    denom = n * (nq or 1)

    tok = {k: sum(s["token_usage"][k] for s in shards) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    cost = sum(s["cost"]["total_cost"] for s in shards)
    n_err = sum(s["error_summary"]["total_errors"] for s in shards)
    err_types: dict[str, int] = {}
    for s in shards:
        for k, v in s["error_summary"]["error_types"].items():
            err_types[k] = err_types.get(k, 0) + v

    first = shards[0]
    return {
        "run_id": first["run_id"],
        "llm_name": first["llm_name"],
        "n_samples": n,
        "n_queries_per_sample": nq,
        "query_multiplier": first["query_multiplier"],
        "query_mode": first["query_mode"],
        "seed": first["seed"],
        "timestamp": datetime.now().isoformat(),
        "max_edges_constraint": first["max_edges_constraint"],
        "n_shards": len(shards),
        "statistics": {m: _stat([r.get(m, 1.0) if m == "parse_success_rate" else r[m] for r in per_sample]) for m in METRICS},
        "token_usage": {
            **tok,
            "avg_tokens_per_sample": tok["total_tokens"] / n if n else 0,
            "avg_tokens_per_query": tok["total_tokens"] / denom if n else 0,
        },
        "cost": {
            "total_cost": cost,
            "avg_cost_per_sample": cost / n if n else 0,
            "avg_cost_per_query": cost / denom if n else 0,
        },
        "error_summary": {"total_errors": n_err, "error_types": err_types, "error_rate": n_err / denom if n else 0},
        "truncation": {
            "n_truncated_queries": sum(len(r.get("truncated_queries", [])) for r in per_sample),
            "truncated_sample_ids": [r["observation_set_id"] for r in per_sample if r.get("truncated_queries")],
        },
        "per_sample_results": per_sample,
    }


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out, paths = Path(sys.argv[1]), sys.argv[2:]
    out.write_text(json.dumps(merge([json.loads(Path(p).read_text()) for p in paths]), indent=2))
    print(f"merged {len(paths)} shards -> {out}")


if __name__ == "__main__":
    main()
