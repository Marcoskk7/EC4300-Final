"""Replace whole per-sample records of a causal result with those of a rerun, then recompute.

Used to rerun the instances that hit max_tokens at a larger max_tokens: every record whose
observation_set_id appears in the rerun is swapped in as a whole (original order kept); run-level
statistics, tokens, errors and the truncation summary are recomputed from the per-sample records
with merge_shards.merge, i.e. exactly as upstream.

Usage: python tools/replace_samples.py OUT.json BASE.json RERUN.json [RERUN_MAX_TOKENS]
"""

import json
import sys
from pathlib import Path

from merge_shards import merge


def _totals(per_sample: list[dict]) -> dict:
    """Run-level token/cost/error fields that merge() sums, rebuilt from per-sample records."""
    err_types: dict[str, int] = {}
    for r in per_sample:
        for k, v in r["error_summary"]["error_types"].items():
            err_types[k] = err_types.get(k, 0) + v
    return {
        "token_usage": {
            k: sum(r["token_usage"][k] for r in per_sample) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "cost": {"total_cost": sum(r["cost"] for r in per_sample)},
        "error_summary": {"total_errors": sum(r["error_summary"]["total_errors"] for r in per_sample), "error_types": err_types},
    }


def replace(base: dict, rerun: dict, max_tokens: int | None = None) -> dict:
    new = {r["observation_set_id"]: r for r in rerun["per_sample_results"]}
    known = {r["observation_set_id"] for r in base["per_sample_results"]}
    if not set(new) <= known:
        raise SystemExit(f"rerun ids not in base: {sorted(set(new) - known)[:10]}")
    per_sample = [new.get(r["observation_set_id"], r) for r in base["per_sample_results"]]
    out = merge([{**base, **_totals(per_sample), "per_sample_results": per_sample}])
    out["n_shards"] = base.get("n_shards", 1)
    out["replaced_samples"] = {"n": len(new), "ids": sorted(new), "from_run_id": rerun["run_id"], "max_tokens": max_tokens}
    return out


def main() -> None:
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    out, base, rerun = (Path(p) for p in sys.argv[1:4])
    mt = int(sys.argv[4]) if len(sys.argv) == 5 else None
    res = replace(json.loads(base.read_text()), json.loads(rerun.read_text()), mt)
    out.write_text(json.dumps(res, indent=2))
    print(f"replaced {res['replaced_samples']['n']} samples -> {out}")


if __name__ == "__main__":
    main()
