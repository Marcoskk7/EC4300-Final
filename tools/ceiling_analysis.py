"""Measure the structural ceiling of the official protocol, per domain.

For each observation set the official runner draws n_queries = max(1, |H_O| * multiplier)
samples. An oracle that emits every admissible hypothesis once reaches RR=1.0 only if the
budget covers it; with multiplier m the ceiling is min(1, m). More useful in practice: the
ceiling for a *non-enumerating* method is bounded by the tail mass it can reach, so we report
the distribution of |H_O| per domain/setting, which decides where the headroom is.

Usage: .venv/bin/python tools/ceiling_analysis.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def summarize(name, counts):
    counts = sorted(counts)
    print(f"{name:34s} n={len(counts):4d}  |H_O| min={counts[0]:4d} "
          f"median={statistics.median(counts):6.1f} max={counts[-1]:4d} "
          f"mean={statistics.mean(counts):6.1f}  ==1: {sum(c == 1 for c in counts)}")


def causal():
    for n in (3, 4):
        d = json.load(open(DATA / f"causal/n{n}.json"))
        sets = [s for v in d["datasets_by_n_observations"].values() for s in v]
        summarize(f"causal n={n} (space {d['metadata']['hypothesis_space_size']})",
                  [s["n_compatible_graphs"] for s in sets])
        for n_obs in sorted(d["datasets_by_n_observations"]):
            sub = d["datasets_by_n_observations"][n_obs]
            big = [s["n_compatible_graphs"] for s in sub if s["n_compatible_graphs"] > 1]
            print(f"    n_observations={n_obs}: {len(sub):4d} sets, "
                  f"{len(big):4d} with |H_O|>1")


def three_d():
    d = json.load(open(DATA / "3d/g3_h3_b3.json"))
    sets = d["observation_sets"]
    summarize(f"3d g=3 h=3 b<=3", [s["n_compatible_structures"] for s in sets])


def boolean():
    for tag in ("basic_d2",):
        d = json.load(open(DATA / f"boolean/{tag}.json"))
        sets = [s for v in d["datasets_by_n_observations"].values() for s in v]
        summarize(f"boolean {tag} (space {d['metadata']['hypothesis_space_size']})",
                  [s["n_compatible_expressions"] for s in sets])
        for n_obs in sorted(d["datasets_by_n_observations"]):
            sub = d["datasets_by_n_observations"][n_obs]
            big = [s["n_compatible_expressions"] for s in sub if s["n_compatible_expressions"] > 1]
            print(f"    n_observations={n_obs}: {len(sub):3d} sets, "
                  f"{len(big):3d} with |H_O|>1")


if __name__ == "__main__":
    causal()
    three_d()
    boolean()
