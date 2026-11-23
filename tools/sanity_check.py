"""Sanity-check the official HypoSpace evaluators for all three domains with scripted fake LLMs.

Runs no API calls. Each domain is fed a known policy (oracle / mode-collapse / all-invalid /
unparseable) whose (VR, NR, RR) is known analytically, so a mismatch means the evaluator
or our understanding of it is wrong.

Usage: .venv/bin/python tools/sanity_check.py [--domain causal|3d|boolean|all]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from bench import ensure  # noqa: E402

HS = ensure()
for sub in ("causal", "3d", "boolean"):
    sys.path.insert(0, str(HS / sub))

from modules.llm_interface import LLMInterface  # noqa: E402


class ScriptedLLM(LLMInterface):
    """Replays a fixed list of responses, cycling when exhausted."""

    def __init__(self, responses):
        self.responses, self.i = responses, 0

    def query(self, prompt):
        r = self.responses[self.i % len(self.responses)]
        self.i += 1
        return r

    def get_name(self):
        return "scripted"


def run_policies(name, bench, obs_set, n_queries, policies):
    print(f"[{name}] {obs_set.get('observation_set_id', obs_set.get('observation_id'))} "
          f"|H_O|={n_queries - 0}d n_queries={n_queries}")
    ok = True
    for label, (responses, expected) in policies.items():
        r = bench.evaluate_single_observation_set(
            ScriptedLLM(responses), obs_set, n_queries=n_queries, verbose=False, max_retries=1)
        got = (r["valid_rate"], r["novelty_rate"], r["recovery_rate"])
        match = all(abs(a - b) < 1e-6 for a, b in zip(got, expected))
        ok &= match
        print(f"  {'OK ' if match else 'BAD'} {label:30s} VR={got[0]:.3f} NR={got[1]:.3f} "
              f"RR={got[2]:.3f}  expected {tuple(round(x, 3) for x in expected)}")
    return ok


def causal(data):
    from run_causal_benchmark import CausalBenchmarkEnhanced
    from generate_causal_dataset import CausalDatasetGenerator

    bench = CausalBenchmarkEnhanced(str(data))
    obs_set = next(s for s in bench.all_observation_sets if s["n_compatible_graphs"] >= 8)
    gts = [g["edges"] for g in obs_set["ground_truth_graphs"]]
    gt_set = {tuple(sorted(map(tuple, e))) for e in gts}
    invalid = next(d.edges for d in CausalDatasetGenerator.generate_all_dags(bench.nodes)
                   if tuple(sorted(d.edges)) not in gt_set)
    n = len(gts)

    def fmt(edges):
        return "Graph: " + (", ".join(f"{u}->{v}" for u, v in edges) if edges else "No edges")

    return run_policies("causal", bench, obs_set, n, {
        "oracle (every GT once)": ([fmt(e) for e in gts], (1.0, 1.0, 1.0)),
        "mode collapse (GT[0])": ([fmt(gts[0])], (1.0, 1 / n, 1 / n)),
        "always invalid": ([fmt(invalid)], (0.0, 1 / n, 0.0)),
        "unparseable": (["I think the answer is a DAG."], (0.0, 0.0, 0.0)),
    })


def three_d(data):
    from run_3d_benchmark import Benchmark3D

    bench = Benchmark3D(str(data))
    obs_set = next(s for s in bench.all_observation_sets if s["n_compatible_structures"] >= 9)
    gts = obs_set["ground_truth_structures"]
    n = len(gts)

    grid = int(bench.metadata["grid_size"])

    def fmt(gt):
        rows = ["Structure:"]
        for i, layer in enumerate(gt["layers"], 1):
            rows.append(f"Layer {i}:")
            for r in range(grid):  # layers are stored flattened; parser wants one line per row
                rows.append(" ".join(layer[r * grid:(r + 1) * grid]))
        return "\n".join(rows)

    # Invalid: one block on the top view's first filled cell set to full height, but a
    # structure that contradicts the top view is out of the admissible set.
    top = obs_set["observation"]
    fake = [c if top[i] == "1" else "0" for i, c in enumerate(top)]
    fake = "".join("1" if i != top.index("1") else "0" for i in range(len(top)))

    return run_policies("3d", bench, obs_set, n, {
        "oracle (every GT once)": ([fmt(g) for g in gts], (1.0, 1.0, 1.0)),
        "mode collapse (GT[0])": ([fmt(gts[0])], (1.0, 1 / n, 1 / n)),
        "contradicts top view": ([f"Structure:\nLayer 1:\n{' '.join(fake)}"], (0.0, 1 / n, 0.0)),
        "unparseable": (["I am not sure."], (0.0, 0.0, 0.0)),
    })


def boolean(data):
    from boolean_benchmark import BooleanBenchmarkRefined

    bench = BooleanBenchmarkRefined(str(data))
    obs_set = next(s for s in bench.all_observation_sets if s["n_compatible_expressions"] >= 5)
    gts = [g["formula"] for g in obs_set["ground_truth_expressions"]]
    n = len(gts)

    return run_policies("boolean", bench, obs_set, n, {
        "oracle (every GT once)": ([f"Expression: {g}" for g in gts], (1.0, 1.0, 1.0)),
        "mode collapse (GT[0])": ([f"Expression: {gts[0]}"], (1.0, 1 / n, 1 / n)),
        "unparseable": (["???"], (0.0, 0.0, 0.0)),
        # Known evaluator hole: validity checks consistency only, not space membership,
        # so junk text can validate when every observed output is 0. See docs/protocol.md §5.
        "junk text (expects VR=1.0)": (["No idea"], (1.0, 0.0, 0.0)),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all", choices=["causal", "3d", "boolean", "all"])
    a = ap.parse_args()
    data = {
        "causal": ROOT / "data/causal/n4.json",
        "3d": ROOT / "data/3d/g3_h3_b3.json",
        "boolean": ROOT / "data/boolean/basic_d2.json",
    }
    fns = {"causal": causal, "3d": three_d, "boolean": boolean}
    domains = list(fns) if a.domain == "all" else [a.domain]
    ok = all([fns[d](data[d]) for d in domains])
    print("\nALL CHECKS PASSED" if ok else "\nMISMATCH — evaluator behaviour differs from expectation")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
