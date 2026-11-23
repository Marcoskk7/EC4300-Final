# Improving LLM hypothesis-set generation on HypoSpace

Course project (EC4300). Benchmark: **HypoSpace** (ICML 2026 spotlight), which measures whether
an LLM can enumerate the *set* of hypotheses consistent with underdetermined observations in
three domains — causal graphs, 3D voxel reconstruction, and Boolean genetic interactions.

Three metrics, all reported as percentages:

| Metric | Meaning |
|---|---|
| **VR** validity | fraction of proposals consistent with the observations |
| **NR** novelty | fraction of proposals that are not duplicates |
| **RR** recovery | fraction of the admissible set $\mathcal{H}_O$ that was recovered |

Budget is per instance: `n_queries = max(1, floor(|H_O| * multiplier))`. RR is not comparable
across different multipliers, so every row below states its budget.

## Approach

VR is saturated for strong models on causal and 3D, and the paper shows that temperature, top-p,
and a 3× budget do not move RR (their Table 6). The headroom is therefore in *how the query
budget is spent*: the failures are duplicates and invalid proposals, not a lack of capacity.

Two served endpoints:

- Qwen3.5-9B (port 18001) — base weights or a LoRA adapter, scored through the tunnel.
- Qwen3.8-27B (port 18000) — inference-side changes (memory of prior proposals, coverage-guided
  prompting, batching, prefix reuse). The vendored runner issues requests serially and resends
  all history every turn, so the engine has real headroom there.

Both are scored by the same protocol and the same command, so numbers are comparable.

## Results

Zero-shot baselines, `configs/suites/main.yaml`, seed 33550336 (`make report` regenerates this
table from `runs/`):

| Task | n | Budget | Model | parse | VR | NR | RR |
|---|---|---|---|---|---|---|---|
| causal n=4 | 200 | 2.0× | Qwen3.5-9B | 100.0 | 25.9 | 38.1 | 8.3 |
| causal n=4 | 200 | 2.0× | Qwen3.8-27B | 95.7 | 28.7 | 42.0 | 20.7 |
| 3D g3/b3 | 129 | 1.0× | Qwen3.5-9B | 100.0 | 57.1 | 42.5 | 22.7 |
| 3D g3/b3 | 129 | 1.0× | Qwen3.8-27B | 99.9 | 71.7 | 69.9 | 49.8 |
| boolean basic | 35 | 1.0× | Qwen3.5-9B | 100.0 | 80.8 | 32.7 | 26.6 |
| boolean basic | 35 | 1.0× | Qwen3.8-27B | 100.0 | 70.9 | 71.1 | 49.5 |
| boolean extended | 78 | 1.0× | Qwen3.5-9B | 100.0 | 66.1 | 32.1 | 16.5 |
| boolean extended | 78 | 1.0× | Qwen3.8-27B | 100.0 | 50.5 | 56.4 | 36.0 |
| boolean full | 80 | 1.0× | Qwen3.5-9B | 100.0 | 59.7 | 14.3 | 8.0 |
| boolean full | 80 | 1.0× | Qwen3.8-27B | 100.0 | 43.3 | 37.4 | 20.0 |

causal uses `--gt-filter 8-200`: instances with `|H_O| = 1` are free points and are excluded.
The same 200 instance ids are used for every row (`data/splits/causal_n4_test.json`).

## Commands

```bash
make setup                        # .venv from env/bench.txt (Python 3.11)
make test                         # evaluator self-check, scripted LLMs, no network
make tunnel                       # SSH tunnels to both GPU nodes (see docs/hardware.md)
make tunnel-status
make smoke                        # 2 instances x 2 queries per domain against $MODEL
make eval MODEL=configs/models/qwen3.5-9b.yaml SUITE=configs/suites/main.yaml
make eval MODEL=... ARGS="--served-model qwen3.5-9b-sft-v1 --tag sft-v1"
make report                       # results/leaderboard.{md,csv} from runs/
make monitor                      # GPU, vLLM queue, per-run progress
make monitor WATCH=30
```

`make eval` is resumable: rerunning the same `--run-dir` skips finished tasks. Every run writes
`runs/<date>_<model>_<suite>[_<tag>]/meta.json` with the resolved endpoint, sampling settings,
budget, and git commit; `report.py` reads only those files, so no number is ever retyped.

## Layout

```
configs/models/          one file per served endpoint (url, sampling, thinking mode)
configs/suites/          evaluation protocol: tasks, subsets, budgets, seed
scripts/tunnel.sh        SSH tunnels: 18000 -> 27B node, 18001 -> 9B node
data/                    benchmark instances (fixed seed) + held-out instance ids
third_party/HypoSpace/   upstream benchmark, pinned (see third_party/UPSTREAM.md)
patches/                 our edits to upstream, as a patch
tools/bench.py           .build/hypospace = pristine upstream + patches (gitignored)
tools/run_eval.py        the only evaluation entry point
tools/report.py          leaderboard generator
tools/monitor.py         live status
tools/sanity_check.py    evaluator correctness check
runs/                    one directory per evaluation run (results + logs + provenance)
```

## Honesty statement

- Baselines and methods use the same budget for the same task; the multiplier is in every table.
- The 3D and Boolean tasks score the full datasets; causal uses the 200 hard instances listed in
  `data/splits/causal_n4_test.json`; the scored ids are committed so leakage is checkable.
- Known evaluator quirk: the Boolean validator does not check that a proposal belongs to the
  hypothesis space, so a meaningless string can score VR=1 on the 7/35 basic instances whose
  observations are all 0. We report the official numbers and flag affected rows.
- Local models are quantised/served differently from the paper's API models; those rows are an
  open-weights reference, not a reproduction of the paper's table.
