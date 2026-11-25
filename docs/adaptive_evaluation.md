# Feedback-guided evaluation on Qwen3.5-9B

This extension compares baseline, fixed repair, random action selection and UCB. It does not fine-tune model weights. UCB is an established algorithm; the contribution is its hypothesis-discovery integration and controlled evaluation.

## Setup

Use Python 3.11 and install `pip install -r env/adaptive-win-tested.txt`. Serve the model with an OpenAI-compatible endpoint and served name `qwen3.5-9b`. The supplied model config fixes temperature 0.7, maximum output 1024 tokens, and thinking disabled. No model weights or credentials are included.

```bash
python -m unittest discover -s tools -p 'test_*causal.py'
python -m unittest discover -s tools -p test_cross_domain.py
python tools/check_cross_domain.py
```

The exhaustive check validates all 4,254 stored cross-domain reference candidates. Cross-domain Boolean parsing uses the dataset's structural identity rules. The adapter accepts NOT+OR in Extended and counts primitive NOR as one depth level in Full, fixing inconsistencies in the original validator. All four methods use the same corrections. Historical baselines are not interchangeable with these reruns.

## Run

Use a new output directory for a new experiment. Replace the example endpoint with your own model service; start with smoke validation.

```bash
python tools/run_formal_causal.py --smoke --workers 4 --base-url http://127.0.0.1:8001/v1 --output runs/causal_smoke
python tools/run_formal_causal.py --workers 4 --base-url http://127.0.0.1:8001/v1 --output runs/causal_formal
python tools/compare_adaptive_seeds.py runs/causal_formal/seed_33550336 runs/causal_formal/seed_33550337 --output runs/causal_formal/analysis
```

For cross-domain evaluation, set `PYTHONHASHSEED=0` before starting Python (`export PYTHONHASHSEED=0` in bash; `$env:PYTHONHASHSEED='0'` in PowerShell).

```bash
python tools/run_cross_domain.py --smoke --workers 32 --base-url http://127.0.0.1:8001/v1 --output runs/cross_smoke
python tools/run_cross_domain.py --workers 32 --base-url http://127.0.0.1:8001/v1 --output runs/cross_formal
```

Cross-domain summaries are generated automatically after completion. The default worker count is now 32; the reported causal run used 4 and cross-domain runs used 32. Set concurrency to the capacity of your service. Do not expect bitwise-identical responses under different serving conditions.

## Protocol and results

- Two seeds: 33550336 and 33550337; four methods on the same instances.
- Causal: 200 predefined n=4 instances, 2 × reference-set size requests; 63,920 formal requests.
- Cross-domain: all 129 3D, 35 Basic, 78 Extended and 80 Full instances, 1 × reference-set size; 34,032 formal requests.
- Smoke: two instances per cross-domain dataset, three requests per task, for plumbing checks only. These instances overlap full evaluation; no untouched-test claim.
- Feedback uses observations and public constraints. Reference keys are used after generation for scoring; reference-set size defines the benchmark budget.
- Reward is one for a previously unseen valid candidate, zero otherwise. Repair is available only after an invalid response. Statistics reset per instance.
- Equal request budgets do not imply equal token budgets. Bootstrap intervals average seeds within instances before resampling instances; no multiple-comparison correction.

Read [the full report](adaptive_results_zh.md) and `results/adaptive_9b/*_comparison.json`. UCB does not uniformly beat fixed repair or random selection.

## Logs and recovery

Each output has a frozen manifest, status, per-request event and attempt journals, and per-instance results. Keep raw runs locally; generated runs and remote launch records are excluded from this public change. Completed jobs can be reused only under the same manifest. Never remove a lock without checking its PID. Cross-domain unresolved requests stop for explicit auditing rather than silently retrying. The causal runner marks an interrupted unknown request as budget-consuming; the reported completed causal run had zero such events and zero context rejections.

Frozen server copies retain the exact code used for the completed experiments; local default concurrency was subsequently increased from 4 to 32. Full raw logs are retained locally, not included in the public summary directory.
