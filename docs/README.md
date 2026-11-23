# Docs

| File | What it answers |
|---|---|
| `protocol.md` | Evaluation protocol: budget, subsets, seed, holdout, reporting rules. |
| `baselines.md` | Measured numbers: paper Tables 1–6, Qwen3.5-9B / Qwen3.8-27B baselines, failure breakdown. |
| `hardware.md` | GPU nodes, vLLM endpoints and mandatory flags, 3D prompt. |
| `lessons.md` | Pitfalls already paid for: environment, data formats, serving. |
| `papers/HypoSpace_arXiv_2510.15614.pdf` | Paper PDF, source of Tables 1–6. |

`results/leaderboard.{md,csv}` is generated from `runs/` by `make report`; `baselines.md` is the
annotated snapshot with paper numbers and caveats.
