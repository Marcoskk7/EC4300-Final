# Evaluation protocol

One protocol, `configs/suites/main.yaml`, used for every reported number. If a run does not use
it, its numbers go in a separate table with the difference stated.

## 1. Budget

$$n\_queries = \max\left(1, \left\lfloor |\mathcal{H}_O| \cdot \text{multiplier} \right\rfloor\right)$$

per instance, where $|\mathcal{H}_O|$ is the number of hypotheses consistent with the observations.
More queries means more chances to recover the set, so **RR is not comparable across
multipliers**. The paper itself reports that 3× budget does not fix RR (their Table 6), which is
why we compete at the same budget instead of buying coverage.

| Task | multiplier | Why |
|---|---|---|
| causal n=4 | 2.0 | upstream default for this domain; both baselines used it |
| 3D, boolean | 1.0 | upstream default for these domains |

## 2. Tasks and subsets

| Task | Dataset | Instances | Notes |
|---|---|---|---|
| `causal_n4` | `data/causal/n4.json` | 200 of 847 | `--gt-filter 8-200`: drops instances with `|H_O| = 1`, which are free points. |
| `3d` | `data/3d/g3_h3_b3.json` | 129 (all) | top view, grid 3, max height 3, max blocks 3 |
| `boolean_basic` | `data/boolean/basic_d2.json` | 35 (all) | depth 2, basic operators |
| `boolean_extended` | `data/boolean/extended_d2.json` | 78 (all) | depth 2, extended operators |
| `boolean_full` | `data/boolean/full_d2.json` | 80 (all) | depth 2, all operators |

Seed is `33550336` everywhere, so the same instances are drawn on every machine. The 200 causal
instance ids are committed in `data/splits/causal_n4_test.json`.

## 3. Holdout

The scored instance ids are committed (`data/splits/causal_n4_test.json` for causal), so a leak
is checkable: any model that saw a scored instance invalidates its row.

## 4. Comparability rules

- Same `configs/suites/main.yaml` or the row does not enter the leaderboard.
- Same serving configuration (quantisation, `max_tokens`, thinking mode); all of it is recorded
  in `meta.json`.
- One change per run: a run that changes method and prompt and budget proves nothing.

## 5. Reporting and the Boolean evaluator caveat

- Every run: `runs/<date>_<model>_<suite>[_<tag>]/meta.json` + one JSON per task.
- The leaderboard is generated, never typed: `make report`.
- Report the budget column next to RR, and parse rate next to VR/NR.
- Boolean's validator does not check hypothesis-space membership, so garbage can score VR=1 on
  the 7/35 basic instances whose observations are all zero. Report the official number and flag
  those rows.
- Local quantised models are an open-weights reference, not a reproduction of the paper.
