"""Build the result figures embedded in the root README.

    .venv/bin/python tools/plot_latest_results.py
    .venv/bin/python tools/plot_latest_results.py --run runs/autoresearch/<dir>

Sources: repair_only comes from results/adaptive_9b/*_comparison.json; the composed
policy's per-task RR is the final-evaluation table in the root README (that run's raw
directory is kept off the repo); the 847 run is runs/autoresearch/*_causal_full
(jobs.jsonl + summary.json) and the zero-shot 847 is the causal_full baseline run.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'docs/figures'

SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, BEFORE, AFTER = '#e1e0d9', '#c3c2b7', '#5598e7', '#184f95'

# Root README, "最终评测" table: composed policy on the same instance sets as the
# formal report (causal = 200 test ids; 3D/Boolean = all instances, 2 seeds).
FINAL_RR = {'causal': 24.20, '3d': 44.10, 'boolean_basic': 43.43,
            'boolean_extended': 27.19, 'boolean_full': 18.26}
TASKS = [('causal', 'causal n=4 · 200', 200), ('3d', '3D · 129', 129),
         ('boolean_basic', 'boolean basic · 35', 35),
         ('boolean_extended', 'boolean extended · 78', 78),
         ('boolean_full', 'boolean full · 80', 80)]
BUCKETS = [('1', 1, 1), ('2–3', 2, 3), ('4–7', 4, 7), ('8–15', 8, 15),
           ('16–31', 16, 31), ('32–200', 32, 200)]
DEV_SCORES = [('起点', 19.59), ('#1', 22.02), ('#2', 24.39), ('#3', 26.45),
              ('#4', 27.69), ('#5', 30.22), ('#6', 33.29)]


def style() -> None:
    plt.rcParams.update({
        'font.family': ['PingFang SC', 'Arial', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'text.color': INK,
        'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK2,
        'axes.edgecolor': AXIS, 'grid.color': GRID,
        'figure.dpi': 200, 'savefig.dpi': 200,
    })


def finish(ax, title: str, out: Path) -> None:
    ax.set_title(title, loc='left', fontsize=12, color=INK, pad=14)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    ax.tick_params(axis='y', length=0)
    ax.figure.tight_layout()
    ax.figure.savefig(out)
    plt.close(ax.figure)


def dumbbell(labels: list[str], before: list[float], after: list[float],
             title: str, out: Path) -> None:
    """Before -> after per row: two shades of one hue, values labelled, delta in a column."""
    fig, ax = plt.subplots(figsize=(8.8, 0.72 * len(labels) + 1.9))
    ys = list(range(len(labels)))[::-1]
    right = max(after) + 20
    for y, b, a in zip(ys, before, after):
        ax.plot([b, a], [y, y], color=AXIS, linewidth=2.5, zorder=1,
                solid_capstyle='round')
        ax.scatter([b], [y], s=70, color=BEFORE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([a], [y], s=80, color=AFTER, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(b - 1.6, y, f'{b:.1f}', ha='right', va='center', fontsize=8.5, color=INK2)
        ax.text(a + 1.6, y, f'{a:.1f}', ha='left', va='center', fontsize=9, color=INK)
        ax.text(right, y, f'+{a - b:.1f}', ha='right', va='center', fontsize=9, color=INK)
    ax.set_yticks(ys, labels, fontsize=9.5)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(0, right)
    ax.set_xlabel('RR（%），右侧一列为提升（百分点）', fontsize=9)
    ax.scatter([], [], s=60, color=BEFORE, label='对照')
    ax.scatter([], [], s=70, color=AFTER, label='组合方法')
    ax.legend(loc='lower right', frameon=False, fontsize=9, ncol=2,
              handletextpad=0.3, columnspacing=1.1, bbox_to_anchor=(1, 1.0))
    finish(ax, title, out)


def causal_full_figure(run: Path, baseline: Path, out: Path) -> list[tuple[str, int, float, float]]:
    ours_by_id = defaultdict(list)
    for line in (run / 'jobs.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        ours_by_id[row['id']].append(row['RR'])
    ours = {k: statistics.mean(v) for k, v in ours_by_id.items()}
    base = {s['observation_set_id']: s['recovery_rate']
            for s in json.loads(baseline.read_text(encoding='utf-8'))['per_sample_results']}
    data = json.loads((ROOT / 'data/causal/n4.json').read_text(encoding='utf-8'))
    size = {o['observation_set_id']: len(o['ground_truth_graphs'])
            for docs in data['datasets_by_n_observations'].values() for o in docs}

    rows = []
    for label, lo, hi in BUCKETS:
        ids = [i for i, n in size.items() if lo <= n <= hi]
        rows.append((f'{label}（{len(ids)}）', len(ids),
                     100 * statistics.mean(base[i] for i in ids),
                     100 * statistics.mean(ours[i] for i in ids)))
    dumbbell([r[0] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
             'causal 全量 847：按解集大小分层（RR，%）', out)
    return rows


def dev_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    x = list(range(len(DEV_SCORES)))
    y = [s for _, s in DEV_SCORES]
    ax.axhline(y[0], color=GRID, linewidth=1)
    ax.plot(x, y, color=AFTER, linewidth=2, marker='o', markersize=6,
            markerfacecolor=SURFACE, markeredgewidth=2, zorder=3)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.55, f'{yi:.2f}', ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax.set_xticks(x, [t for t, _ in DEV_SCORES])
    ax.set_ylim(18, 35.5)
    ax.set_xlabel('通过 keep 门槛的改动', fontsize=9)
    ax.set_ylabel('dev 分数（%）', fontsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(AXIS)
    ax.spines['bottom'].set_color(AXIS)
    ax.set_title(f'autoresearch dev 分数：6 次保留改动累计 +{y[-1] - y[0]:.2f}',
                 loc='left', fontsize=12, color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, help='847 causal run dir (default: latest non-smoke)')
    p.add_argument('--baseline', type=Path,
                   default=ROOT / 'runs/20261123_qwen3.5-9b_causal_full/causal_n4_all.json')
    args = p.parse_args()
    candidates = sorted(d for d in (ROOT / 'runs/autoresearch').glob('*_causal_full')
                        if (d / 'summary.json').exists() and 'smoke' not in d.name)
    run = args.run or (candidates[-1] if candidates else None)
    if run is None or not (run / 'jobs.jsonl').exists():
        raise SystemExit('no causal_full run with jobs.jsonl found; pass --run')
    if not args.baseline.exists():
        raise SystemExit(f'missing 847 zero-shot baseline: {args.baseline}')

    style()
    FIG.mkdir(parents=True, exist_ok=True)

    repair = {}
    for key, _, _ in TASKS:
        comp = json.loads((ROOT / f'results/adaptive_9b/{key}_comparison.json').read_text(encoding='utf-8'))
        repair[key] = 100 * comp['pooled']['repair_only']['RR']
    dumbbell([label for _, label, _ in TASKS],
             [repair[k] for k, _, _ in TASKS],
             [FINAL_RR[k] for k, _, _ in TASKS],
             '五个任务：组合方法 vs 固定修复 repair_only（RR，%）',
             FIG / 'rr_by_task.png')
    rows = causal_full_figure(run, args.baseline, FIG / 'causal_847_by_size.png')
    dev_figure(FIG / 'dev_score_trajectory.png')

    print(f'figures -> {FIG.relative_to(ROOT)}')
    for (key, _, _) in TASKS:
        print(f'  {key:<17} repair_only {repair[key]:5.2f} -> {FINAL_RR[key]:5.2f} '
              f'(+{FINAL_RR[key] - repair[key]:.2f})')
    for label, n, b, a in rows:
        print(f'  |H_O| {label:<12} n={n:3d} zero-shot {b:5.1f} -> {a:5.1f} (+{a - b:.1f})')


if __name__ == '__main__':
    main()
