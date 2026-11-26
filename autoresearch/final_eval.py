"""Final evaluation of the best policy found by the autoresearch loop. Run once, after the loop stops.

On the 9B node (via remote_run.sh, AR_SCRIPT=autoresearch/final_eval.py):
    runs both seeds on (a) every held-out test instance and (b) the dev half of 3D/Boolean,
    so that test + dev = the same instance sets as the formal report (causal: the 200 test ids).
Locally, afterwards:
    final_eval.py --report runs/autoresearch/<run>_final  -> final.md next to the run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

from harness import DEFAULT_MODEL, ROOT, SEEDS, TASKS  # noqa: E402  (harness sets up sys.path)

FORMAL = ROOT / 'results/adaptive_9b'
FORMAL_METHODS = ('baseline', 'repair_only', 'random', 'ucb')


def run_final(workers: int, model_path: str) -> None:
    import harness
    from harness import Adapter, CausalTask, MULTIPLIER, make_query, run_job, split
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone
    import hashlib
    import time

    harness.check_policy_source(harness.HERE / 'policy.py')
    from policy import Policy
    import yaml
    model = yaml.safe_load((harness.ROOT / model_path).read_text(encoding='utf-8'))
    adapters = {t: (CausalTask() if t == 'causal' else Adapter(t)) for t in TASKS}
    jobs = []
    for task, a in adapters.items():
        parts = split(task, a)
        for part in ('test',) if task == 'causal' else ('test', 'dev'):
            for seed in SEEDS:
                for idx, o in enumerate(parts[part]):
                    jobs.append((part, task, a, o, idx, seed, max(1, MULTIPLIER[task] * a.budget(o))))
    jobs.sort(key=lambda j: -j[-1])
    out = harness.RUNS / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_final')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with requests.Session() as session:
        session.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=workers))
        query = make_query(model, session)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(j[0], pool.submit(run_job, Policy, *j[1:], query)) for j in jobs]
            results = [dict(f.result(), part=part) for part, f in futures]
    with (out / 'jobs.jsonl').open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    meta = dict(model=model, model_config=model_path, workers=workers, git=harness.git_state(), seeds=list(SEEDS),
                policy_sha256=hashlib.sha256((harness.HERE / 'policy.py').read_bytes()).hexdigest(),
                seconds=time.monotonic() - started, transport_retries=len(harness.transport_errors),
                heldout=harness.summarize([r for r in results if r['part'] == 'test']),
                full=harness.summarize(results))
    (out / 'summary.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    (out / 'policy.py').write_bytes((harness.HERE / 'policy.py').read_bytes())
    print('---')
    for name in ('heldout', 'full'):
        print(f'score_{name}: {100 * statistics.mean(v["RR"] for v in meta[name].values()):.4f}')
        for task, v in meta[name].items():
            print(f'  {name}_{task}: RR {100 * v["RR"]:.2f}  VR {100 * v["VR"]:.2f}  NR {100 * v["NR"]:.2f}  '
                  f'dup {100 * v["duplicate_rate"]:.1f}  instances {v["instances"]}  requests {v["requests"]}')
    print(f'transport_retries:{meta["transport_retries"]}')
    print(f'seconds:          {meta["seconds"]:.1f}')
    print(f'run_dir:          {out.relative_to(ROOT)}')


def report(run_dir: Path) -> None:
    meta = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    pct = lambda x: f'{100 * x:.2f}'
    lines = ['# autoresearch 最终评测', '',
             f'policy sha256 `{meta["policy_sha256"][:12]}`，commit `{meta["git"]["commit"]}`，'
             f'seeds {meta["seeds"]}，耗时 {meta["seconds"] / 60:.0f} 分钟。', '',
             '- **全量**：与正式报告相同的实例集合（causal = 200 个 test id；3D/Boolean = 全部实例）。'
             '3D/Boolean 中有一半是循环中用来挑选策略的 dev 实例，所以这一列偏乐观。',
             '- **留出**：只有循环中从未用过的实例（causal 200 个，3D/Boolean 各一半），这是干净的数字。',
             '- 正式方法列取自 `results/adaptive_9b/*_comparison.json`（相同预算、相同 2 个 seed、全部实例）。', '',
             '| 任务 | baseline | repair_only | random | ucb | **autoresearch 全量** | autoresearch 留出 | 留出实例数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        formal = json.loads((FORMAL / f'{task}_comparison.json').read_text(encoding='utf-8'))['pooled']
        cells = [pct(formal[m]['RR']) for m in FORMAL_METHODS]
        lines.append(f'| {task} | ' + ' | '.join(cells) + f' | **{pct(meta["full"][task]["RR"])}** | '
                     f'{pct(meta["heldout"][task]["RR"])} | {meta["heldout"][task]["instances"]} |')
    lines += ['', '各列都是 RR（%），先在实例内平均两个 seed，再对实例取平均。']
    (run_dir / 'final.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--report', type=Path, help='build final.md from a finished *_final run dir (local)')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args()
    if args.report:
        report(args.report)
        return
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, 'PYTHONHASHSEED': '0'})
    if os.environ.get('AUTORESEARCH_FINAL') != '1':
        p.error('final evaluation touches the held-out split; set AUTORESEARCH_FINAL=1 (once, after the loop)')
    run_final(args.workers, args.model)


if __name__ == '__main__':
    main()
