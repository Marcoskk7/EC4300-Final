"""Evaluate the current autoresearch policy on all 847 n=4 causal instances.

`harness.CausalTask` keeps the gt_filter (8, 200) of the loop, so its causal "full" column
is only the 200 hard test ids. This script drops the filter and runs every instance in
`data/causal/n4.json` (175 of them have |H_O| = 1) with the same policy, 2x budget and
seeds, which is the instance set of `configs/suites/causal_full.yaml`.

On the 9B node, where vLLM is on localhost:
    AR_CAUSAL_FULL=1 autoresearch/remote_run.sh > causal_full.log 2>&1
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402  (adds tools/ to sys.path)
from adaptive_causal import CausalBenchmarkEnhanced  # noqa: E402
from harness import DEFAULT_MODEL, MULTIPLIER, ROOT, SEEDS, make_query, run_job, summarize  # noqa: E402
from policy import Policy  # noqa: E402


class CausalFullTask(harness.CausalTask):
    """All 847 observation sets, no |H_O| filter; parsing and feedback as in the loop."""

    def __init__(self) -> None:
        self.bench = CausalBenchmarkEnhanced(str(ROOT / 'data/causal/n4.json'), gt_filter=None)
        self.bench.max_edges = 6
        self.system = 'You are an expert in causal inference and graph theory.'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--base-url')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--smoke', action='store_true', help='2 instances x 3 requests, 1 seed; not a score')
    args = p.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, 'PYTHONHASHSEED': '0'})
    if not 1 <= args.workers <= 32:
        p.error('workers must be 1..32 (server max-num-seqs)')
    harness.check_policy_source(harness.HERE / 'policy.py')

    model = yaml.safe_load((ROOT / args.model).read_text(encoding='utf-8'))
    if args.base_url:
        model['base_url'] = args.base_url
    adapter = CausalFullTask()
    instances = adapter.bench.filtered_observation_sets[:2 if args.smoke else None]
    seeds = SEEDS[:1] if args.smoke else SEEDS
    test_ids = set(json.loads((ROOT / 'data/splits/causal_n4_test.json').read_text(encoding='utf-8'))['ids'])
    jobs = [(adapter, o, i, s, 3 if args.smoke else max(1, MULTIPLIER['causal'] * adapter.budget(o)))
            for s in seeds for i, o in enumerate(instances)]
    jobs.sort(key=lambda j: -j[-1])

    out = harness.RUNS / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                          + ('_causal_full_smoke' if args.smoke else '_causal_full'))
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with requests.Session() as session:
        session.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=args.workers))
        query = make_query(model, session)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, Policy, 'causal', a, o, i, s, b, query)
                       for a, o, i, s, b in jobs]
            results = [f.result() for f in futures]
    seconds = time.monotonic() - started

    with (out / 'jobs.jsonl').open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    full = summarize(results)['causal']
    subset_rows = [r for r in results if r['id'] in test_ids]
    subset = summarize(subset_rows).get('causal') if subset_rows else None
    meta = dict(model=model, model_config=args.model, workers=args.workers, smoke=args.smoke,
                git=harness.git_state(), seeds=list(seeds),
                policy_sha256=hashlib.sha256((harness.HERE / 'policy.py').read_bytes()).hexdigest(),
                seconds=seconds, transport_retries=len(harness.transport_errors),
                causal_full=full, causal_test200=subset)
    (out / 'summary.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    (out / 'policy.py').write_bytes((harness.HERE / 'policy.py').read_bytes())

    print('---')
    for name, v in (('causal_full', full), ('causal_test200', subset)):
        if not v:
            continue
        print(f'rr_{name}: {100 * v["RR"]:.4f}  vr {100 * v["VR"]:.2f}  nr {100 * v["NR"]:.2f}  '
              f'parse {100 * v["parse"]:.1f}  dup {100 * v["duplicate_rate"]:.1f}  '
              f'instances {v["instances"]}  requests {v["requests"]}')
    print(f'tokens:            {full["tokens"]}')
    print(f'context_rejected:  {full["context_rejected"]}')
    print(f'truncated:         {full["truncated"]}')
    print(f'transport_retries: {len(harness.transport_errors)}')
    print(f'seconds:           {seconds:.1f}')
    print(f'run_dir:           {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
