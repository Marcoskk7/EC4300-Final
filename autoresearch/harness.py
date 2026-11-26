"""Locked autoresearch harness: splits, budgets, LLM calls, validity checks, scoring.

The agent edits only autoresearch/policy.py (see autoresearch/program.md).
This file and everything it imports are read-only during the loop.

    PYTHONHASHSEED=0 .venv/bin/python autoresearch/harness.py > run.log 2>&1
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import requests
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(HERE))

from adaptive_causal import CausalBenchmarkEnhanced, feedback, split_instances  # noqa: E402
from cross_domain import Adapter  # noqa: E402
from modules.models import CausalGraph  # noqa: E402

TASKS = ('causal', '3d', 'boolean_basic', 'boolean_extended', 'boolean_full')
SEEDS = (33550336, 33550337)
MULTIPLIER = {'causal': 2, '3d': 1, 'boolean_basic': 1, 'boolean_extended': 1, 'boolean_full': 1}
CAUSAL_DEV_INSTANCES = 32
CAUSAL_DEV_MAX_GT = 24  # keeps one dev run within the time budget
SPLIT_SALT = 'autoresearch-v1:'
DEFAULT_MODEL = 'configs/models/qwen3.5-9b.yaml'
RUNS = ROOT / 'runs/autoresearch'
ALLOWED_IMPORTS = {'math', 'random', 're', 'itertools', 'collections', 'dataclasses',
                   'typing', 'statistics', 'functools', '__future__'}
FORBIDDEN_NAMES = {'open', 'eval', 'exec', 'compile', '__import__', 'globals', 'vars', 'breakpoint'}


@dataclass(frozen=True)
class TaskInfo:
    """Everything the policy may know about one instance. No ground truth, no budget."""
    domain: str
    observations: Any
    public: dict
    system: str
    base_prompt: Callable[[list], str]


@dataclass(frozen=True)
class Outcome:
    step: int
    response: str
    finish_reason: str | None
    candidate: Any
    key: Any
    valid: bool
    duplicate: bool
    errors: tuple


class CausalTask:
    """Same interface as cross_domain.Adapter for the n=4 causal task."""

    def __init__(self) -> None:
        self.bench = CausalBenchmarkEnhanced(str(ROOT / 'data/causal/n4.json'), gt_filter=(8, 200))
        self.bench.max_edges = 6
        self.system = 'You are an expert in causal inference and graph theory.'

    def iid(self, o: dict) -> str:
        return o['observation_set_id']

    def observations(self, o: dict) -> list:
        return o['observations']

    def truth(self, o: dict) -> set:
        return {CausalGraph.from_dict(g).get_hash() for g in o['ground_truth_graphs']}

    def budget(self, o: dict) -> int:
        return len(o['ground_truth_graphs'])

    def parse(self, response: str):
        return self.bench.parse_llm_response(response)

    def inspect(self, candidate, obs: list):
        return (candidate.get_hash() if candidate else None), feedback(self.bench, candidate, obs)

    def public(self) -> dict:
        return {'nodes': list(self.bench.nodes), 'max_edges': self.bench.max_edges}


def cross_public(a: Adapter) -> dict:
    if a.is3d:
        m = a.bench.metadata
        return {'grid_size': m.get('grid_size', 3), 'max_height': m['max_height']}
    return {'variables': list(a.bench.variables), 'operators': sorted(a.bench.operators),
            'max_depth': a.bench.max_depth}


def split(task: str, adapter) -> dict:
    if task == 'causal':
        parts = split_instances(adapter.bench)
        dev = [o for o in parts['dev'] if len(o['ground_truth_graphs']) <= CAUSAL_DEV_MAX_GT]
        return {'dev': dev[:CAUSAL_DEV_INSTANCES], 'test': parts['test']}
    ordered = sorted(adapter.instances,
                     key=lambda o: hashlib.sha256((SPLIT_SALT + adapter.iid(o)).encode()).hexdigest())
    half = len(ordered) // 2
    return {'dev': ordered[:half], 'test': ordered[half:]}


def check_policy_source(path: Path) -> None:
    """Reject policies that could reach data, network or the harness internals."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split('.')[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or '').split('.')[0]]
        else:
            mods = []
        bad = [m for m in mods if m not in ALLOWED_IMPORTS]
        if bad:
            raise ValueError(f'policy.py imports forbidden module(s): {bad}')
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f'policy.py uses forbidden name: {node.id}')
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            raise ValueError(f'policy.py accesses dunder attribute: {node.attr}')


def validate_messages(messages) -> list:
    if not isinstance(messages, list) or not messages:
        raise ValueError('policy must return a non-empty list of messages')
    for m in messages:
        if set(m) != {'role', 'content'} or m['role'] not in ('system', 'user', 'assistant') \
                or not isinstance(m['content'], str):
            raise ValueError(f'bad message: {str(m)[:200]}')
    return [dict(m) for m in messages]


TRANSPORT_RETRIES = 5
transport_errors: list = []


def post_with_retry(session: requests.Session, url: str, payload: dict) -> requests.Response:
    """Retry only transport failures (no response reached the policy, so no extra information)."""
    for attempt in range(TRANSPORT_RETRIES + 1):
        try:
            r = session.post(url, json=payload, timeout=600)
            if r.status_code < 500:
                return r
            err = f'HTTP {r.status_code}'
        except (requests.ConnectionError, requests.Timeout) as e:
            err = repr(e)[:200]
        transport_errors.append(err)
        if attempt == TRANSPORT_RETRIES:
            raise RuntimeError(f'endpoint failed {TRANSPORT_RETRIES + 1} times: {err}')
        time.sleep(2 ** attempt)


def make_query(model: dict, session: requests.Session) -> Callable[[list, int], dict]:
    def query(messages: list, seed: int) -> dict:
        payload = dict(model=model['served_model'], messages=messages, temperature=model['temperature'],
                       max_tokens=model['max_tokens'], seed=seed)
        payload.update(model.get('extra_body', {}))
        r = post_with_retry(session, model['base_url'].rstrip('/') + '/chat/completions', payload)
        if r.status_code == 400 and ('context' in r.text.lower() or 'max_model_len' in r.text.lower()):
            return dict(response='', usage={}, finish_reason='context_rejected')  # consumes the request
        r.raise_for_status()
        data = r.json()
        choice = data['choices'][0]
        return dict(response=choice['message'].get('content') or '', usage=data.get('usage', {}),
                    finish_reason=choice.get('finish_reason'))
    return query


def task_info(task: str, adapter, obs) -> TaskInfo:
    def base_prompt(history: list) -> str:
        return adapter.bench.create_prompt(obs, history)
    public = adapter.public() if task == 'causal' else cross_public(adapter)
    return TaskInfo(task, obs, public, adapter.system, base_prompt)


def run_job(policy_cls, task: str, adapter, o: dict, idx: int, seed: int, budget: int, query) -> dict:
    """One instance x one seed. Exactly `budget` requests, one candidate per request."""
    obs = adapter.observations(o)
    policy = policy_cls(task_info(task, adapter, obs), random.Random(seed + idx + 1))
    seen, valid_seen, events = set(), set(), []
    for step in range(budget):
        messages = validate_messages(policy.next_messages())
        result = query(messages, seed + step)
        candidate = adapter.parse(result['response'])
        key, errors = adapter.inspect(candidate, obs)
        valid = key is not None and not errors
        duplicate = key in seen if key is not None else False
        if key is not None:
            seen.add(key)
        if valid:
            valid_seen.add(key)
        events.append(dict(step=step + 1, valid=valid, duplicate=duplicate, parsed=candidate is not None,
                           key=key, errors=errors, finish_reason=result['finish_reason'],
                           tokens=result['usage'].get('total_tokens', 0), response=result['response'],
                           prompt_chars=sum(len(m['content']) for m in messages)))
        policy.observe(Outcome(step + 1, result['response'], result['finish_reason'], candidate, key,
                               valid, duplicate, tuple(errors)))
    truth = adapter.truth(o)  # scoring only after generation
    return dict(task=task, id=adapter.iid(o), seed=seed, n_requests=budget,
                RR=len(set(truth) & valid_seen) / len(set(truth)),
                VR=sum(e['valid'] for e in events) / budget, NR=len(seen) / budget,
                parse=sum(e['parsed'] for e in events) / budget,
                duplicate_rate=sum(e['duplicate'] for e in events) / budget,
                tokens=sum(e['tokens'] for e in events),
                context_rejected=sum(e['finish_reason'] == 'context_rejected' for e in events),
                truncated=sum(e['finish_reason'] == 'length' for e in events), events=events)


def summarize(results: list) -> dict:
    out = {}
    for task in TASKS:
        rows = [r for r in results if r['task'] == task]
        if not rows:
            continue
        ids = sorted({r['id'] for r in rows})
        per = lambda k: statistics.mean(statistics.mean(r[k] for r in rows if r['id'] == i) for i in ids)
        out[task] = {k: per(k) for k in ('RR', 'VR', 'NR', 'parse', 'duplicate_rate')}
        out[task].update(instances=len(ids), requests=sum(r['n_requests'] for r in rows),
                         tokens=sum(r['tokens'] for r in rows),
                         context_rejected=sum(r['context_rejected'] for r in rows),
                         truncated=sum(r['truncated'] for r in rows))
    return out


def git_state() -> dict:
    run = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'commit': run('rev-parse', '--short', 'HEAD'), 'dirty': bool(run('status', '--porcelain'))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--base-url', help='override the model config endpoint')
    p.add_argument('--model', default=DEFAULT_MODEL, help='model config (endpoint, sampling) to evaluate')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--smoke', action='store_true', help='2 instances x 3 requests per task, 1 seed; not a score')
    args = p.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, 'PYTHONHASHSEED': '0'})
    if args.split == 'test' and os.environ.get('AUTORESEARCH_FINAL') != '1':
        p.error('test split is for the final human-run evaluation only (set AUTORESEARCH_FINAL=1)')
    if not 1 <= args.workers <= 32:
        p.error('workers must be 1..32 (server max-num-seqs)')
    check_policy_source(HERE / 'policy.py')
    from policy import Policy

    model_config = ROOT / args.model
    model = yaml.safe_load(model_config.read_text(encoding='utf-8'))
    if args.base_url:
        model['base_url'] = args.base_url
    adapters = {t: (CausalTask() if t == 'causal' else Adapter(t)) for t in TASKS}
    seeds = SEEDS[:1] if args.smoke else SEEDS
    jobs = []
    for task, a in adapters.items():
        selected = split(task, a)[args.split][:2 if args.smoke else None]
        for seed in seeds:
            for idx, o in enumerate(selected):
                budget = 3 if args.smoke else max(1, MULTIPLIER[task] * a.budget(o))
                jobs.append((task, a, o, idx, seed, budget))
    jobs.sort(key=lambda j: -j[-1])  # longest first so the pool drains evenly

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = RUNS / f'{stamp}_{args.split}{"_smoke" if args.smoke else ""}'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with requests.Session() as session:
        adapter_pool = requests.adapters.HTTPAdapter(pool_maxsize=args.workers)
        session.mount('http://', adapter_pool)
        query = make_query(model, session)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, Policy, t, a, o, i, s, b, query) for t, a, o, i, s, b in jobs]
            results = [f.result() for f in futures]
    seconds = time.monotonic() - started

    with (out / 'jobs.jsonl').open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    summary = summarize(results)
    score = 100 * statistics.mean(v['RR'] for v in summary.values())
    meta = dict(split=args.split, smoke=args.smoke, seeds=list(seeds), model=model,
                model_config=args.model, workers=args.workers, git=git_state(),
                policy_sha256=hashlib.sha256((HERE / 'policy.py').read_bytes()).hexdigest(),
                seconds=seconds, score=score, transport_retries=len(transport_errors), tasks=summary)
    (out / 'summary.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    (out / 'policy.py').write_bytes((HERE / 'policy.py').read_bytes())

    print('---')
    print(f'score:            {score:.4f}')
    for task, v in summary.items():
        print(f'rr_{task + ":":<18}{100 * v["RR"]:.4f}   vr {100 * v["VR"]:.2f}  nr {100 * v["NR"]:.2f}  '
              f'parse {100 * v["parse"]:.1f}  dup {100 * v["duplicate_rate"]:.1f}')
    print(f'requests:         {sum(v["requests"] for v in summary.values())}')
    print(f'tokens:           {sum(v["tokens"] for v in summary.values())}')
    print(f'context_rejected: {sum(v["context_rejected"] for v in summary.values())}')
    print(f'truncated:        {sum(v["truncated"] for v in summary.values())}')
    print(f'transport_retries:{len(transport_errors)}')
    print(f'seconds:          {seconds:.1f}')
    print(f'run_dir:          {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
