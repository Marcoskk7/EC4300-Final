"""Request-budget-matched causal pilot. No model training or oracle-guided actions."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import networkx as nx
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/HypoSpace/causal'))
from run_causal_benchmark import CausalBenchmarkEnhanced

ARMS = ('explore', 'repair', 'structure')


class Scheduler:
    def __init__(self, method, seed, c=1.0):
        self.method, self.rng, self.c = method, random.Random(seed), c
        self.counts = dict.fromkeys(ARMS, 0)
        self.rewards = dict.fromkeys(ARMS, 0)

    def choose(self, repair_available):
        active = [a for a in ARMS if a != 'repair' or repair_available]
        if self.method == 'baseline':
            return 'explore'
        if self.method == 'repair_only':
            return 'repair' if repair_available else 'explore'
        if self.method == 'random':
            return self.rng.choice(active)
        untried = [a for a in active if not self.counts[a]]
        if untried:
            return self.rng.choice(untried)
        total = sum(self.counts.values())
        scores = {a: self.rewards[a] / self.counts[a] + self.c * math.sqrt(
            2 * math.log(total + 1) / self.counts[a]) for a in active}
        best = max(scores.values())
        return self.rng.choice([a for a in active if scores[a] == best])

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward


def feedback(bench, graph, observations):
    """Only observations and public constraints; never accesses ground truth."""
    if graph is None:
        return ['Output could not be parsed as a permitted DAG. Use exactly Graph: A->B, B->C (or Graph: No edges), allowed nodes only, no cycles.']
    g = graph.to_networkx()
    problems = []
    for obs in observations:
        source = obs['perturbed_node']
        descendants = nx.descendants(g, source)
        for node, expected in obs['effects'].items():
            actual = int(node in descendants)
            if actual != expected:
                problems.append(f'Perturb({source}): {node} must be {expected}, but your graph predicts {actual}.')
    return problems


def make_prompt(bench, observations, history, arm, last_response, last_errors, rng):
    prompt = bench.create_prompt(observations, history)
    target = None
    if arm == 'repair':
        prompt += '\nRepair the previous candidate, while satisfying ALL observations and avoiding prior predictions.\nPrevious response:\n' + last_response[:2000]
        prompt += '\nValidation feedback:\n' + '\n'.join(last_errors)
    elif arm == 'structure':
        target = rng.randint(0, bench.max_edges)
        prompt += f'\nExploration direction: try a DAG with exactly {target} edges. If this is impossible, return another valid, non-repeated DAG. Keep the same output format.'
    return prompt, target


def evaluate(bench, observations, budget, method, seed, query, on_event=None):
    # Deliberately receives no ground truths. Scheduler resets for every instance.
    scheduler = Scheduler(method, seed)
    rng = random.Random(seed + 1)
    history, seen, valid_seen, events = [], set(), set(), []
    last_response, last_errors = '', []
    for step in range(budget):
        arm = scheduler.choose(bool(last_errors))
        prompt, target = make_prompt(bench, observations, history, arm, last_response, last_errors, rng)
        started = time.monotonic()
        result = query(prompt, step)  # one HTTP request; no hidden retry
        response = result['response']
        graph = bench.parse_llm_response(response)
        errors = feedback(bench, graph, observations)
        valid = graph is not None and not errors
        key = graph.get_hash() if graph else None
        duplicate = key in seen if key else False
        reward = int(valid and key not in valid_seen)
        if graph:
            history.append(graph)
            seen.add(key)
        if valid:
            valid_seen.add(key)
        scheduler.update(arm, reward)
        events.append(dict(step=step + 1, arm=arm, target_edges=target, reward=reward,
                           parsed=graph is not None, valid=valid, duplicate=duplicate,
                           graph=graph.to_dict() if graph else None, graph_hash=key,
                           cumulative_unique_valid=len(valid_seen), errors=errors,
                           prompt=prompt, response=response, usage=result.get('usage', {}),
                           finish_reason=result.get('finish_reason'), seconds=time.monotonic()-started))
        if on_event:
            on_event(events[-1])
        last_response, last_errors = response, errors
    return dict(method=method, n_requests=budget, events=events,
                parse=sum(e['parsed'] for e in events)/budget,
                VR=sum(e['valid'] for e in events)/budget, NR=len(seen)/budget,
                duplicate_rate=sum(e['duplicate'] for e in events)/budget,
                arm_counts=scheduler.counts, arm_rewards=scheduler.rewards)


def observation_key(instance):
    # Groups identical labeled observations regardless of their list order.
    return json.dumps(sorted((o['perturbed_node'], sorted(o['effects'].items()))
                             for o in instance['observations']), sort_keys=True)


def split_instances(bench):
    # Preserve the team's existing held-out IDs. Development uses other observations.
    test_ids = set(json.loads((ROOT/'data/splits/causal_n4_test.json').read_text())['ids'])
    test = [o for o in bench.filtered_observation_sets if o['observation_set_id'] in test_ids]
    forbidden = {observation_key(o) for o in test}
    dev = [o for o in bench.filtered_observation_sets
           if o['observation_set_id'] not in test_ids and observation_key(o) not in forbidden]
    # Stable ordering independent of input file order, not selected by outcomes.
    order = lambda o: hashlib.sha256(('adaptive-v1:'+o['observation_set_id']).encode()).hexdigest()
    return {'dev': sorted(dev, key=order), 'test': sorted(test, key=order)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='configs/models/qwen3.5-9b.yaml')
    p.add_argument('--base-url')
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--queries', type=int, help='fixed smoke budget; omit for 2*|H_O|')
    p.add_argument('--methods', nargs='+', choices=['baseline', 'random', 'ucb', 'repair_only'], default=['baseline','random','ucb'])
    p.add_argument('--seed', type=int, default=33550336)
    p.add_argument('--output', required=True, help='new output directory; never overwrites an experiment')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.limit < 1 or (args.queries is not None and args.queries < 1):
        p.error('limit and queries must be positive')
    model = yaml.safe_load((ROOT/args.model).read_text(encoding='utf-8'))
    if args.base_url:
        model['base_url'] = args.base_url
    bench = CausalBenchmarkEnhanced(str(ROOT/'data/causal/n4.json'), gt_filter=(8,200))
    # Public DAG upper bound rather than an inferred per-instance solution statistic.
    bench.max_edges = bench.metadata.get('max_edges') or len(bench.nodes)*(len(bench.nodes)-1)//2
    selected = split_instances(bench)[args.split][:args.limit]
    if not selected:
        p.error('empty split')
    manifest = dict(protocol='adaptive-causal-v1', split=args.split, model=model, seed=args.seed,
                    methods=args.methods, fixed_queries=args.queries, multiplier=2,
                    ids=[o['observation_set_id'] for o in selected],
                    notes='One request per round, no retry. Re-run baseline; not identical to historical retry protocol. Test IDs were already evaluated by the team. No isomorphism-disjoint generalization claim.')
    manifest['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    manifest['git_dirty'] = bool(subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip())
    manifest['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['dataset_sha256'] = hashlib.sha256((ROOT/'data/causal/n4.json').read_bytes()).hexdigest()
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    totals = []
    def query(prompt, step):
        payload = dict(model=model['served_model'], messages=[
            dict(role='system', content='You are an expert in causal inference and graph theory.'),
            dict(role='user', content=prompt)], temperature=model['temperature'],
            max_tokens=model['max_tokens'], seed=args.seed+step)
        payload.update(model.get('extra_body', {}))
        response = requests.post(model['base_url'].rstrip('/')+'/chat/completions', json=payload,
            headers={'Authorization': 'Bearer '+os.environ.get('HYPOSPACE_API_KEY','EMPTY')}, timeout=180)
        response.raise_for_status()  # Fail visibly on outages; do not treat as model mistakes.
        data = response.json()
        choice = data['choices'][0]
        return dict(response=choice['message'].get('content') or '', usage=data.get('usage',{}),
                    finish_reason=choice.get('finish_reason'))
    with (output/'instances.jsonl').open('w', encoding='utf-8') as records, (output/'events.jsonl').open('w', encoding='utf-8') as trace:
        for idx, instance in enumerate(selected):
            gt = instance['ground_truth_graphs']
            budget = args.queries or max(1, 2*len(gt))
            methods = list(args.methods)
            random.Random(args.seed+idx).shuffle(methods)
            for method in methods:
                def on_event(event):
                    trace.write(json.dumps(dict(id=instance['observation_set_id'], method=method, **event), ensure_ascii=False)+'\n')
                    trace.flush()
                result = evaluate(bench, instance['observations'], budget, method, args.seed+idx, query, on_event)
                # Oracle matching occurs only AFTER generation finishes.
                from modules.models import CausalGraph
                gt_keys = {CausalGraph.from_dict(g).get_hash() for g in gt}
                found = {e['graph_hash'] for e in result['events'] if e['valid']}
                result.update(id=instance['observation_set_id'], n_ground_truths=len(gt_keys),
                              RR=len(found & gt_keys)/len(gt_keys),
                              total_tokens=sum(e['usage'].get('total_tokens',0) for e in result['events']),
                              seconds=sum(e['seconds'] for e in result['events']))
                records.write(json.dumps(result, ensure_ascii=False)+'\n')
                records.flush()
                totals.append(result)
                print(f"{result['id']} {method}: VR={result['VR']:.3f} RR={result['RR']:.3f}", flush=True)
    summary = {m: {k: statistics.mean(r[k] for r in totals if r['method']==m)
                   for k in ['parse','VR','NR','RR','duplicate_rate','n_requests','total_tokens','seconds']}
               for m in args.methods}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
