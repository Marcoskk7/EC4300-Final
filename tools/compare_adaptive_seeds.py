"""Combine complete matched runs; bootstrap instances after averaging seeds."""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics as stats

METHODS = ('baseline', 'random', 'ucb', 'repair_only')


def compare(folders, output):
    records, manifests = {}, []
    for folder in map(Path, folders):
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        manifests.append(manifest)
        rows = [json.loads(line) for line in (folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
        actual = {(r['id'], r['method']) for r in rows}
        expected = {(i, m) for i in manifest['ids'] for m in manifest['methods']}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(f'Incomplete or duplicated run: {folder}')
        for row in rows:
            key = (manifest['seed'], row['id'], row['method'])
            if key in records:
                raise ValueError(f'Duplicated seed/instance/method: {key}')
            assert row['n_requests'] == len(row['events'])
            records[key] = row
    first = manifests[0]
    for m in manifests[1:]:
        for key in ['protocol', 'split', 'ids', 'fixed_queries', 'multiplier', 'dataset_sha256']:
            if m[key] != first[key]:
                raise ValueError(f'Mismatched protocol field {key}')
        for key in ['served_model', 'temperature', 'max_tokens', 'extra_body']:
            if m['model'][key] != first['model'][key]:
                raise ValueError(f'Mismatched model setting {key}')
    seeds = sorted({m['seed'] for m in manifests})
    ids = first['ids']
    expected = {(s, i, m) for s in seeds for i in ids for m in METHODS}
    if set(records) != expected:
        raise ValueError('All seeds must contain all instances and all four methods')
    for seed in seeds:
        for iid in ids:
            assert len({records[seed,iid,m]['n_requests'] for m in METHODS}) == 1
    report = dict(seeds=seeds, n_instances=len(ids), split=first['split'],
                  limitation=f"{first['split']} split, {len(seeds)} seeds on {len(ids)} shared instances; bootstrap clusters by instance, not individual requests. Runs may differ in wall-clock serving conditions.",
                  per_seed={}, pooled={}, paired_RR={})
    for seed in seeds:
        report['per_seed'][str(seed)] = {m: {k:stats.mean(records[seed,i,m][k] for i in ids)
            for k in ['VR','NR','RR','duplicate_rate','parse']} for m in METHODS}
    for method in METHODS:
        rows = [records[s,i,method] for s in seeds for i in ids]
        report['pooled'][method] = dict(
            **{k:stats.mean(r[k] for r in rows) for k in ['VR','NR','RR','duplicate_rate','parse']},
            requests=sum(r['n_requests'] for r in rows),
            tokens=sum(r['total_tokens'] for r in rows),
            seconds=sum(r['seconds'] for r in rows),
            new_valid=sum(e['reward'] for r in rows for e in r['events']))
    for method, control in [('ucb','baseline'),('ucb','random'),('ucb','repair_only'),('repair_only','baseline')]:
        diffs = [stats.mean(records[s,i,method]['RR']-records[s,i,control]['RR'] for s in seeds) for i in ids]
        rng = random.Random(5821)
        boot = sorted(stats.mean(rng.choices(diffs,k=len(ids))) for _ in range(10000))
        report['paired_RR'][method+'_minus_'+control] = dict(
            mean=stats.mean(diffs), descriptive_instance_bootstrap_95=[boot[249],boot[9749]],
            wins=sum(d>1e-12 for d in diffs), ties=sum(abs(d)<=1e-12 for d in diffs), losses=sum(d < -1e-12 for d in diffs))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'per_instance.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['seed','id','method','VR','NR','RR','n_requests','total_tokens','seconds']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
        for (seed,iid,method), row in sorted(records.items()):
            writer.writerow(dict(row,seed=seed))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folders',nargs='+'); p.add_argument('--output',required=True)
    args=p.parse_args(); compare(args.folders,args.output)
