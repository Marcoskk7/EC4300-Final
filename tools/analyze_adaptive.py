"""Paired descriptive development analysis; never treats requests as independent instances."""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics


def analyze(folder):
    folder = Path(folder)
    rows = [json.loads(line) for line in (folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    methods = manifest['methods']
    by_id = {}
    for row in rows:
        by_id.setdefault(row['id'], {})[row['method']] = row
    complete = {i: r for i, r in by_id.items() if set(r) == set(methods)}
    if len(complete) != len(manifest['ids']):
        raise ValueError('Incomplete experiment: do not compare unmatched methods')
    report = {'split':manifest['split'], 'instances':len(complete), 'seed':manifest['seed'],
              'note':'Development-only, one generation seed. Bootstrap is descriptive across instances, not proof of generalization.',
              'methods':{}, 'paired_RR':{}}
    for method in methods:
        group = [r[method] for r in complete.values()]
        events = [e for r in group for e in r['events']]
        assert all(r['n_requests'] == len(r['events']) for r in group)
        assert all(sum(e['reward'] for e in r['events']) == len({e['graph_hash'] for e in r['events'] if e['valid']}) for r in group)
        report['methods'][method] = {
            'macro':{k:statistics.mean(r[k] for r in group) for k in ['VR','NR','RR','parse','duplicate_rate']},
            'requests':len(events), 'total_tokens':sum(r['total_tokens'] for r in group),
            'seconds':sum(r['seconds'] for r in group),
            'new_valid':sum(e['reward'] for e in events),
            'truncated':sum(e['finish_reason']=='length' for e in events),
            'arms':{a:{'calls':sum(e['arm']==a for e in events),
                       'new_valid':sum(e['reward'] for e in events if e['arm']==a)}
                    for a in ['explore','repair','structure']}}
    for control in ['baseline','random']:
        diffs = [r['ucb']['RR']-r[control]['RR'] for r in complete.values()]
        rng = random.Random(5821)
        boot = sorted(statistics.mean(rng.choices(diffs,k=len(diffs))) for _ in range(10000))
        report['paired_RR']['ucb_minus_'+control] = dict(mean=statistics.mean(diffs),
            descriptive_bootstrap_95=[boot[249],boot[9749]],
            wins=sum(d>1e-12 for d in diffs),ties=sum(abs(d)<=1e-12 for d in diffs),losses=sum(d < -1e-12 for d in diffs))
    # Descriptive prefix analysis only: completed calls within each instance's
    # baseline token cost. Not a separately rerun token-budget experiment.
    report['baseline_token_cap_prefix_RR'] = {}
    for method in methods:
        rr = []
        for pair in complete.values():
            cap = pair['baseline']['total_tokens']
            spent, found = 0, set()
            for e in pair[method]['events']:
                spent += e['usage'].get('total_tokens', 0)
                if spent > cap:
                    break
                if e['valid']:
                    found.add(e['graph_hash'])
            rr.append(len(found)/pair[method]['n_ground_truths'])
        report['baseline_token_cap_prefix_RR'][method] = statistics.mean(rr)
    fields = ['id','method','n_ground_truths','n_requests','parse','VR','NR','RR','duplicate_rate','total_tokens','seconds']
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    (folder/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');analyze(p.parse_args().folder)
