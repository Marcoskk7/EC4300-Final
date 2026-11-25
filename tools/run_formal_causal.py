"""Frozen causal evaluation with per-request journals and independent job workers.

Does not change adaptive_causal's prompts, rewards, scheduling or sampling.
Use --smoke only on development instances before starting the locked test run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

import requests
import yaml

from adaptive_causal import ROOT, CausalBenchmarkEnhanced, evaluate, split_instances
from modules.models import CausalGraph

METHODS = ['baseline', 'random', 'ucb', 'repair_only']
SEEDS = [33550336, 33550337]


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def read_journal(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    # A killed write may leave one incomplete trailing line; trim only that tail.
    end = data.rfind(b'\n')+1
    if end != len(data):
        with path.open('r+b') as f:
            f.truncate(end)
    return [json.loads(line) for line in data[:end].splitlines()]


def run_job(bench, instance, idx, method, seed, budget, model, folder):
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder/'result.json'
    if result_path.exists():
        r = json.loads(result_path.read_text(encoding='utf-8'))
        return {k:v for k,v in r.items() if k != 'events'}
    events = read_journal(folder/'events.jsonl')
    attempts = read_journal(folder/'attempts.jsonl')
    if len(events) > budget or any(e['step'] != i+1 for i,e in enumerate(events)):
        raise ValueError('Corrupt event journal')
    if len(attempts) not in (len(events), len(events)+1):
        raise ValueError('Attempt/event counts disagree')
    pending = len(attempts) > len(events)
    with (folder/'events.jsonl').open('a', encoding='utf-8') as trace, (folder/'attempts.jsonl').open('a', encoding='utf-8') as audit:
        session = requests.Session()
        def query(prompt, step):
            if step < len(events):
                e = events[step]
                if e['prompt'] != prompt:
                    raise ValueError('Resume prompt differs from frozen run')
                return dict(response=e['response'], usage=e['usage'], finish_reason=e['finish_reason'])
            if step == len(events) and pending:
                # An interrupted request may have reached the server: consume it,
                # never silently give the resumed method a free additional attempt.
                return dict(response='', usage={}, finish_reason='interrupted_unknown')
            payload = dict(model=model['served_model'], messages=[
                dict(role='system',content='You are an expert in causal inference and graph theory.'),
                dict(role='user',content=prompt)], temperature=model['temperature'],
                max_tokens=model['max_tokens'], seed=seed+step)
            payload.update(model.get('extra_body',{}))
            audit.write(json.dumps(dict(step=step+1, time=datetime.now(timezone.utc).isoformat(),
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))+'\n')
            audit.flush()
            os.fsync(audit.fileno())
            response = session.post(model['base_url'].rstrip('/')+'/chat/completions', json=payload,
                headers={'Authorization':'Bearer '+os.environ.get('HYPOSPACE_API_KEY','EMPTY')}, timeout=300)
            # All methods retain the same 8192-context service limit. An explicit
            # context rejection consumes a request; do not shorten history or retry.
            if response.status_code == 400 and ('context' in response.text.lower() or 'max_model_len' in response.text.lower()):
                return dict(response='',usage={},finish_reason='context_rejected')
            response.raise_for_status()
            data=response.json(); choice=data['choices'][0]
            return dict(response=choice['message'].get('content') or '',usage=data.get('usage',{}),
                        finish_reason=choice.get('finish_reason'))
        def on_event(event):
            step = event['step']-1
            if step < len(events):
                old = events[step]
                for key in ['arm','target_edges','reward','graph_hash','valid','duplicate']:
                    if event[key] != old[key]:
                        raise ValueError(f'Resume replay differs: {key}')
                event.update(old)
                return
            if event['finish_reason']=='interrupted_unknown':
                event['seconds']=0.0
            trace.write(json.dumps(event,ensure_ascii=False)+'\n')
            trace.flush()
            os.fsync(trace.fileno())
        try:
            result=evaluate(bench,instance['observations'],budget,method,seed+idx,query,on_event)
        finally:
            session.close()
    gt_keys={CausalGraph.from_dict(g).get_hash() for g in instance['ground_truth_graphs']}
    found={e['graph_hash'] for e in result['events'] if e['valid']}
    result.update(id=instance['observation_set_id'],seed=seed,n_ground_truths=len(gt_keys),
        RR=len(found & gt_keys)/len(gt_keys),total_tokens=sum(e['usage'].get('total_tokens',0) for e in result['events']),
        seconds=sum(e['seconds'] for e in result['events']),
        interrupted_unknown=sum(e['finish_reason']=='interrupted_unknown' for e in result['events']),
        context_rejected=sum(e['finish_reason']=='context_rejected' for e in result['events']))
    atomic_json(result_path,result)
    return {k:v for k,v in result.items() if k != 'events'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8001/v1')
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if not 1 <= args.workers <= 32:
        p.error('Use 1..32 workers, matching the available server capacity')
    bench=CausalBenchmarkEnhanced(str(ROOT/'data/causal/n4.json'),gt_filter=(8,200))
    bench.max_edges=6
    split='dev' if args.smoke else 'test'
    selected=split_instances(bench)[split][:2 if args.smoke else 200]
    if not args.smoke and len(selected)!=200:
        raise ValueError('Formal run requires all 200 predefined test IDs')
    model=yaml.safe_load((ROOT/'configs/models/qwen3.5-9b.yaml').read_text())
    model['base_url']=args.base_url
    frozen_paths=['tools/adaptive_causal.py','tools/run_formal_causal.py',
                  'configs/models/qwen3.5-9b.yaml',
                  'data/causal/n4.json','data/splits/causal_n4_test.json']
    frozen_paths += sorted(str(f.relative_to(ROOT)).replace('\\','/')
                           for f in (ROOT/'third_party/HypoSpace/causal').rglob('*.py'))
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in frozen_paths}
    manifest=dict(protocol='adaptive-causal-v1',split=split,model=model,seeds=SEEDS,methods=METHODS,
        fixed_queries=4 if args.smoke else None,multiplier=2,workers=args.workers,
        ids=[o['observation_set_id'] for o in selected],source_hashes=hashes,
        dataset_sha256=hashes['data/causal/n4.json'],
        context_policy='No prompt truncation. Explicit context-limit rejection consumes request.',
        recovery_policy='Replay saved events. Unknown interrupted requests consume budget and are disclosed.',
        notes='Test IDs were previously evaluated by the team; no label-isomorphism-disjoint generalization claim. Concurrent jobs may affect numerical reproducibility. No model/service changes.')
    jobs=[]
    for seed in SEEDS:
        for idx,instance in enumerate(selected):
            methods=list(METHODS);random.Random(seed+idx).shuffle(methods)
            for method in methods:
                budget=4 if args.smoke else 2*len(instance['ground_truth_graphs'])
                jobs.append((instance,idx,method,seed,budget))
    manifest['total_requests']=sum(j[-1] for j in jobs)
    if args.dry_run:
        print(json.dumps(manifest,indent=2));return
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    lock=output/'run.lock'
    if lock.exists():
        raise RuntimeError('Run lock exists. Check recorded PID before manually removing a stale lock.')
    lock.write_text(str(os.getpid()))
    try:
        mp=output/'manifest.json'
        if mp.exists() and json.loads(mp.read_text())!=manifest:
            raise ValueError('Frozen manifest mismatch; cannot resume different settings')
        atomic_json(mp,manifest)
        status=dict(state='running',pid=os.getpid(),total_jobs=len(jobs),total_requests=manifest['total_requests'],
                    completed_jobs=0,completed_requests=0,started=datetime.now(timezone.utc).isoformat())
        atomic_json(output/'status.json',status)
        health=requests.get(args.base_url.rstrip('/')+'/models',timeout=10);health.raise_for_status()
        if model['served_model'] not in [m['id'] for m in health.json()['data']]:
            raise RuntimeError('Expected model not served')
        atomic_json(output/'service_snapshot.json',health.json())
        completed=[]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures={executor.submit(run_job,bench,o,idx,m,s,n,model,output/'jobs'/str(s)/o['observation_set_id']/m):(s,o['observation_set_id'],m)
                     for o,idx,m,s,n in jobs}
            for future in as_completed(futures):
                try:
                    row=future.result()
                except Exception:
                    for f in futures:
                        f.cancel()
                    raise
                completed.append(row)
                status.update(completed_jobs=len(completed),completed_requests=sum(r['n_requests'] for r in completed),
                              updated=datetime.now(timezone.utc).isoformat())
                atomic_json(output/'status.json',status)
                print(f"completed {len(completed)}/{len(jobs)} jobs",flush=True)
        for seed in SEEDS:
            dest=output/f'seed_{seed}';dest.mkdir(exist_ok=True)
            sm=dict(manifest,seed=seed);atomic_json(dest/'manifest.json',sm)
            with (dest/'instances.jsonl').open('w',encoding='utf-8') as out:
                for o in selected:
                    for method in METHODS:
                        r=json.loads((output/'jobs'/str(seed)/o['observation_set_id']/method/'result.json').read_text())
                        out.write(json.dumps(r,ensure_ascii=False)+'\n')
        status.update(state='complete',finished=datetime.now(timezone.utc).isoformat(),
            interrupted_unknown=sum(r['interrupted_unknown'] for r in completed),
            context_rejected=sum(r['context_rejected'] for r in completed))
        atomic_json(output/'status.json',status)
    except Exception as e:
        atomic_json(output/'failure.json',dict(error=repr(e),time=datetime.now(timezone.utc).isoformat()))
        if 'status' in locals():
            status.update(state='failed',error=repr(e));atomic_json(output/'status.json',status)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
