"""Frozen cross-domain evaluation, one request per step, resumable journals."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import requests
import yaml
from adaptive_causal import ROOT,Scheduler
from run_formal_causal import atomic_json,read_journal,METHODS,SEEDS
from cross_domain import Adapter,DATASETS


def job(a,o,idx,method,seed,budget,model,folder):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        return json.loads((folder/'result.json').read_text())
    saved=read_journal(folder/'events.jsonl');attempts=read_journal(folder/'attempts.jsonl')
    if len(attempts)!=len(saved):
        raise RuntimeError(f'Unresolved request in {folder}; explicit audit required, no automatic retry')
    s=Scheduler(method,seed+idx);rng=random.Random(seed+idx+1)
    obs=a.observations(o);history=[];seen=set();valid_seen=set();events=[];last='';errors=[]
    with requests.Session() as session, (folder/'events.jsonl').open('a',encoding='utf-8') as trace, (folder/'attempts.jsonl').open('a') as audit:
        for step in range(budget):
            arm=s.choose(bool(errors));prompt,target=a.prompt(obs,history,arm,last,errors,rng)
            started=time.monotonic()
            if step<len(saved):
                e=saved[step]
                assert e['prompt']==prompt and e['arm']==arm and e['target']==target
                response=e['response'];usage=e['usage'];finish=e['finish_reason']
            else:
                audit.write(json.dumps({'step':step+1,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})+'\n');audit.flush();os.fsync(audit.fileno())
                payload=dict(model=model['served_model'],messages=[dict(role='system',content=a.system),dict(role='user',content=prompt)],temperature=model['temperature'],max_tokens=model['max_tokens'],seed=seed+step)
                payload.update(model['extra_body'])
                r=session.post(model['base_url']+'/chat/completions',json=payload,timeout=600)
                r.raise_for_status();data=r.json();choice=data['choices'][0]
                response=choice['message'].get('content') or '';usage=data.get('usage',{});finish=choice.get('finish_reason')
            candidate=a.parse(response);key,errors=a.inspect(candidate,obs)
            valid=key is not None and not errors;duplicate=key in seen if key is not None else False
            reward=int(valid and key not in valid_seen)
            if candidate is not None: history.append(candidate)
            if key is not None: seen.add(key)
            if valid: valid_seen.add(key)
            s.update(arm,reward);last=response
            event=dict(step=step+1,arm=arm,target=target,prompt=prompt,response=response,usage=usage,finish_reason=finish,
                       parsed=candidate is not None,valid=valid,duplicate=duplicate,reward=reward,key=key,errors=errors,seconds=time.monotonic()-started)
            if step<len(saved):
                for k in ['key','valid','reward','duplicate']: assert event[k]==saved[step][k]
                event=saved[step]
            else:
                trace.write(json.dumps(event,ensure_ascii=False)+'\n');trace.flush();os.fsync(trace.fileno())
            events.append(event)
    truth=a.truth(o)  # Only score against ground truths AFTER all generation.
    result=dict(id=a.iid(o),method=method,seed=seed,n_requests=budget,events=events,
                VR=sum(e['valid'] for e in events)/budget,NR=len(seen)/budget,
                RR=sum(k in valid_seen for k in truth)/len(truth),n_ground_truths=len(truth),
                parse=sum(e['parsed'] for e in events)/budget,duplicate_rate=sum(e['duplicate'] for e in events)/budget,
                total_tokens=sum(e['usage'].get('total_tokens',0) for e in events),seconds=sum(e['seconds'] for e in events),
                arm_counts=s.counts,arm_rewards=s.rewards)
    atomic_json(folder/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--smoke',action='store_true');p.add_argument('--base-url',default='http://127.0.0.1:8001/v1')
    args=p.parse_args();assert 1<=args.workers<=32
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    model=yaml.safe_load((ROOT/'configs/models/qwen3.5-9b.yaml').read_text());model['base_url']=args.base_url
    adapters={d:Adapter(d) for d in DATASETS}
    paths=list((ROOT/'third_party/HypoSpace').rglob('*.py'))+[Path(__file__),ROOT/'tools/cross_domain.py',ROOT/'tools/adaptive_causal.py',ROOT/'tools/run_formal_causal.py']
    hashes={str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    jobs=[];manifests={}
    for domain,a in adapters.items():
        ordered=sorted(a.instances,key=lambda o:hashlib.sha256(('cross-v1:'+a.iid(o)).encode()).hexdigest())
        # Smoke reuses two benchmark items strictly for plumbing validation; no tuning.
        selected=ordered[:2] if args.smoke else ordered
        manifests[domain]=dict(protocol='adaptive-cross-v1',split='smoke' if args.smoke else 'full_benchmark',ids=[a.iid(o) for o in selected],methods=METHODS,model=model,fixed_queries=3 if args.smoke else None,multiplier=1,workers=args.workers,source_hashes=hashes,dataset_sha256=hashlib.sha256(a.path.read_bytes()).hexdigest(),notes='All benchmark instances; two smoke instances reused only for pipeline validation. No disjoint development/test or untouched-test claim. Strict public constraints plus upstream structural identity. No retry or truncation.')
        for seed in SEEDS:
            for idx,o in enumerate(selected):
                methods=list(METHODS);random.Random(seed+idx).shuffle(methods)
                for method in methods: jobs.append((domain,o,idx,method,seed,3 if args.smoke else a.budget(o)))
    manifest=dict(domains=manifests,seeds=SEEDS,total_jobs=len(jobs),total_requests=sum(j[-1] for j in jobs))
    lock=output/'run.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        mp=output/'manifest.json'
        if mp.exists():assert json.loads(mp.read_text())==manifest,'Frozen manifest mismatch'
        atomic_json(mp,manifest)
        status=dict(state='running',pid=os.getpid(),completed_jobs=0,completed_requests=0,total_jobs=len(jobs),total_requests=manifest['total_requests'],started=datetime.now(timezone.utc).isoformat())
        atomic_json(output/'status.json',status)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(job,adapters[d],o,idx,m,s,b,model,output/d/'jobs'/str(s)/adapters[d].iid(o)/m):d for d,o,idx,m,s,b in jobs}
            for f in as_completed(futures):
                try:r=f.result()
                except Exception:
                    for other in futures:other.cancel()
                    raise
                status['completed_jobs']+=1;status['completed_requests']+=r['n_requests'];status['updated']=datetime.now(timezone.utc).isoformat()
                atomic_json(output/'status.json',status)
                print(f"completed {status['completed_jobs']}/{len(jobs)}",flush=True)
        from compare_adaptive_seeds import compare
        for domain,a in adapters.items():
            folders=[]
            for seed in SEEDS:
                dest=output/domain/f'seed_{seed}';dest.mkdir(exist_ok=True);folders.append(dest)
                atomic_json(dest/'manifest.json',dict(manifests[domain],seed=seed))
                with (dest/'instances.jsonl').open('w',encoding='utf-8') as f:
                    for iid in manifests[domain]['ids']:
                        for m in METHODS:
                            r=json.loads((output/domain/'jobs'/str(seed)/iid/m/'result.json').read_text())
                            f.write(json.dumps(r,ensure_ascii=False)+'\n')
            compare(folders,output/domain/'analysis')
        status.update(state='complete',finished=datetime.now(timezone.utc).isoformat());atomic_json(output/'status.json',status)
    except Exception as e:
        atomic_json(output/'failure.json',dict(error=repr(e)))
        if 'status' in locals():status.update(state='failed',error=repr(e));atomic_json(output/'status.json',status)
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
