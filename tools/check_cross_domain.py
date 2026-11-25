"""Exhaustively verify stored ground truths and observation feedback before inference."""
from cross_domain import Adapter,DATASETS,Structure3D

def main():
    for domain in DATASETS:
        a=Adapter(domain);count=0;fail=[]
        for o in a.instances:
            obs=a.observations(o);truth=a.truth(o)
            gs=o['ground_truth_structures' if a.is3d else 'ground_truth_expressions']
            for g,k in zip(gs,truth):
                candidate=(Structure3D([a.bench._parse_observation(l).tolist() for l in g['layers']]) if a.is3d else g['formula'])
                key,errors=a.inspect(candidate,obs)
                if key!=k or errors:fail.append((a.iid(o),str(candidate),errors,key,k))
                count+=1
        print(domain,'instances',len(a.instances),'ground truths',count,'failures',len(fail),'requests',sum(a.budget(o) for o in a.instances)*8,flush=True)
        if fail:raise AssertionError(str(fail[:3]))
        obs=a.observations(a.instances[0])
        assert a.inspect(None,obs)[1]
        if not a.is3d:
            assert a.parse("Expression: __import__('os').system('echo bad')") is None
            for o in a.instances:
                candidate='x'
                _,errors=a.inspect(candidate,a.observations(o))
                expected=any(x['inputs']['x']!=x['output'] for x in o['observations'])
                assert bool(errors)==expected
    print('Cross-domain checks passed',flush=True)

if __name__=='__main__':main()
