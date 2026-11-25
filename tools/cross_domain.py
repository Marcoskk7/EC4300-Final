"""Observation-only adapters; upstream prompts, parsing and structural identity."""
import ast
import json
import re
import sys
from pathlib import Path
import numpy as np
import sympy as sp
from adaptive_causal import ROOT
sys.path.insert(0,str(ROOT/'third_party/HypoSpace/boolean'))
sys.path.insert(0,str(ROOT/'third_party/HypoSpace/3d'))
from boolean_benchmark import BooleanBenchmarkRefined
from boolean_dataset import BooleanObservation, BooleanExpression
from run_3d_benchmark import Benchmark3D, Structure3D

DATASETS={'3d':'3d/g3_h3_b3.json', 'boolean_basic':'boolean/basic_d2.json',
          'boolean_extended':'boolean/extended_d2.json','boolean_full':'boolean/full_d2.json'}


class Adapter:
    def __init__(self,domain):
        self.domain=domain
        self.path=ROOT/'data'/DATASETS[domain]
        self.is3d=domain=='3d'
        self.bench=(Benchmark3D if self.is3d else BooleanBenchmarkRefined)(str(self.path))
        self.instances=self.bench.all_observation_sets
        self.system=('You are an expert in 3D spatial reasoning. Answer with the requested Structure/Layer format only, no explanation.'
                     if self.is3d else 'You are an expert in Boolean logic and expression synthesis.')

    def iid(self,o):
        return o['observation_id' if self.is3d else 'observation_set_id']

    def observations(self,o):
        return o['observation'] if self.is3d else [BooleanObservation(x['inputs'],x['output']) for x in o['observations']]

    def truth(self,o):
        if not self.is3d:
            return [repr(ast.literal_eval(g['mechanistic_key'])) for g in o['ground_truth_expressions']]
        return [Structure3D([self.bench._parse_observation(l).tolist() for l in g['layers']]).get_hash()
                for g in o['ground_truth_structures']]

    def budget(self,o):
        return len(o['ground_truth_structures' if self.is3d else 'ground_truth_expressions'])

    def parse(self,response):
        candidate=self.bench.parse_llm_response(response)
        if not self.is3d and candidate:
            # SymPy's parser must never receive arbitrary generated Python.
            tokens=re.findall(r'[A-Za-z_]+|\d+|[^\s]',candidate)
            allowed=set(self.bench.variables)|{'AND','OR','NOT','NOR','XOR','(',')',','}
            if any(t not in allowed for t in tokens): return None
        return candidate

    def inspect(self,candidate,obs):
        if candidate is None:
            return None,['Cannot parse the required output. Follow the specified format exactly.']
        if not self.is3d:
            if not self.in_space(candidate):
                return None,[f'Expression violates the allowed variables/operators, no-constants rule or maximum depth {self.bench.max_depth}.']
            expr=BooleanExpression(candidate,self.bench.variables,self.bench.operators)
            errors=[f'{o.to_string()}: your expression predicts {expr.evaluate(o.inputs)}.'
                    for o in obs if expr.evaluate(o.inputs)!=o.output]
            return repr(self.bench.get_expression_mechanistic_key(candidate)),errors
        errors=[]
        target=self.bench._parse_observation(obs)
        if not candidate.layers or candidate.shape!=target.shape:
            return None,['Use the specified grid dimensions and at least one nonempty bottom layer.']
        if candidate.height>self.bench.metadata['max_height']:
            errors.append('Exceeds maximum layer count.')
        if not any(np.any(layer) for layer in candidate.layers):
            return None,['Structure is empty; match the nonempty observed top view.']
        if not np.any(candidate.layers[0]): errors.append('Bottom layer must not be empty.')
        for z,layer in enumerate(candidate.layers):
            if not np.isin(layer,[0,1]).all(): errors.append('Cells must be 0 or 1.')
            if z and np.any((layer==1)&(candidate.layers[z-1]!=1)):
                errors.append(f'Layer {z+1} contains blocks without direct support below.')
        top=candidate.get_top_view()
        for r,c in np.argwhere(top!=target):
            errors.append(f'Top view row {r+1}, column {c+1}: expected {target[r,c]}, predicted {top[r,c]}.')
        return candidate.get_hash(),errors

    def in_space(self,candidate):
        expr=BooleanExpression(candidate,self.bench.variables,self.bench.operators).sympy_expr
        if expr is None or not expr.free_symbols.issubset({sp.Symbol(v) for v in self.bench.variables}):return False
        def allowed(node):
            if isinstance(node,sp.Symbol):return True
            if node in (sp.true,sp.false):return False
            if isinstance(node,sp.Not) and isinstance(node.args[0],sp.Or) and 'NOR' in self.bench.operators:
                return all(allowed(x) for x in node.args[0].args)
            op={sp.And:'AND',sp.Or:'OR',sp.Not:'NOT',sp.Xor:'XOR'}.get(node.func)
            return op in self.bench.operators and all(allowed(x) for x in node.args)
        def depth(node):
            if isinstance(node,sp.Symbol):return 0
            kids=node.args
            if isinstance(node,sp.Not) and isinstance(node.args[0],sp.Or) and 'NOR' in self.bench.operators:
                kids=node.args[0].args  # Dataset counts primitive NOR as one level.
            return 1+max((depth(x) for x in kids),default=0)
        return allowed(expr) and depth(expr)<=self.bench.max_depth

    def prompt(self,obs,history,arm,last,errors,rng):
        prompt=self.bench.create_prompt(obs,history)
        target=None
        if arm=='repair':
            prompt+='\nRepair the previous candidate while satisfying ALL observations and avoiding prior predictions.\nPrevious response:\n'+last[:2000]+'\nValidation feedback:\n'+'\n'.join(errors)
        elif arm=='structure':
            if self.is3d:
                target=rng.randint(1,self.bench.metadata['max_height'])
                direction=f'try a structure with exactly {target} nonempty layers'
            else:
                target=rng.choice(sorted(self.bench.operators))
                direction=f'try an expression whose outermost operator is {target}'
            prompt+=f'\nExploration direction: {direction}. If impossible, return another valid, non-repeated candidate. Keep the same output format.'
        return prompt,target
