import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from cross_domain import Adapter,Structure3D
from run_cross_domain import job

class Tests(unittest.TestCase):
    def test_duplicate_budget_and_replay(self):
        a=Adapter('boolean_basic');o=a.instances[0]
        model=dict(served_model='test',temperature=.7,max_tokens=1024,extra_body={},base_url='http://unused')
        with tempfile.TemporaryDirectory() as d, patch('run_cross_domain.requests.Session') as mocked:
            post=mocked.return_value.__enter__.return_value.post
            post.return_value.json.return_value={'choices':[{'message':{'content':'Expression: x'},'finish_reason':'stop'}],'usage':{}}
            r=job(a,o,0,'ucb',1,3,model,Path(d))
            self.assertEqual(post.call_count,3)
            self.assertEqual([e['reward'] for e in r['events']],[1,0,0])
            (Path(d)/'result.json').unlink()
            self.assertEqual(job(a,o,0,'ucb',1,3,model,Path(d)),r)
            self.assertEqual(post.call_count,3)
    def test_domain_constraints(self):
        a=Adapter('boolean_extended')
        self.assertTrue(a.in_space('NOT (x OR y)'))
        self.assertFalse(a.in_space('NOT (NOT (x OR y))'))
        a=Adapter('boolean_full')
        self.assertTrue(a.in_space('NOT (NOT (x OR y))'))
        a=Adapter('3d');obs='100000000'
        _,errors=a.inspect(Structure3D([[[0]*3 for _ in range(3)]]),obs)
        self.assertTrue(errors)
        _,errors=a.inspect(Structure3D([[[0]*3 for _ in range(3)],[[1,0,0],[0,0,0],[0,0,0]]]),obs)
        self.assertTrue(errors)

if __name__=='__main__':unittest.main()
