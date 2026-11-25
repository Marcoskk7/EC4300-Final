"""Check that recovery cannot silently retry or alter saved decisions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from run_formal_causal import run_job, read_journal
from adaptive_causal import CausalBenchmarkEnhanced, ROOT, split_instances


class Response:
    status_code=200
    def raise_for_status(self):
        pass
    def json(self):
        return {'choices':[{'message':{'content':'Graph: A->B'},'finish_reason':'stop'}],
                'usage':{'total_tokens':10}}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench=CausalBenchmarkEnhanced(str(ROOT/'data/causal/n4.json'),gt_filter=(8,200))
        cls.bench.max_edges=6
        cls.instance=split_instances(cls.bench)['dev'][0]
        cls.model=dict(served_model='fake',temperature=.7,max_tokens=1024,base_url='http://unused')

    def test_replay_and_completed_job_make_no_extra_requests(self):
        with tempfile.TemporaryDirectory() as d, patch('run_formal_causal.requests.Session') as session:
            post=session.return_value.post
            post.return_value=Response()
            folder=Path(d)
            run_job(self.bench,self.instance,0,'ucb',42,4,self.model,folder)
            self.assertEqual(post.call_count,4)
            original=json.loads((folder/'result.json').read_text())
            (folder/'result.json').unlink()
            run_job(self.bench,self.instance,0,'ucb',42,4,self.model,folder)
            self.assertEqual(post.call_count,4)
            self.assertEqual(original,json.loads((folder/'result.json').read_text()))
            run_job(self.bench,self.instance,0,'ucb',42,4,self.model,folder)
            self.assertEqual(post.call_count,4)

    def test_unknown_interrupted_request_consumes_budget(self):
        with tempfile.TemporaryDirectory() as d, patch('run_formal_causal.requests.Session') as session:
            folder=Path(d)
            (folder/'attempts.jsonl').write_text('{"step":1}\n')
            session.return_value.post.return_value=Response()
            result=run_job(self.bench,self.instance,0,'ucb',42,4,self.model,folder)
            self.assertEqual(session.return_value.post.call_count,3)
            self.assertEqual(result['n_requests'],4)
            self.assertEqual(result['interrupted_unknown'],1)
            self.assertEqual(len(read_journal(folder/'attempts.jsonl')),4)
            self.assertEqual(len(read_journal(folder/'events.jsonl')),4)

    def test_partial_tail_preserves_completed_events(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'journal.jsonl'
            p.write_bytes(b'{"step":1}\n{"step":')
            self.assertEqual(read_journal(p),[{'step':1}])
            self.assertEqual(p.read_bytes(),b'{"step":1}\n')


if __name__=='__main__':
    unittest.main()
