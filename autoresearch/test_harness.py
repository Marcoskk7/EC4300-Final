"""Offline checks for the locked harness. No network.

    .venv/bin/python -m unittest autoresearch/test_harness.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
from adaptive_causal import evaluate  # noqa: E402
import policy as policy_mod  # noqa: E402


def scripted(responses):
    calls = []

    def query(messages, seed):
        calls.append(messages)
        return dict(response=responses[(len(calls) - 1) % len(responses)], usage={'total_tokens': 1},
                    finish_reason='stop')
    return query, calls


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.causal = harness.CausalTask()
        cls.inst = harness.split('causal', cls.causal)['dev'][0]

    def test_repair_only_matches_original_causal_prompts(self):
        responses = ['Graph: A->B', 'nonsense', 'Graph: No edges', 'Graph: A->B, B->C']
        q1, calls_new = scripted(responses)
        harness.run_job(policy_mod.Policy, 'causal', self.causal, self.inst, 0, 7, 4, q1)
        q2, calls_old = scripted(responses)
        evaluate(self.causal.bench, self.inst['observations'], 4, 'repair_only', 7,
                 lambda prompt, step: q2([{'role': 'user', 'content': prompt}], step))
        self.assertEqual([c[-1]['content'] for c in calls_new], [c[-1]['content'] for c in calls_old])

    def test_exact_budget_and_scoring_after_generation(self):
        q, calls = scripted(['Graph: A->B'])
        r = harness.run_job(policy_mod.Policy, 'causal', self.causal, self.inst, 0, 1, 5, q)
        self.assertEqual(len(calls), 5)
        self.assertEqual(r['n_requests'], 5)
        self.assertLessEqual(r['RR'], 1 / len(self.inst['ground_truth_graphs']))

    def test_split_disjoint_and_stable(self):
        for task in harness.TASKS:
            a = self.causal if task == 'causal' else harness.Adapter(task)
            s1, s2 = harness.split(task, a), harness.split(task, a)
            dev = {a.iid(o) for o in s1['dev']}
            test = {a.iid(o) for o in s1['test']}
            self.assertFalse(dev & test, task)
            self.assertEqual(dev, {a.iid(o) for o in s2['dev']})

    def test_policy_source_guard(self):
        bad = ['import os\n', 'from pathlib import Path\n', 'x = open("f")\n', 'y = ().__class__\n',
               'import requests\n', 'from harness import split\n']
        for src in bad:
            with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
                f.write(src)
            with self.assertRaises(ValueError, msg=src):
                harness.check_policy_source(Path(f.name))
        harness.check_policy_source(harness.HERE / 'policy.py')

    def test_message_validation(self):
        with self.assertRaises(ValueError):
            harness.validate_messages([])
        with self.assertRaises(ValueError):
            harness.validate_messages([{'role': 'user', 'content': 'x', 'n': 3}])


if __name__ == '__main__':
    unittest.main()
