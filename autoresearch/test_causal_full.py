"""Offline checks for the full-847 causal evaluation. No network.

    .venv/bin/python -m unittest autoresearch/test_causal_full.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import causal_full_eval  # noqa: E402
import harness  # noqa: E402
from policy import Policy  # noqa: E402


def scripted(responses):
    calls = []

    def query(messages, seed):
        calls.append(messages)
        return dict(response=responses[(len(calls) - 1) % len(responses)], usage={'total_tokens': 1},
                    finish_reason='stop')
    return query, calls


class CausalFullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = causal_full_eval.CausalFullTask()
        cls.inst = cls.task.bench.filtered_observation_sets[0]

    def test_loads_all_847_including_trivial_sets(self):
        instances = self.task.bench.filtered_observation_sets
        self.assertEqual(len(instances), 847)
        self.assertEqual(sum(len(o['ground_truth_graphs']) == 1 for o in instances), 175)
        self.assertEqual(self.task.bench.max_edges, 6)

    def test_budget_is_twice_the_reference_set(self):
        budget = harness.MULTIPLIER['causal'] * self.task.budget(self.inst)
        self.assertEqual(budget, 2 * len(self.inst['ground_truth_graphs']))

    def test_run_job_uses_exact_budget(self):
        q, calls = scripted(['Graph: A->B'])
        r = harness.run_job(Policy, 'causal', self.task, self.inst, 0, 1, 4, q)
        self.assertEqual(len(calls), 4)
        self.assertEqual(r['n_requests'], 4)
        self.assertLessEqual(r['RR'], 1.0)


if __name__ == '__main__':
    unittest.main()
