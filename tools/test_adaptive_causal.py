"""Offline behavioral tests: budget, reward, feedback and test isolation."""
import unittest
from adaptive_causal import CausalBenchmarkEnhanced, Scheduler, evaluate, feedback, split_instances, ROOT
from modules.models import CausalGraph


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = CausalBenchmarkEnhanced(str(ROOT/'data/causal/n4.json'), gt_filter=(8,200))

    def test_duplicate_and_invalid_consume_budget(self):
        bench = self.bench
        observations = [{'perturbed_node':'A','effects':{'A':0,'B':1,'C':0,'D':0},
                         'string':'Perturb(A) -> A:0 B:1 C:0 D:0'}]
        answers = ['Graph: A->B', 'Graph: A->B', 'Graph: No edges', 'garbage']
        calls = []
        def query(prompt, step):
            calls.append(prompt)
            return {'response':answers[step]}
        result = evaluate(bench, observations, 4, 'baseline', 1, query)
        self.assertEqual(len(calls), 4)
        self.assertEqual([e['reward'] for e in result['events']], [1,0,0,0])
        self.assertEqual(result['VR'], .5)
        self.assertEqual(result['NR'], .5)
        self.assertIn('A->B', calls[1])
        self.assertIn('must be 1', result['events'][2]['errors'][0])

    def test_scheduler_availability_and_learning(self):
        s = Scheduler('ucb', 1)
        for _ in range(100):
            arm = s.choose(False)
            self.assertNotEqual(arm, 'repair')
            s.update(arm, int(arm == 'explore'))
        self.assertGreater(s.counts['explore'], s.counts['structure'])

    def test_fixed_repair_switches_on_observed_error(self):
        s = Scheduler('repair_only', 1)
        self.assertEqual(s.choose(False), 'explore')
        self.assertEqual(s.choose(True), 'repair')
        observations = [{'perturbed_node':'A','effects':{'A':0,'B':1,'C':0,'D':0},
                         'string':'Perturb(A) -> A:0 B:1 C:0 D:0'}]
        answers = ['Graph: No edges', 'Graph: A->B', 'Graph: A->B']
        result = evaluate(self.bench, observations, 3, 'repair_only', 1,
                          lambda prompt, step: {'response':answers[step]})
        self.assertEqual([e['arm'] for e in result['events']], ['explore','repair','explore'])
        self.assertIn('Validation feedback', result['events'][1]['prompt'])
        self.assertNotIn('Validation feedback', result['events'][2]['prompt'])

    def test_split_disjoint(self):
        splits = split_instances(self.bench)
        dev = {o['observation_set_id'] for o in splits['dev']}
        test = {o['observation_set_id'] for o in splits['test']}
        self.assertTrue(dev)
        self.assertEqual(len(test), 200)
        self.assertFalse(dev & test)

    def test_cycle_and_unknown_node_rejected(self):
        self.assertIsNone(self.bench.parse_llm_response('Graph: A->B, B->A'))
        self.assertIsNone(self.bench.parse_llm_response('Graph: A->Z'))

    def test_feedback_matches_enumerated_space(self):
        graphs = {CausalGraph.from_dict(g).get_hash(): CausalGraph.from_dict(g)
                  for o in self.bench.all_observation_sets for g in o['ground_truth_graphs']}
        self.assertEqual(len(graphs), 543)
        for instance in split_instances(self.bench)['dev'][:12]:
            truth = {CausalGraph.from_dict(g).get_hash() for g in instance['ground_truth_graphs']}
            for key, graph in graphs.items():
                self.assertEqual(not feedback(self.bench, graph, instance['observations']), key in truth)


if __name__ == '__main__':
    unittest.main()
