"""The only file the autoresearch agent edits.

The harness builds one Policy per (instance, seed) and calls, `budget` times:
    messages = policy.next_messages()   -> one chat request, one candidate
    policy.observe(outcome)             -> the harness's parse/check result for it

`info` (harness.TaskInfo) carries domain, observations, public constraints, the
upstream system prompt and `base_prompt(history)` = upstream prompt with prior
candidates. Outcome carries response, candidate, key, valid, duplicate, errors.
Validity feedback comes only from observations and public constraints.

Starting point: repair_only from the feedback-guided study (best mean RR on
causal and Boolean Extended).
"""
import math

ARMS = ('explore', 'repair', 'structure')
METHOD = 'repair_only'  # baseline | repair_only | random | ucb

MUTATIONS = {
    'caus': ('add one edge', 'remove one edge', 'reverse one edge (keep it acyclic)',
             'replace one edge with a different edge'),
    '3d': ('raise one column by 1', 'lower one column by 1 (a column keeps at least 1 block)',
           'raise one column and lower another', 'change the heights of two columns'),
    'bool': ('replace one operator', 'negate or double-negate a sub-term',
             'wrap a sub-term with another operator', 'add a redundant term'),
}


class Policy:
    def __init__(self, info, rng):
        self.info, self.rng = info, rng
        self.history = []
        self.last_response, self.last_errors = '', ()
        self.counts = dict.fromkeys(ARMS, 0)
        self.rewards = dict.fromkeys(ARMS, 0)
        self.valid_seen = set()
        self.last_duplicate = False
        self.history_keys = set()
        self.valid_texts = []
        self.arm = None

    def choose(self):
        active = [a for a in ARMS if a != 'repair' or self.last_errors]
        if METHOD == 'baseline':
            return 'explore'
        if METHOD == 'repair_only':
            return 'repair' if self.last_errors else 'explore'
        if METHOD == 'random':
            return self.rng.choice(active)
        untried = [a for a in active if not self.counts[a]]
        if untried:
            return self.rng.choice(untried)
        total = sum(self.counts.values())
        scores = {a: self.rewards[a] / self.counts[a]
                  + math.sqrt(2 * math.log(total + 1) / self.counts[a]) for a in active}
        best = max(scores.values())
        return self.rng.choice([a for a in active if scores[a] == best])

    def direction(self):
        d, pub = self.info.domain, self.info.public
        if d == 'causal':
            return f'try a DAG with exactly {self.rng.randint(0, pub["max_edges"])} edges'
        if d == '3d':
            return f'try a structure with exactly {self.rng.randint(1, pub["max_height"])} nonempty layers'
        return f'try an expression whose outermost operator is {self.rng.choice(pub["operators"])}'

    def causal_hints(self):
        """Restate each observation as reachability constraints (no candidate graphs are given)."""
        lines = ['\nHow to read the observations as constraints:']
        for o in self.info.observations:
            p = o['perturbed_node']
            on = [n for n, v in o['effects'].items() if v == 1 and n != p]
            off = [n for n, v in o['effects'].items() if v == 0 and n != p]
            if on:
                lines.append(f'- {", ".join(on)} must each be reachable from {p} by a directed path '
                             f'(a direct edge {p}->X or a chain {p}->...->X).')
            if off:
                lines.append(f'- {", ".join(off)} must NOT be reachable from {p}: no edge {p}->X for these, '
                             f'and no node reachable from {p} may have a path into them.')
        return '\n'.join(lines)
    def mutation_prompt(self):
        base = self.rng.choice(self.valid_texts)
        return ('\nA known valid answer is:\n' + base + '\nProduce a NEW answer by making one small '
                'structural change to it: ' + self.rng.choice(MUTATIONS[self.info.domain[:4]]) + '. It must '
                'still satisfy ALL observations and differ from every prior prediction (if this edit cannot, '
                'use another small edit). Keep the same output format.')

    def next_messages(self):
        self.arm = self.choose()
        prompt = self.info.base_prompt(self.history)
        if self.info.domain == 'causal':
            prompt += self.causal_hints()
        if self.arm == 'repair':
            comma = ',' if self.info.domain == 'causal' else ''  # matches the formal runs verbatim
            prompt += (f'\nRepair the previous candidate{comma} while satisfying ALL observations and avoiding '
                       'prior predictions.\nPrevious response:\n' + self.last_response[:2000]
                       + '\nValidation feedback:\n' + '\n'.join(self.last_errors))
        elif self.last_duplicate:
            prompt += ('\nYour previous answer was IDENTICAL to a prior prediction, which wastes the attempt:\n'
                       + self.last_response[:500].strip()
                       + '\nOutput a candidate that is genuinely different from EVERY prior prediction listed above, '
                       'while still satisfying all observations. Keep the same output format.')
        elif self.valid_texts:
            prompt += self.mutation_prompt()
        elif self.arm == 'structure':
            prompt += (f'\nExploration direction: {self.direction()}. If impossible, return another valid, '
                       'non-repeated candidate. Keep the same output format.')
        return [{'role': 'system', 'content': self.info.system}, {'role': 'user', 'content': prompt}]

    def observe(self, outcome):
        reward = int(outcome.valid and outcome.key not in self.valid_seen)
        if outcome.valid and outcome.key not in self.valid_seen:
            self.valid_texts.append(outcome.response.strip()[:600])
        if outcome.valid:
            self.valid_seen.add(outcome.key)
        if outcome.candidate is not None and outcome.key not in self.history_keys:
            self.history_keys.add(outcome.key)
            self.history.append(outcome.candidate)
        self.counts[self.arm] += 1
        self.rewards[self.arm] += reward
        self.last_response, self.last_errors = outcome.response, outcome.errors
        self.last_duplicate = outcome.duplicate
