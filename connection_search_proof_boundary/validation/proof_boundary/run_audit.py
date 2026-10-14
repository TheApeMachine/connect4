#!/usr/bin/env python3
"""Dependency-free audit. Replays observed selections, not model inference."""
from pathlib import Path
import collections
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from connection_search.backend import Generation
from connection_search.completion import ClosingPlanner
from connection_search.dsl import R, holes, replace, seq
from connection_search.feasibility import Feasibility
from connection_search.graph import Graph
from connection_search.inference import Budget, BudgetExhausted, Inference, Log
from connection_search.library import Library
from connection_search.report import write_outputs
from connection_search.search import Config, Search
from connection_search.tasks import get_task
from connection_search.verify import Checker

OUT = Path(__file__).resolve().parent
fixture = json.loads((ROOT/'tests/fixtures/uploaded_semantic_run.json').read_text())
task = get_task('affine_prefix')
checker = Checker(task)
feasibility = Feasibility(task.inputs, task.output, task.public()['public_examples'])
checks = []
for row in fixture['checked']:
    original = copy.deepcopy(row['expression'])
    check = checker.check(row['expression'])
    if check.status == 'partial_type_checked':
        check = feasibility.check(row['expression']) or check
    assert row['expression'] == original
    checks.append({'state': row['state'], 'calls_so_far': row['calls_so_far'],
                   'old_status': row['check']['status'], 'new_status': check.status,
                   'new_check': check.to_dict()})
planner = ClosingPlanner(Library())
late = []
for row in fixture['calls'][26:]:
    after = replace(row['construction'], tuple(row['path']), row['selected_term'])
    plan = planner.fill(after, task.inputs)
    remaining = 32 - row['call_number']
    graph = Graph()
    for prior in fixture['checked']:
        if prior['calls_so_far'] >= row['call_number']:
            break
        graph.record(prior['expression'], task.inputs, Checker(task).check(prior['expression']))
    with_reuse = ClosingPlanner(Library(), components=graph.available).fill(after, task.inputs)
    late.append({'recorded_call': row['call_number'], 'calls_left_after_selection': remaining,
                 'original_syntax_only_witness_steps': plan.steps if plan else None,
                 'exceeds_original_syntax_only_witness_budget': plan is None or plan.steps > remaining,
                 'witness_steps_with_prior_recorded_components': with_reuse.steps if with_reuse else None,
                 'excluded_with_prior_recorded_components': with_reuse is None or with_reuse.steps > remaining,
                 'note': 'Local counterfactual at the historical state, not a prediction of the changed search.'})

class CharacterTokenizer:
    eos_token_id = 0
    pad_token_id = 1
    all_special_ids = [0, 1]
    def encode(self, text, add_special_tokens=False):
        return [ord(c)+3 for c in text]
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(int(i)-3) for i in ids if int(i) >= 3)

class ReplayBackend:
    synthetic = True
    tokenizer = CharacterTokenizer()
    def __init__(self):
        self.count = 0
    def count_prompt_tokens(self, system, user):
        # Synthetic accounting units, explicitly not Gemma tokenization.
        return len((system+user).encode('utf-8'))
    def generate_choice(self, *, system, user, seed, max_new_tokens, temperature, labels):
        payload = json.loads(user)
        if self.count < 26:
            row = fixture['calls'][self.count]
            assert payload['stage'] == row['stage']
            assert payload['focus']['path'] == row['path']
            assert payload['construction'] == row['construction']
            choice = next(c for c in payload['legal_choices'] if c['term'] == row['selected_term'])
        else:
            choice = next(c for c in payload['legal_choices'] if c['term'] == 'bs')
        self.count += 1
        return Generation(choice['id'], self.count_prompt_tokens(system, user), 5, 0.0)

class CutoffInference(Inference):
    def ask(self, *args, **kwargs):
        if self.budget.calls >= 28:
            raise BudgetExhausted('synthetic_replay_cutoff')
        return super().ask(*args, **kwargs)

run_out = OUT/'recorded_prefix_plus_two_synthetic_choices'
log = Log(run_out/'events.jsonl')
backend = ReplayBackend()
inf = CutoffInference(backend, Budget(max_calls=32, max_tokens=1_000_000), log)
search = Search(task, inf, log, Config(proposal_mode='typed', frames=3, bridge_checks=2))
log.write('audit_provenance', historical_selections_replayed=26,
          synthetic_continuation='Choose bs for the two remaining sequence holes',
          real_model_inference_executed=False,
          note='This is an interface/controller replay, not a fresh model result; prompts can differ.')
search.graph_search()
summary = search.finish('graph')
summary['audit'] = {'historical_selections': 26, 'synthetic_choices': 2,
                    'new_model_behavior_established': False}
write_outputs(run_out, search, summary)
assert backend.count == 28 and summary['metrics']['case_evaluations'] > 0
assert summary['status'] == 'not_solved'
audit = {
    'basis': 'Uploaded semantic-build expressions and decisions; no fresh model inference',
    'checked_recorded_states': len(checks),
    'old_status_counts': dict(collections.Counter(c['old_status'] for c in checks)),
    'new_status_counts': dict(collections.Counter(c['new_status'] for c in checks)),
    'changed_checks': [c for c in checks if c['old_status'] != c['new_status']],
    'late_horizon_audit': late,
    'replay': {'historical_decisions_replayed': 26, 'synthetic_decisions': 2,
               'complete_candidates': search.full_candidates,
               'case_evaluations': search.checker.case_evaluations,
               'status': summary['status'], 'genuine_model_run': False},
    'unchanged_runtime_programs': True,
}
(OUT/'uploaded_run_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='changed_checks'}, indent=2))
