"""Observed proof-boundary failures and synthetic controller regression tests.

None of the policies in this file are evidence of model discovery.
"""
import copy
import itertools
import json
from pathlib import Path

import pytest
from connection_search.algebra import (SOpaque, prove_associative, scan_laws,
                                       serialize_poly, symbolic)
from connection_search.completion import ClosingPlanner
from connection_search.dsl import (Invalid, Runtime, R, at, fn, holes, infer, key,
                                    pair, plain, replace, seq, walk_typed)
from connection_search.library import Library
from connection_search.tasks import Case, Task, get_task
from connection_search.verify import Checker
from test_typed_choices import ChoiceBackend, setup

FIXTURE = Path(__file__).parent/'fixtures/uploaded_semantic_run.json'
ADD = ['fn', [['a', R], ['b', R]], ['add', 'a', 'b']]
SUB = ['fn', [['a', R], ['b', R]], ['sub', 'a', 'b']]
LEFT = ['fn', [['a', R], ['b', R]], 'a']
RIGHT = ['fn', [['a', R], ['b', R]], 'b']


def recorded():
    return json.loads(FIXTURE.read_text())


def wrapped(callback, argument=None):
    return ['apply', ['fn', [['unused', seq(pair())]], callback],
            argument or ['zip', 'as_', ['reverse', 'bs']]]


@pytest.mark.parametrize('callback', [ADD, LEFT, RIGHT])
def test_ignored_sequence_construction_has_a_checked_polynomial_result(callback):
    term = wrapped(callback)
    proof = prove_associative(term, {'as_': seq(R), 'bs': seq(R)}, R)
    assert proof['status'] == 'proved_polynomial_identity'
    assert proof['normal_form'] == prove_associative(callback, {}, R)['normal_form']
    assert 'strict' in proof['construction_precondition']


def test_exact_uploaded_call26_no_longer_hits_unsupported_zip():
    row = next(x for x in recorded()['checked']
               if x['calls_so_far'] == 26 and x['check']['status'] == 'law_unknown')
    before = copy.deepcopy(row['expression'])
    check = Checker(get_task('affine_prefix')).check(row['expression'])
    assert check.status == 'partial_type_checked'
    assert len(check.certificates) == 1  # Outer callback still contains a hole.
    assert check.certificates[0]['normal_form'] == [[['x'], 1]]
    assert check.evaluated_cases == 0
    assert row['expression'] == before  # No program rewriting or answer insertion.


def test_nested_scans_keep_their_own_law_obligations_even_in_unused_arguments():
    # The outer callback ignores the bad inner scan, but the language still
    # requires EVERY scan operator in the original program to have its law.
    term = ['scan', ['apply', ['fn', [['unused', seq(R)]], ADD],
                             ['scan', SUB, 'xs']], 'xs']
    check = Checker(get_task('prefix_sum')).check(term)
    assert check.status == 'law_counterexample'
    assert check.path == (1, 2, 1)
    assert not check.passed


def test_strict_runtime_still_evaluates_an_unused_bad_zip():
    class UnequalTask(Task):
        def cases(self, split, seed=1729, count=None):
            return (Case({'xs': [1, 2], 'ys': [3]}, 5, 2),)
    task = UnequalTask('prefix_sum', 'Prefix sum',
                       (('xs', seq(R)), ('ys', seq(R))), seq(R))
    term = ['scan', ['apply', ['fn', [['unused', seq(pair())]], ADD],
                             ['zip', 'xs', 'ys']], 'xs']
    check = Checker(task).check(term)
    assert check.status == 'precondition_failed'
    assert check.evaluated_cases == 1
    assert not check.passed
    assert len(scan_laws(term, task.inputs)) == 1


def test_strict_runtime_keeps_work_and_resource_limits_on_unused_arguments():
    inputs = {'as_': seq(R), 'bs': seq(R)}
    term = wrapped(ADD)
    prove_associative(term, inputs, R)
    rt = Runtime(7)
    env = {n: rt.wrap([1, 2, 3], typ) for n, typ in inputs.items()}
    result = rt.eval(term, env)
    work = rt.work
    assert plain(rt.call(result, [rt.wrap(3, R), rt.wrap(4, R)])) == 0
    simple = Runtime(7); simple.eval(ADD, {})
    assert work > simple.work
    limited = Runtime(7, step_limit=3)
    with pytest.raises(Invalid) as exc:
        limited.eval(term, env)
    assert exc.value.kind == 'resource_limit'


@pytest.mark.parametrize('value', [SOpaque('test'), (SOpaque('test'), SOpaque('test'))])
def test_equal_opaque_values_cannot_be_serialized_as_a_polynomial_proof(value):
    with pytest.raises(Invalid) as exc:
        serialize_poly(value)
    assert exc.value.kind == 'law_unknown'


def test_an_opaque_function_that_is_actually_used_remains_unknown():
    term = ['fn', [['a', R], ['b', R]], ['apply', 'external', 'a']]
    with pytest.raises(Invalid) as exc:
        prove_associative(term, {'external': fn([R], R)}, R)
    assert exc.value.kind == 'law_unknown'


def test_ignored_bad_type_cannot_acquire_a_certificate():
    bad = ['apply', ['fn', [['unused', seq(R)]], ADD], 7]
    with pytest.raises(Invalid):
        prove_associative(bad, {}, R)


def test_holes_are_not_erased_from_a_polynomial_law_request():
    term = ['apply', ['fn', [['unused', seq(R)]], ADD], ['hole', seq(R)]]
    with pytest.raises(Invalid) as exc:
        prove_associative(term, {}, R)
    assert exc.value.kind == 'law_unknown'


def test_nonassociative_callback_wrapped_in_ignored_sequences_is_not_approved():
    with pytest.raises(Invalid) as exc:
        prove_associative(wrapped(SUB), {'as_': seq(R), 'bs': seq(R)}, R)
    assert exc.value.kind == 'law_counterexample'
    assert exc.value.evidence['left'] != exc.value.evidence['right']


def test_enumeration_keeps_its_original_strict_symbolic_policy():
    with pytest.raises(Invalid) as exc:
        symbolic(['zip', 'xs', 'xs'], {}, [100])
    assert exc.value.kind == 'law_unknown'


def test_wrapped_laws_agree_with_concrete_execution_exhaustively():
    callbacks = [ADD, LEFT, RIGHT,
                 ['fn', [['a', R], ['b', R]], ['mul', 'a', 'b']],
                 ['fn', [['a', R], ['b', R]], ['add', ['add', 'a', 'b'], 'k']]]
    tested = 0
    for modulus in (2, 3, 5):
        for callback in callbacks:
            term = wrapped(callback)
            types = {'as_': seq(R), 'bs': seq(R), 'k': R}
            assert prove_associative(term, types, R)['status'] == 'proved_polynomial_identity'
            for k in range(modulus):
                rt = Runtime(modulus)
                env = {n: rt.wrap([1, 2] if n != 'k' else k, typ) for n, typ in types.items()}
                closure = rt.eval(term, env)
                for triple in itertools.product(range(modulus), repeat=3):
                    x, y, z = [rt.wrap(v, R) for v in triple]
                    a = rt.call(closure, [rt.call(closure, [x, y]), z])
                    b = rt.call(closure, [x, rt.call(closure, [y, z])])
                    assert plain(a) == plain(b)
                    tested += 1
    assert tested == 3610


def test_lexical_capture_is_preserved_through_opaque_construction():
    inner = ['fn', [['a', R], ['b', R]], ['add', ['add', 'a', 'b'], 'k']]
    term = ['apply', ['fn', [['k', R]], wrapped(inner)], 3]
    types = {'as_': seq(R), 'bs': seq(R), 'k': R}
    proof = prove_associative(term, types, R)
    assert proof['normal_form'] == [[[], 6], [['x'], 1], [['y'], 1], [['z'], 1]]


def test_recorded_late_expansion_is_filtered_by_global_budget(tmp_path):
    calls = recorded()['calls']; row = calls[26]  # Call 27, five calls afterward.
    task = get_task('affine_prefix'); planner = ClosingPlanner(Library())
    after = replace(row['construction'], tuple(row['path']), row['selected_term'])
    assert planner.fill(after, task.inputs).steps == 8
    assert not planner.admissible({'chosen': row['selected_term']}, row['construction'],
                                  tuple(row['path']), task.inputs, 6)
    # Checking the real controller boundary rather than only the helper.
    def policy(payload):
        assert row['selected_term'] not in [c['term'] for c in payload['legal_choices']]
        for c in payload['legal_choices']:
            witness = planner.fill(replace(payload['construction'], tuple(payload['focus']['path']), c['term']), task.inputs)
            assert witness and witness.steps <= 5
        return payload['legal_choices'][0]['id']
    s = setup(tmp_path, ChoiceBackend(policy), task='affine_prefix', max_calls=32)
    s.inference.budget.calls = 26
    state = s.submit(row['construction'])
    spot = next(v for v in walk_typed(state.expr, task.inputs)
                if v[0] == tuple(row['path']))
    s.focus = lambda state: spot
    s.ask_expansion(state)
    assert s.inference.budget.calls == 27


def test_over_budget_partial_state_is_retained_but_not_selected(tmp_path):
    s = setup(tmp_path, ChoiceBackend(lambda p:p['legal_choices'][0]['id']), max_calls=1)
    state = s.submit(['add', ['hole', R], ['hole', R]])
    assert not s.eligible(state)
    assert not state.archived and state.id in s.states
    assert state.check.status == 'partial_type_checked'
    assert any(e['event'] == 'state_horizon_deferred' for e in s.log.events)


def test_law_unknown_gets_a_repair_turn_without_becoming_a_refutation(tmp_path):
    s = setup(tmp_path, ChoiceBackend(lambda p:p['legal_choices'][0]['id']), task='prefix_sum')
    # Force the normalizer's finite degree limit via an otherwise typed function.
    from connection_search.verify import Check
    state = s.submit(['scan', ['hole', fn([R, R], R)], 'xs'])
    state.check = Check('law_unknown', 'Unsupported proof dependency', path=(1,))
    s.observe_failure(state)
    assert s.select() is state
    assert s.last_turn_was_revision
    assert not s.refuted(state)
    event = next(e for e in s.log.events if e['event'] == 'repair_turn')
    assert not event['mathematical_refutation']
    assert s.context(observations=True)['search_observations'][0]['observation_kind'] == 'checker_limitation'
    assert state.completion_trials == 0


def test_recorded_26_selection_prefix_then_synthetic_closure_reaches_execution(tmp_path):
    fixture = recorded(); calls = fixture['calls']; consumed = []
    def policy(payload):
        if len(consumed) < 26:
            row = calls[len(consumed)]
            assert payload['stage'] == row['stage']
            assert payload['focus']['path'] == row['path']
            assert payload['construction'] == row['construction']
            choice = next(c for c in payload['legal_choices'] if c['term'] == row['selected_term'])
            consumed.append(choice['term'])
            return choice['id']
        # These choices are explicitly synthetic, not new Gemma responses.
        terminal = next((c for c in payload['legal_choices'] if c['term'] == 'bs'), None)
        return (terminal or payload['legal_choices'][0])['id']
    s = setup(tmp_path, ChoiceBackend(policy), task='affine_prefix', max_calls=28)
    # Replay begins from the 32-call experiment's horizon. Stop after the two
    # synthetic closures via a checker failure, not a changed historical budget.
    s.inference.budget.max_calls = 32
    s.config.frames = 3; s.config.bridge_checks = 2
    # Avoid generating further synthetic paths after the first executed failure.
    from connection_search.inference import BudgetExhausted
    original = s.inference.ask
    def ask(*args, **kwargs):
        if s.inference.budget.calls >= 28:
            raise BudgetExhausted('synthetic_replay_cutoff')
        return original(*args, **kwargs)
    s.inference.ask = ask
    s.graph_search()
    assert len(consumed) == 26
    assert s.checker.case_evaluations > 0
    assert s.full_candidates >= 1
    assert not s.solution
    assert s.inference.budget.calls == 28


def test_global_horizon_keeps_a_one_call_reusable_completion(tmp_path):
    s = setup(tmp_path, ChoiceBackend(lambda p: next(c['id'] for c in p['legal_choices']
                 if c['term'] == ADD)), task='prefix_sum', max_calls=1)
    # Record a component through the same graph mechanism used by real search.
    from connection_search.verify import Check
    s.graph.record(ADD, s.task.inputs, Check('partial_type_checked'))
    state = s.submit(['scan', ['hole', fn([R, R], R)], 'xs'])
    assert s.eligible(state)
    child = s.ask_expansion(state)
    assert child and child.check.passed
    assert s.inference.budget.calls == 1


def test_closing_planner_rechecks_after_graph_component_growth(tmp_path):
    s = setup(tmp_path, ChoiceBackend(lambda p:p['legal_choices'][0]['id']), task='prefix_sum', max_calls=1)
    state = s.submit(['scan', ['hole', fn([R, R], R)], 'xs'])
    assert not s.eligible(state)
    from connection_search.verify import Check
    s.graph.record(ADD, s.task.inputs, Check('partial_type_checked'))
    assert s.eligible(state)
    assert s.closing_planner.fill(state.expr, s.task.inputs).steps == 1
