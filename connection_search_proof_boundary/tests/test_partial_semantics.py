"""Counterexample propagation and conservative semantics, not discovery tests."""
import itertools
import json
from pathlib import Path
import random
import pytest
from connection_search.dsl import R, Runtime, at, fn, holes, infer, pair, plain, replace, seq, walk_typed
from connection_search.feasibility import AbstractRuntime, Feasibility
from connection_search.tasks import get_task
from connection_search.verify import Checker
from test_typed_choices import ChoiceBackend, setup


def analyzer(task='affine_prefix', **kwargs):
    task=get_task(task)
    return Feasibility(task.inputs,task.output,task.public()['public_examples'],**kwargs)


def poisoned():
    return ['accumulate',['fn',[['s',R],['e',R]],['mul','x0','s']],0,['hole',seq(R)]]


def concrete(expr, inputs, types, modulus):
    rt=Runtime(modulus)
    return plain(rt.eval(expr,{n:rt.wrap(inputs[n],t) for n,t in types.items()}))


def test_actual_uploaded_branch_is_refuted_at_call_six():
    records=json.loads((Path(__file__).parent/'fixtures/uploaded_bounded_states.json').read_text())
    f=analyzer()
    rejected=[(v,f.check(v['expression'])) for v in records if holes(v['expression'])]
    rejected=[(v,c) for v,c in rejected if c is not None]
    first,check=rejected[0]
    assert first['calls_so_far']==6
    assert check.status=='partial_output_counterexample'
    assert check.evidence['expected']==[0,0,1]
    assert check.evidence['abstract_output']['every_element']['possible_residues']==[0]
    assert first['original_check']['status']=='partial_type_checked'
    assert check.evaluated_cases==0


def test_zero_initial_state_is_an_invariant_despite_unknown_sequence():
    check=analyzer().check(poisoned())
    assert check is not None
    assert check.evidence['abstract_output']['length']=='unknown'
    assert not check.passed


def test_unknown_function_is_not_guessed_to_be_zero():
    expr=['accumulate',['hole',fn([R,R],R)],0,['hole',seq(R)]]
    assert analyzer().check(expr) is None


def test_higher_order_and_lexical_shadowing():
    typ={'x':R,'xs':seq(R)}
    expr=['apply',['fn',[['x',R]],['map',['fn',[['y',R]],['add','x','y']],'xs']],7]
    inputs={'x':3,'xs':[1,2,4]}
    rt=AbstractRuntime(5)
    values={n:rt.wrap(inputs[n],t) for n,t in typ.items()}
    abstract=rt.eval(expr,values,typ)
    assert rt.contains(abstract,concrete(expr,inputs,typ,5),seq(R))
    assert concrete(expr,inputs,typ,5)==[3,4,1]


def test_resource_exhaustion_and_large_modulus_are_unknown():
    f=analyzer(max_steps=1)
    assert f.check(poisoned()) is None
    assert f.unknown_limits==4
    f=Feasibility({'x':R},R,[{'inputs':{'x':0},'output':1,'modulus':101}])
    assert f.check(['mul',0,['hole',R]]) is None
    assert f.unknown_limits==1


def test_unknown_sequence_map_and_pair_projection():
    expr=['map',['fn',[['p',pair()]],['snd',['pair',['fst','p'],0]]],['hole',seq(pair())]]
    check=analyzer().check(expr)
    assert check and check.evidence['abstract_output']['every_element']['possible_residues']==[0]


def test_fixed_point_widens_instead_of_rejecting_reachable_values():
    rt=AbstractRuntime(17)
    expr=['accumulate',['fn',[['s',R],['e',R]],['add','s',1]],0,['hole',seq(R)]]
    abstract=rt.eval(expr,{}, {})
    assert rt.contains(abstract,[1,2,3,4,5,6,7,8,9],seq(R))
    assert abstract.element==frozenset(range(17))


def test_scan_uses_runtime_order_without_assuming_associativity():
    types={'xs':seq(R)};inputs={'xs':[1,2,3,4,5]}
    expr=['scan',['fn',[['a',R],['b',R]],['sub','a','b']],'xs']
    rt=AbstractRuntime(7)
    abstract=rt.eval(expr,{'xs':rt.wrap(inputs['xs'],seq(R))},types)
    assert rt.contains(abstract,concrete(expr,inputs,types,7),seq(R))


def test_scalar_differential_soundness():
    # 8,000 concrete executions; every one-hole abstraction must enclose them.
    types={'x':R,'y':R}
    leaves=[-1,0,1,'x','y']
    programs=leaves+[[op,a,b] for op in ('add','sub','mul') for a in leaves for b in leaves]
    executions=0
    for modulus in (2,3,5,7):
        for x,y in itertools.product(range(-2,3),repeat=2):
            inputs={'x':x,'y':y}
            for expression in programs:
                expected=concrete(expression,inputs,types,modulus)
                for path,_,_,t in walk_typed(expression,types):
                    partial=replace(expression,path,['hole',t])
                    rt=AbstractRuntime(modulus)
                    val=rt.eval(partial,{n:rt.wrap(v,R) for n,v in inputs.items()},types)
                    assert rt.contains(val,expected,R),(partial,inputs,modulus)
                executions+=1
    assert executions==8000


def test_composite_differential_soundness_under_hole_erasure():
    from connection_search.diagnostic import AFFINE_PROGRAM
    types={'as_':seq(R),'bs':seq(R),'x0':R}
    programs=[AFFINE_PROGRAM,
        ['zip','as_','bs'],
        ['reverse','bs'],
        ['map',['fn',[['v',R]],['mul','v','v']],'as_'],
        ['scan',['fn',[['a',R],['b',R]],['sub','a','b']],'as_'],
        ['accumulate',['fn',[['s',R],['e',R]],['add','s','e']],'x0','as_'],
        ['apply',['fn',[['z',seq(R)]],['map',['fn',[['x0',R]],['add','x0',1]],'z']],'bs'],
        ['fst',['pair','x0',['zip','as_','bs']]],
        ['apply',['fn',[['f',fn([R],R)]],['map','f','as_']],['fn',[['v',R]],['add','v','x0']]]]
    rng=random.Random(431)
    for modulus in (2,3,5,7):
        for n in range(5):
            inputs={'as_':[rng.randrange(-9,10) for _ in range(n)],
                    'bs':[rng.randrange(-9,10) for _ in range(n)],'x0':rng.randrange(-9,10)}
            for expression in programs:
                output_type=infer(expression,types)
                expected=concrete(expression,inputs,types,modulus)
                for path,_,_,t in walk_typed(expression,types):
                    partial=replace(expression,path,['hole',t])
                    rt=AbstractRuntime(modulus)
                    values={k:rt.wrap(v,types[k]) for k,v in inputs.items()}
                    val=rt.eval(partial,values,types)
                    assert rt.contains(val,expected,output_type),(partial,expected,modulus)


def test_all_hole_erased_known_affine_programs_survive():
    from connection_search.diagnostic import AFFINE_PROGRAM
    task=get_task('affine_prefix');f=analyzer()
    for path,_,_,typ in walk_typed(AFFINE_PROGRAM,task.inputs):
        assert f.check(replace(AFFINE_PROGRAM,path,['hole',typ])) is None
    assert Checker(task).check(AFFINE_PROGRAM).passed


def first(payload):
    return payload['legal_choices'][0]['id']


def test_refuted_partial_revises_fixed_syntax_not_another_hole(tmp_path):
    s=setup(tmp_path,ChoiceBackend(first),task='affine_prefix')
    state=s.submit(poisoned())
    assert state.check.status=='partial_output_counterexample'
    assert (3,) not in s.focus_paths(state)
    assert (2,) in s.focus_paths(state)  # A fixed constant is a revisable decision.
    assert s.select() is state
    s.ask_expansion(state)
    assert s.inference.backend.calls[-1]['stage']=='revise_decision'
    assert s.inference.backend.calls[-1]['feedback']['status']=='partial_output_counterexample'
    assert s.revision_count==1


def test_concrete_failure_gets_repair_turn_despite_unvisited_ancestors(tmp_path):
    s=setup(tmp_path,ChoiceBackend(first),task='affine_prefix',partial_pruning=False)
    root=s.submit(['hole',seq(R)])
    bad=s.submit(['accumulate',['fn',[['s',R],['e',R]],'s'],'x0','bs'],root)
    for i in range(16):
        s.submit(['accumulate',['fn',[['s',R],['e',R]],['mul',i,'s']],i,['hole',seq(R)]],root)
    assert s.select() is bad
    assert s.log.events[-1]['event']=='repair_turn'


def test_concrete_observation_reaches_other_branch_and_rollout(tmp_path):
    backend=ChoiceBackend(first)
    s=setup(tmp_path,backend,task='affine_prefix',partial_pruning=False,completion_steps=4)
    bad=s.submit('as_')
    root=s.submit(['hole',seq(R)])
    s.ask_expansion(root)
    assert backend.calls[-1]['search_observations'][0]['program']=='as_'
    # A new rollout on a different branch also sees the existing counterexample.
    shell=s.submit(['reverse',['hole',seq(R)]])
    bad.repair_turns=1  # The separate scheduling test covers pending repairs.
    s.rollouts(shell)
    assert backend.calls[-1]['search_observations'][0]['state']==bad.id
    assert 'not every shared component' in backend.calls[-1]['observation_scope']


def test_independent_frames_and_direct_control_do_not_receive_other_attempts(tmp_path):
    backend=ChoiceBackend(first)
    s=setup(tmp_path,backend,task='affine_prefix',max_calls=2)
    s.submit('as_')
    s.ask_expansion(s.submit(['hole',seq(R)]),independent=True)
    assert 'search_observations' not in backend.calls[0]
    s.direct_search()
    assert all('search_observations' not in p for p in backend.calls)


def test_no_revision_does_not_expand_a_refuted_partial_program(tmp_path):
    s=setup(tmp_path,ChoiceBackend(first),task='affine_prefix',revision=False)
    state=s.submit(poisoned())
    assert not s.eligible(state)
    before=s.inference.budget.calls
    s.rollouts(state)
    assert s.inference.budget.calls==before


def test_partial_pruning_ablation_and_concrete_accounting(tmp_path):
    s=setup(tmp_path,ChoiceBackend(first),task='affine_prefix',partial_pruning=False)
    assert s.submit(poisoned()).check.status=='partial_type_checked'
    assert s.feasibility.calls==0
    s.config.partial_pruning=True
    # A fresh, structurally distinct poisoned candidate.
    expr=poisoned();expr[1][2]=['mul',0,'s']
    assert s.submit(expr).check.status=='partial_output_counterexample'
    result=s.finish('graph')
    assert result['metrics']['case_evaluations']==0
    assert result['metrics']['partial_analysis']['refutations']==1
    assert result['status']=='not_solved'


def test_typed_inference_keeps_direction_and_output_contract(tmp_path):
    backend=ChoiceBackend(first);s=setup(tmp_path,backend)
    s.typed_step('test',{'instruction':'Preserve every fixed decision; use this counterexample.',
        'construction':['hole',R], 'focus':{'path':[],'scope':{'x':R,'y':R},'expected_type':R}})
    assert 'Preserve every fixed decision' in backend.calls[0]['instruction']
    assert 'Return its ID only' in backend.calls[0]['instruction']


def test_cli_and_report_expose_semantic_pruning(tmp_path):
    from connection_search.cli import parser,make_config
    from connection_search.report import write_outputs
    args=parser().parse_args(['run','--out',str(tmp_path),'--proposal-mode','typed','--no-partial-pruning'])
    assert make_config(args).partial_pruning is False
    s=setup(tmp_path,ChoiceBackend(first),task='affine_prefix')
    s.submit(poisoned())
    write_outputs(tmp_path,s,s.finish('graph'))
    text=(tmp_path/'report.html').read_text()
    assert 'Partial refutations: 1' in text
    assert 'Abstract checks do not count as executed task cases' in text


def test_waiting_repair_preempts_another_long_rollout(tmp_path):
    s=setup(tmp_path,ChoiceBackend(first),task='affine_prefix')
    bad=s.submit(poisoned())
    other=s.submit(['reverse',['hole',seq(R)]])
    s.initial_frame_ids=[other.id]
    assert s.select() is other
    child=s.ask_expansion(other)
    before=s.inference.budget.calls
    s.rollouts(child)
    assert s.inference.budget.calls==before
    assert s.select() is bad
    assert any(e['event']=='repair_preempts_rollout' for e in s.log.events)
