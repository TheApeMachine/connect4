"""Uploaded-trace regression and opt-in policy contracts, not model performance."""
import copy
import json
import random
from pathlib import Path
import pytest
from connection_search.cli import make_config, parser
from connection_search.completion import ClosingPlanner
from connection_search.dsl import (R, Invalid, at, digest, fn, holes, infer, key, pair,
                                   replace, seq, validate_shape, valid_type, walk_typed)
from connection_search.enumerate import Enumerator, EnumerationLimits
from connection_search.graph import Graph
from connection_search.library import Library
from connection_search.report import write_outputs
from connection_search.search import Config
from connection_search.tasks import get_task
from connection_search.typed import menu
from connection_search.verify import Check, Checker
from test_typed_choices import ChoiceBackend, setup, target_policy, canonical

FIXTURE=Path(__file__).parent/'fixtures/uploaded_proof_boundary_run.json'

def fixture():
    return json.loads(FIXTURE.read_text())


def test_recorded_planner_exception_does_not_reject_accepted_state():
    e=fixture()['boundary_expression']; env=get_task('affine_prefix').inputs
    before=copy.deepcopy(e)
    validate_shape(e)
    planner=ClosingPlanner(Library())
    for path,expr,local,typ in walk_typed(e,env):
        if isinstance(expr,list) and expr[0]=='hole':
            choices=menu(typ,local,Library(),construction=e,path=path)
            # This exact admissible() call raised Malformed hole type before.
            accepted=planner.admissible(choices,e,path,env,18)
            assert accepted
            for term in accepted.values():
                plan=planner.fill(replace(e,path,term),env)
                assert plan and plan.steps<=17
            break
    assert e==before


@pytest.mark.parametrize('depth', range(9))
def test_all_generated_prerequisites_respect_existing_type_depth(depth):
    t=R
    for _ in range(depth):
        t=fn([R],t)
    assert valid_type(t)
    grammar=Enumerator({'x':R},t,Library(),EnumerationLimits())
    for policy in ('unrestricted','existing'):
        terms=list(grammar.one_step(t,{'x':R},application_policy=policy))
        assert terms
        for term in terms:
            validate_shape(term)
            assert infer(term,{'x':R})==t
        for op,types in grammar.productions(t):
            assert all(valid_type(ty) for ty in types)


def test_bad_unselected_component_does_not_poison_planner():
    bad=['hole',['NoSuchType']]
    planner=ClosingPlanner(Library(),components=lambda t,e:[{'term':bad},{'term':'not_in_scope'}])
    assert planner.close(R,{}).steps==1


def test_invalid_library_alternative_is_skipped_before_map_witness():
    # A fake provider exposes a bad alternative; production Library validates
    # imports already. This tests the defensive per-alternative boundary.
    class BadProvider(Library):
        def offered(self,limit=8):
            return [{'id':'absent','type':seq(pair()),'term':['use','absent']}]
    planner=ClosingPlanner(BadProvider())
    plan=planner.fill(['hole',seq(pair())], {'xs':seq(R)})
    assert plan and infer(plan.expression,{'xs':seq(R)})==seq(pair())


def test_recorded_next_decision_is_available_without_fake_rejection(tmp_path):
    row=fixture()['calls'][13]
    e=fixture()['boundary_expression']
    s=setup(tmp_path,ChoiceBackend(lambda p:p['legal_choices'][0]['id']),task='affine_prefix')
    state=s.submit(e); s.inference.budget.calls=14
    path,expr,local,typ=next(x for x in walk_typed(e,s.task.inputs)
                            if isinstance(x[1],list) and x[1][0]=='hole')
    payload={'stage':'bounded_completion','construction':e,
             'focus':{'path':list(path),'expected_type':typ,'scope':local,'current_term':expr}}
    term=s.typed_step('bounded_completion',payload,planner=s.closing_planner,
                      remaining_steps=18,completion_env=s.task.inputs)
    assert term is not None and s.inference.budget.calls==15
    assert not any(x['event']=='rejected_transition' for x in s.log.events)
    assert state.check.status=='partial_type_checked'


def test_existing_policy_removes_unknown_heads_but_keeps_function_construction():
    t=fn([R,R],R); scope={'xs':seq(R)}; e=['hole',t]
    legacy=menu(t,scope,Library(),construction=e,path=())
    focused=menu(t,scope,Library(),construction=e,path=(),application_policy='existing')
    assert any(isinstance(x,list) and x[0]=='apply' for x in legacy.values())
    assert focused and all(x[0]=='fn' for x in focused.values())
    assert all(holes(x)==[(2,)] for x in focused.values())


@pytest.mark.parametrize('projected',[False,True])
def test_existing_scoped_unary_function_can_be_applied(projected):
    scope={'box':pair(fn([R],R),R)} if projected else {'f':fn([R],R)}
    head=['fst','box'] if projected else 'f'
    choices=menu(R,scope,Library(),construction=['hole',R],path=(),application_policy='existing')
    assert ['apply',head,['hole',R]] in choices.values()
    for term in choices.values():
        validate_shape(term);assert infer(term,scope)==R


def test_existing_library_callable_has_transparent_application():
    term=['fn',[['x',R]],['add','x',1]];typ=fn([R],R)
    lib=Library([{'id':'lib_'+digest(term),'type':typ,'term':term}])
    choices=menu(R,{},lib,construction=['hole',R],path=(),application_policy='existing')
    assert ['apply',term,['hole',R]] in choices.values()


def test_existing_graph_callables_are_scope_checked_and_hole_free():
    g=Graph();env={'k':R}
    term=['fn',[['x',R]],['add','x','k']]
    g.record(term,env,Check('partial_type_checked','fixture'))
    g.record(['fn',[['x',R]],['hole',R]],env,Check('partial_type_checked','fixture'))
    assert not g.available(None,{},functions_only=True)
    inventory=g.available(None,env,functions_only=True)
    assert len(inventory)==1
    choices=menu(R,env,Library(),construction=['hole',R],path=(),
                 application_policy='existing',callables=inventory)
    assert ['apply',term,['hole',R]] in choices.values()
    assert all(not holes(x['term']) for x in inventory)
    # Even a bogus inventory cannot smuggle an unavailable free variable.
    choices=menu(R,{},Library(),construction=['hole',R],path=(),
                 application_policy='existing',callables=inventory)
    assert ['apply',term,['hole',R]] not in choices.values()


def test_existing_planner_witnesses_use_actual_legal_steps():
    scope={'build':fn([R],seq(pair()))}
    planner=ClosingPlanner(Library(),application_policy='existing')
    plan=planner.fill(['hole',seq(pair())],scope)
    assert plan and plan.steps==2 and plan.expression[0]=='apply'
    # Every step of the chosen witness can actually be selected in this mode.
    expr=['hole',seq(pair())];steps=0
    while holes(expr):
        path,term,local,typ=next(x for x in walk_typed(expr,scope)
                                if isinstance(x[1],list) and x[1][0]=='hole')
        choices=menu(typ,local,Library(),construction=expr,path=path,application_policy='existing')
        payload={'focus':{'path':list(path),'scope':local},
                 'legal_choices':[{'id':k,'term':v} for k,v in choices.items()]}
        selected=target_policy(plan.expression)(payload)
        expr=replace(expr,path,choices[selected]);steps+=1
    assert expr==plan.expression and steps==plan.steps


def test_no_speculative_apply_is_added_during_random_typed_construction():
    rng=random.Random(37);scope=get_task('affine_prefix').inputs
    for _ in range(20):
        e=['hole',seq(R)];planner=ClosingPlanner(Library(),application_policy='existing')
        for remaining in range(16,0,-1):
            spots=[x for x in walk_typed(e,scope) if isinstance(x[1],list) and x[1][0]=='hole']
            if not spots:break
            path,current,local,typ=spots[0]
            choices=menu(typ,local,Library(),construction=e,path=path,application_policy='existing')
            choices=planner.admissible(choices,e,path,scope,remaining)
            assert choices
            term=rng.choice(list(choices.values()))
            assert not (isinstance(term,list) and term[0]=='apply')  # no supplied callables
            e=replace(e,path,term);validate_shape(e)
        assert not holes(e)


@pytest.mark.parametrize('mode',['direct','repair','graph'])
def test_policy_is_wired_into_every_typed_controller_and_report(tmp_path,mode):
    seen=[]
    def policy(p):
        seen.append(p)
        assert p['typed_application_policy']=='existing'
        for c in p['legal_choices']:
            e=c['term']
            if isinstance(e,list) and e[0]=='apply':
                assert not holes(e[1])
        return p['legal_choices'][0]['id']
    s=setup(tmp_path,ChoiceBackend(policy),max_calls=10,typed_application_policy='existing')
    if mode=='graph':s.graph_search()
    else:s.direct_search(repair=mode=='repair')
    report=s.finish(mode);write_outputs(tmp_path,s,report)
    assert seen and report['typed_application_policy']=='existing'
    assert 'Existing-callable proposal policy' in (tmp_path/'report.html').read_text()
    if mode!='graph':
        # No graph inventory is injected into direct/repair controls.
        assert all(not (isinstance(c['term'],list) and c['term'][0]=='apply')
                   for p in seen for c in p['legal_choices'])


def test_cli_policy_is_explicit_and_default_is_unchanged():
    a=parser().parse_args(['run','--out','x','--proposal-mode','typed',
                          '--typed-application-policy','existing'])
    assert make_config(a).typed_application_policy=='existing'
    b=parser().parse_args(['run','--out','x','--proposal-mode','typed'])
    assert make_config(b).typed_application_policy=='unrestricted'
    with pytest.raises(ValueError):Config(typed_application_policy='existing')


def test_existing_policy_does_not_rewrite_or_approve_old_failed_program():
    row=fixture()['checked'][-1];e=copy.deepcopy(row['expression'])
    result=Checker(get_task('affine_prefix')).check(e)
    assert result.status=='output_counterexample'
    assert result.evidence['actual']==[1,1,1]
    assert result.evidence['expected']==[0,0,1]
    assert e==row['expression']


def test_known_affine_path_is_still_reachable_by_synthetic_choices(tmp_path):
    from connection_search.diagnostic import AFFINE_PROGRAM
    target=canonical(AFFINE_PROGRAM,get_task('affine_prefix').inputs)
    s=setup(tmp_path,ChoiceBackend(target_policy(target)),task='affine_prefix',
             typed_application_policy='existing',max_calls=32)
    s.graph_search();summary=s.finish('graph')
    assert summary['status']=='passed_holdout_tests'
    assert summary['synthetic_backend'] and not summary['model_inference_executed']
