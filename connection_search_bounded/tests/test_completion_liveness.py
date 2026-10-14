"""Liveness tests use hostile/random policies, not hand-authored math answers."""
import json
import random
from pathlib import Path
import pytest
from connection_search.completion import ClosingPlanner
from connection_search.dsl import R, at, extends, fn, holes, infer, key, pair, replace, seq, size, walk_typed
from connection_search.inference import BudgetExhausted
from connection_search.library import Library
from connection_search.tasks import TASKS
from connection_search.typed import menu
from test_typed_choices import ChoiceBackend, setup, target_policy


@pytest.mark.parametrize('task_name', sorted(TASKS))
def test_closure_witness_is_reachable_in_its_advertised_steps(task_name):
    task=TASKS[task_name]
    planner=ClosingPlanner(Library())
    plan=planner.fill(['hole',task.output],task.inputs)
    assert plan is not None
    partial=['hole',task.output]
    policy=target_policy(plan.expression)
    for i in range(plan.steps):
        if not holes(partial):
            break
        path,term,scope,typ=next(x for x in walk_typed(partial,task.inputs)
                                if isinstance(x[1],list) and x[1][0]=='hole')
        choices=menu(typ,scope,Library(),construction=partial,path=path)
        allowed=planner.admissible(choices,partial,path,task.inputs,plan.steps-i)
        label=policy({'focus':{'path':path,'scope':scope},
                      'legal_choices':[{'id':k,'term':v} for k,v in allowed.items()]})
        partial=replace(partial,path,allowed[label])
    assert not holes(partial)
    assert infer(partial,task.inputs)==task.output


def test_horizon_accounts_for_all_holes_not_just_the_focused_one():
    env={'x':R,'y':R};partial=['add',['hole',R],['hole',R]]
    choices=menu(R,env,Library(),construction=partial,path=(1,))
    planner=ClosingPlanner(Library())
    allowed=planner.admissible(choices,partial,(1,),env,2)
    assert allowed and all(not holes(t) for t in allowed.values())
    # With four decisions, a binary operation at this first hole can fit.
    larger=planner.admissible(choices,partial,(1,),env,4)
    assert ['add',['hole',R],['hole',R]] in larger.values()


def test_horizon_does_not_select_a_correct_answer():
    env={'x':R,'y':R}
    choices=menu(R,env,Library(),construction=['hole',R],path=())
    allowed=ClosingPlanner(Library()).admissible(choices,['hole',R],(),env,1)
    assert set(v for v in allowed.values() if not isinstance(v,list))=={-1,0,1,'x','y'}


def test_unknown_inhabitation_is_not_called_impossible():
    assert ClosingPlanner(Library()).close(seq(R),{}) is None


def test_shape_limit_applies_to_hypothetical_complete_witness():
    partial=['hole',R]
    # Legal partials are not necessarily completable within the DSL size limit.
    for _ in range(128):
        partial=['add',0,partial]
    assert ClosingPlanner(Library()).fill(partial,{}) is None


def test_bloat_seeking_policy_must_finish_in_its_allocated_horizon(tmp_path):
    def bloat(p):
        options=p['legal_choices']
        adds=[x for x in options if x['term']==['add',['hole',R],['hole',R]]]
        if adds:
            return adds[0]['id']
        return next((x['id'] for x in options if x['term']==0), options[0]['id'])
    s=setup(tmp_path,ChoiceBackend(bloat),completion_steps=10,max_calls=16)
    child=s.ask_expansion(s.submit(['hole',R]),independent=True)
    s.rollouts(child)
    assert s.rollout_count<=10
    assert s.full_candidates==1
    assert s.checker.case_evaluations>0
    assert child.completion_trials==1 and child.censored_completions==0
    assert not s.solution
    assert s.finish('fixture')['metrics']['completion_horizon_filtered_choices']>0


@pytest.mark.parametrize('seed',range(20))
def test_random_legal_completion_reaches_a_checker_outcome(tmp_path,seed):
    rng=random.Random(seed)
    backend=ChoiceBackend(lambda p:rng.choice(p['legal_choices'])['id'])
    s=setup(tmp_path,backend,task='affine_prefix',completion_steps=12,max_calls=16)
    # This is a generic prefix fold shell, not the affine solution.
    shell=['accumulate',['hole',fn([R,R],R)],['hole',R],['hole',seq(R)]]
    child=s.submit(shell)
    s.rollouts(child)
    assert s.rollout_count<=12
    outcomes=[e for e in s.log.events if e['event']=='completion_evidence']
    assert len(outcomes)==1
    # A completed unsupported law may be inconclusive; never call it success.
    assert outcomes[0]['outcome'] in ('failure','censored')
    assert s.checker.case_evaluations>0 or any(e['event']=='rejected_completion' and
        e['status'] in ('law_counterexample','law_unknown') for e in s.log.events)
    assert child.completion_trials+child.censored_completions==1


def test_initial_branch_turns_and_rollout_budget_share(tmp_path):
    def choose(p):
        options=p['legal_choices']
        # Distinct independent shells; thereafter deliberately prefer expansion.
        if p['stage']=='independent_frame':
            wanted='accumulate' if not backend.calls else 'reverse'
            return next(x['id'] for x in options if isinstance(x['term'],list) and x['term'][0]==wanted)
        constructors=[x for x in options if holes(x['term'])]
        return (constructors or options)[0]['id']
    backend=ChoiceBackend(choose)
    s=setup(tmp_path,backend,task='affine_prefix',completion_steps=32,max_calls=24)
    s.config.frames=2
    s.graph_search()
    assert len(s.initial_frame_ids)==2
    assert all(s.states[eid].visits>=1 for eid in s.initial_frame_ids)
    allocations=[e for e in s.log.events if e['event']=='rollout_allocation']
    first=allocations[0]
    assert first['pending_initial_branches']==1
    assert first['allocated_calls']<=first['remaining_calls']//2
    assert s.checker.case_evaluations>0


def test_token_interruption_persists_work_and_does_not_record_failure(tmp_path):
    backend=ChoiceBackend(lambda p:next((x['id'] for x in p['legal_choices'] if x['term']=='x'),p['legal_choices'][0]['id']))
    s=setup(tmp_path,backend,max_calls=10)
    state=s.submit(['add',['hole',R],['hole',R]])
    real=backend.generate_choice
    def interrupt_after_one(**kwargs):
        result=real(**kwargs)
        # After charging this call, the next preflight has no output allowance.
        s.inference.budget.max_tokens=result.prompt_tokens+result.generated_tokens
        return result
    backend.generate_choice=interrupt_after_one
    with pytest.raises(BudgetExhausted):
        s.rollouts(state)
    assert len(s.states)==2 and holes(list(s.states.values())[-1].expr)
    assert state.completion_trials==0 and state.successful_completions==0
    assert state.censored_completions==1 and s.rollout_count==1
    evidence=[e for e in s.log.events if e['event']=='completion_evidence'][-1]
    assert evidence['outcome']=='censored' and evidence['reason']=='total_token_cap'


def test_direct_insufficient_horizon_stops_without_zero_call_retry_loop(tmp_path):
    s=setup(tmp_path,ChoiceBackend(lambda p:p['legal_choices'][0]['id']),completion_steps=1)
    # A function output with no matching input requires at least two selections.
    # Exercise the exact preflight shared by direct/repair, without modifying Task.
    with pytest.raises(BudgetExhausted,match='typed_completion_horizon_unavailable'):
        s.complete_typed({'stage':'direct'},['hole',fn([R],R)],{'x':R})
    assert s.inference.budget.calls==0


def test_partial_program_keeps_every_preexisting_choice(tmp_path):
    rng=random.Random(741)
    s=setup(tmp_path,ChoiceBackend(lambda p:rng.choice(p['legal_choices'])['id']),completion_steps=8)
    shell=['add','x',['hole',R]]
    state=s.submit(shell)
    s.rollouts(state)
    assert all(extends(shell,x.expr) for x in s.states.values())


def test_uploaded_rollout_hits_a_concrete_horizon_boundary_before_exhaustion():
    fixture=json.loads((Path(__file__).parent/'fixtures/uploaded_affine_typed_run.json').read_text())
    planner=ClosingPlanner(Library())
    rollout=[c for c in fixture['calls'] if c['stage']=='bounded_completion']
    assert len(rollout)==28
    rejected=[]
    for step,c in enumerate(rollout):
        after=replace(c['construction'],tuple(c['path']),c['selected_term'])
        plan=planner.fill(after,fixture['task_inputs'])
        if plan is None or plan.steps>14-step-1:
            rejected.append(c['call_number'])
            break
    assert rejected==[12]
    # This deliberately does not invent model responses after the first changed menu.
