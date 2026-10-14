from __future__ import annotations
from dataclasses import asdict
import copy
import json
from pathlib import Path
import pytest
from connection_search.dsl import *
from connection_search.diagnostic import *
from connection_search.inference import Budget,BudgetExhausted,Inference,Log,parse_reply
from connection_search.library import Library
from connection_search.search import Config,Search
from connection_search.replay import ReplayBackend
from connection_search.tasks import get_task
from connection_search.verify import Checker
from connection_search.enumerate import EnumerationLimits,run_enumeration


def make(tmp_path,replies,task='affine_prefix',**kwargs):
    backend=ReplayBackend(replies)
    log=Log(tmp_path/'events.jsonl')
    budget=Budget(max_calls=kwargs.pop('max_calls',8),max_tokens=kwargs.pop('max_tokens',500000))
    inf=Inference(backend,budget,log,max_new_tokens=4096)
    config=Config(frames=1,rollouts=kwargs.pop('rollouts',0),exploration=1.0,bridge_checks=kwargs.pop('bridge_checks',0),**kwargs)
    return Search(get_task(task),inf,log,config),backend


def test_failure_revises_only_offending_function(tmp_path):
    s,b=make(tmp_path,[{'replacement':BAD_PROGRAM},{'replacement':AFFINE_COMBINE}],max_calls=2)
    s.graph_search();result=s.finish('graph')
    assert result['status']=='passed_holdout_tests'
    assert s.revision_count==1 and s.inference.budget.calls==2
    assert b.calls[1]['focus']['path']==[2,1]
    assert s.solution.expr==AFFINE_PROGRAM
    assert s.graph.revisions[-1]['replaced_path']==[2,1]
    # One shared zip subexpression and one shared mapper, not recreated mutable copies.
    assert sum(n['expression']==['zip','as_','bs'] for n in s.graph.nodes.values())==1
    assert sum(n['expression']==AFFINE_APPLY for n in s.graph.nodes.values())==1
    assert any(len(e['tails'])==2 for e in s.graph.edges.values())
    assert not result['model_inference_executed']
    assert result['synthetic_backend']


def test_actual_partial_completion_rollout(tmp_path):
    partial=['map',['hole',fn([P],R)],['hole',seq(P)]]
    def fill_one(payload):
        t=payload['focus']['expected_type']
        return {'replacement':AFFINE_APPLY if t==fn([P],R) else AFFINE_SEQUENCE}
    s,b=make(tmp_path,[('independent_frame',{'replacement':partial}),('expand_hole',fill_one),
                        ('bounded_completion',{'program':AFFINE_PROGRAM})],rollouts=1,max_calls=3)
    s.graph_search()
    assert s.solution is not None
    assert s.rollout_count==1 and sum(x.completion_trials for x in s.states.values())==1
    assert sum(x.successful_completions for x in s.states.values())==1
    assert len(b.calls)==3


def test_completion_cannot_rewrite_prior_decisions(tmp_path):
    partial=['map',AFFINE_APPLY,['hole',seq(P)]]
    wrong=copy.deepcopy(AFFINE_PROGRAM);wrong[1]=['fn',[['p',P]],['fst','p']]
    s,b=make(tmp_path,[{'program':wrong}],rollouts=1,max_calls=1)
    state=s.submit(partial)
    s.rollouts(state)
    assert s.solution is None and state.completion_trials==1
    assert any(e['event']=='rejected_completion' for e in s.log.events)


def test_rollout_budget_failure_does_not_create_phantom_trial(tmp_path):
    s,b=make(tmp_path,[],rollouts=1,max_tokens=8)
    state=s.submit(['hole',seq(R)])
    with pytest.raises(BudgetExhausted):
        s.rollouts(state)
    assert state.completion_trials==0 and s.rollout_count==0


def test_forward_frontier_actually_composes_across_branches(tmp_path):
    s,b=make(tmp_path,[],bridge_checks=16)
    # A failed full candidate nonetheless contains a proved associative transform sequence.
    wrong=['map',['fn',[['p',P]],['fst','p']],AFFINE_SEQUENCE]
    s.submit(wrong)
    state=s.submit(['map',AFFINE_APPLY,['hole',seq(P)]])
    s.bridges(state)
    assert s.solution is not None and s.solution.expr==AFFINE_PROGRAM
    assert s.reuse_hits>0 and s.inference.budget.calls==0


def test_reuse_ablation_does_not_bridge(tmp_path):
    s,b=make(tmp_path,[],bridge_checks=16,reuse=False)
    s.submit(['map',['fn',[['p',P]],['fst','p']],AFFINE_SEQUENCE])
    s.bridges(s.submit(['map',AFFINE_APPLY,['hole',seq(P)]]))
    assert s.solution is None and s.reuse_hits==0


def test_independent_frames_do_not_receive_other_frames(tmp_path):
    s,b=make(tmp_path,[{'replacement':'xs'},{'replacement':'xs'}],task='prefix_sum',max_calls=2)
    s.config.frames=2
    s.graph_search()
    assert all(c['stage']=='independent_frame' for c in b.calls)
    assert all('feedback' not in c and 'available_components' not in c for c in b.calls)
    assert b.calls[0]['construction']==b.calls[1]['construction']


def test_direct_and_repair_controls_receive_correct_context(tmp_path):
    for repair in (False,True):
        p=tmp_path/str(repair)
        s,b=make(p,[{'program':'x'},{'program':['add','x','y']}],task='scalar_add',max_calls=2)
        s.direct_search(repair)
        assert s.solution is not None
        assert ('checker_feedback' in b.calls[1])==repair
        assert ('previous_program' in b.calls[1])==repair


def test_fresh_calls_contain_only_deliberately_supplied_state(tmp_path):
    s,b=make(tmp_path,[{'program':'x'},{'program':['add','x','y']}],task='scalar_add',max_calls=2)
    s.direct_search(False)
    calls=[x for x in s.log.events if x['event']=='model_call']
    assert len(calls)==2
    assert 'previous_program' not in calls[1]['user']
    assert calls[0]['seed']!=calls[1]['seed']


def test_hard_token_reservation_includes_prompt_and_invalid_output(tmp_path):
    s,b=make(tmp_path,['not valid json'],task='scalar_add',max_calls=1)
    s.direct_search()
    budget=s.inference.budget
    assert budget.calls==1 and budget.generated_tokens==len('not valid json')
    assert budget.prompt_tokens>0
    assert budget.tokens<=budget.max_tokens
    assert any(e['event']=='rejected_format' for e in s.log.events)


def test_oversized_prompt_never_calls_backend(tmp_path):
    s,b=make(tmp_path,[],task='scalar_add',max_tokens=10)
    s.direct_search()
    assert s.stop_reason=='insufficient_tokens_for_prompt_and_output'
    assert not b.calls and s.inference.budget.calls==0

@pytest.mark.parametrize('text',[
    'prefix {"program":["add","x","y"]}',
    '{"constraints":{},"program":1}',
    '{"program":1} {"program":2}',
    '{"unfinished":{"program":1}',
    '{"program":NaN}',
])
def test_strict_root_parser_does_not_salvage_nested_output(text):
    with pytest.raises(Invalid):
        result=parse_reply(text,'program')
        validate_shape(result)


def test_library_roundtrip_transparent_reuse(tmp_path):
    lib=Library();task=get_task('affine_prefix')
    check=Checker(task).check(AFFINE_PROGRAM)
    ids=lib.harvest(AFFINE_PROGRAM,task.inputs,{'task':task.name},check)
    assert ids
    lib.save(tmp_path/'library.json');loaded=Library.load(tmp_path/'library.json')
    entry=next(v for v in loaded.entries.values() if v['term'][2]==AFFINE_PROGRAM)
    names=[p[0] for p in entry['term'][1]]
    reused=loaded.resolve(['apply',['use',entry['id']],*names])
    assert Checker(task).check(reused).passed
    assert loaded.fingerprint==lib.fingerprint
    bad=copy.deepcopy(loaded.to_dict()['entries']);bad[0]['term'][-1]=0
    with pytest.raises(Invalid):
        Library(bad)


def test_library_cannot_claim_unsupported_macro_proof():
    with pytest.raises(ValueError):
        Library().harvest(['hole',R],{},'x',Checker(get_task('scalar_add')).check('x'))


def test_audit_is_after_selection_and_never_in_prompts(tmp_path):
    s,b=make(tmp_path,[{'program':['add','x','y']}],task='scalar_add',max_calls=1)
    s.direct_search();s.finish('direct')
    events=[x['event'] for x in s.log.events]
    assert events.index('selection_frozen')<events.index('holdout_audit')
    assert not any('holdout' in json.dumps(payload) for payload in b.calls)
    assert next(e for e in s.log.events if e['event']=='holdout_audit')['feedback_to_search'] is False


def test_model_free_enumerator_really_synthesizes_not_fixture_lookup(tmp_path):
    log=Log(tmp_path/'events.jsonl')
    s=Search(get_task('scalar_add'),None,log,Config())
    stats=run_enumeration(s,EnumerationLimits(max_size=3,max_candidates=100,bucket_cap=128,max_expansions=10000,seconds=5))
    assert s.solution and s.solution.expr==['add','x','y']
    assert stats['expansions']>0
    assert s.finish('enum')['status']=='passed_holdout_tests'


def test_complete_bad_program_preserves_typed_component_status(tmp_path):
    s,b=make(tmp_path,[])
    bad=s.submit(BAD_PROGRAM)
    assert bad.check.status=='law_counterexample'
    nodes=[n for n in s.graph.nodes.values() if n['expression']==BAD_COMBINE]
    assert nodes and nodes[0]['status']=='type_checked'
    assert 'associativity' not in nodes[0].get('properties',{})


def test_enumerator_algebraic_pruning_reaches_nontrivial_composition(tmp_path):
    log=Log(tmp_path/'events.jsonl')
    s=Search(get_task('pair_energy'),None,log,Config())
    stats=run_enumeration(s,EnumerationLimits(max_size=7,max_candidates=2000,bucket_cap=256,max_expansions=50000,seconds=8))
    assert s.solution is not None
    assert stats['semantic_duplicates_pruned']>0
    assert s.finish('enum')['status']=='passed_holdout_tests'
