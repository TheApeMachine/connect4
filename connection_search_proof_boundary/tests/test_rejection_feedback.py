"""Regressions for Danny's failed affine_prefix run.

Saved model replies are verbatim. Any corrected reply below is hand-authored
controller-test input, NOT evidence that Gemma corrected itself or discovered a proof.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest

from connection_search.dsl import Invalid, fn, pair, seq
from connection_search.inference import Budget, BudgetExhausted, Inference, Log, parse_reply
from connection_search.replay import ReplayBackend
from connection_search.report import write_outputs
from connection_search.search import Config, Search
from connection_search.tasks import get_task

FIXTURE=json.loads((Path(__file__).parent/'fixtures/uploaded_affine_failures.json').read_text())
RAW=FIXTURE['raw_replies']
P=pair()
# This is only the sequential baseline with legal syntax. It must STILL fail
# the original task's dependency-span bound; no affine-scan solution is supplied.
SEQUENTIAL=['accumulate',['fn',[['x','R'],['p',P]],
             ['add',['mul',['fst','p'],'x'],['snd','p']]],'x0',['zip','as_','bs']]


def make(tmp_path,replies,task='affine_prefix',max_calls=32,**config):
    backend=ReplayBackend(replies)
    log=Log(tmp_path/'events.jsonl')
    inference=Inference(backend,Budget(max_calls=max_calls,max_tokens=500000),log,max_new_tokens=4096)
    settings={'frames':1,'rollouts':0,'bridge_checks':0,**config}
    return Search(get_task(task),inference,log,Config(**settings)),backend


def test_original_trace_regression_exposes_new_feedback(tmp_path):
    def after_bad_name(payload):
        feedback=payload['rejection_feedback'][-1]
        assert feedback['message']=='Unbound variable: bs_'
        assert feedback['error_path']==[3,2]
        assert feedback['focus_path']==[]
        assert feedback['rejected_response']==RAW[3]
        assert feedback['evidence']['in_scope_names']==['as_','bs','x0']
        assert payload['feedback']['status']=='partial_type_checked'
        assert payload['focus']['scope']['bs']==seq('R')
        # Change ONLY the invalid name in this synthetic fixture. The hidden
        # callback-type error must surface; the controller must not fix it for us.
        term=parse_reply(RAW[3],'replacement')
        term[3][2]='bs'
        return {'replacement':term}
    def after_bad_callback(payload):
        f=payload['rejection_feedback'][-1]
        assert f['message']=='accumulate requires (state,element)->state'
        assert f['evidence']['expected_callback_type']==fn(['R',P],'R')
        assert f['evidence']['actual_callback_type']==fn(['R','R','R'],'R')
        return {'replacement':SEQUENTIAL}
    search,backend=make(tmp_path,[RAW[3],after_bad_name,after_bad_callback],max_calls=3)
    search.graph_search()
    result=search.finish('graph')
    assert result['status']=='not_solved'
    assert result['metrics']['case_evaluations']>0
    assert len(search.graph.nodes)>1
    assert any(s.check.status=='constraint_counterexample' for s in search.states.values())
    assert search.solution is None
    assert result['synthetic_backend'] and not result['model_inference_executed']


def test_full_uploaded_failure_replay_stops_before_spending_32_calls(tmp_path):
    search,backend=make(tmp_path,RAW,frames=3)
    search.graph_search()
    result=search.finish('graph')
    assert search.inference.budget.calls==7  # 3 independent + 4 adaptive non-progress attempts
    assert search.stop_reason=='proposal_stalled'
    assert len(search.graph.nodes)==1 and result['metrics']['case_evaluations']==0
    assert len(search.graph.edges)==0 and result['status']=='not_solved'
    assert len(backend.calls[3]['rejection_feedback'])==3
    assert all('rejection_feedback' not in p for p in backend.calls[:3])
    assert result['metrics']['stalled_focuses']==1


def test_noop_proposals_do_not_spend_repeated_completion_trials(tmp_path):
    search,backend=make(tmp_path,[{'replacement':['hole','R']}]*32,task='scalar_add',rollouts=1)
    search.graph_search()
    assert search.stop_reason=='proposal_stalled'
    assert len(backend.calls)==5 and search.rollout_count==0
    assert any(e['event']=='duplicate' for e in search.log.events)
    assert backend.calls[-1]['rejection_feedback'][-1]['status']=='no_progress'


def test_failed_focus_does_not_close_other_holes(tmp_path):
    search,backend=make(tmp_path,[{'replacement':'unknown'}, {'replacement':'y'}],task='scalar_add',max_stalled_attempts=1)
    state=search.submit(['add',['hole','R'],['hole','R']])
    search.rng.choice=lambda options: options[0]
    assert search.ask_expansion(state) is None
    assert state.stalled_attempts[(1,)]==1
    assert search.focus(state)[0]==(2,)
    child=search.ask_expansion(state)
    assert child is not None and child.expr==['add',['hole','R'],'y']
    assert 'rejection_feedback' not in backend.calls[1]  # unrelated focus, unrelated error


def test_successful_new_branch_resets_nonprogress_counter(tmp_path):
    search,backend=make(tmp_path,[{'replacement':'unknown'}, {'replacement':['add','x',['hole','R']]}],task='scalar_add')
    state=search.submit(['hole','R'])
    assert search.ask_expansion(state) is None
    assert state.stalled_attempts[()]==1
    assert search.ask_expansion(state) is not None
    assert state.stalled_attempts[()]==0


def test_local_error_path_is_not_confused_with_construction_path(tmp_path):
    search,backend=make(tmp_path,[{'replacement':['mul','unknown',1]}],task='scalar_add')
    state=search.submit(['add','x',['hole','R']])
    search.ask_expansion(state)
    error=state.rejections[-1]
    assert error['focus_path']==[2] and error['error_path']==[1]
    assert error['error_path_relative_to']=='replacement'
    assert state.expr==['add','x',['hole','R']]
    assert state.check.status=='partial_type_checked'  # rejection is NOT the accepted state's check


@pytest.mark.parametrize('first',['not JSON', '{"program":'])
def test_repair_control_receives_parse_error_even_without_previous_program(tmp_path,first):
    def correction(payload):
        assert payload['stage']=='repair'
        assert payload['rejection_feedback'][-1]['status']=='format_error'
        assert payload['rejection_feedback'][-1]['rejected_response']==first
        assert 'previous_program' not in payload
        return {'program':['add','x','y']}
    search,_=make(tmp_path,[first,correction],task='scalar_add',max_calls=2)
    search.direct_search(repair=True)
    assert search.solution is not None


def test_direct_control_remains_independent_after_invalid_output(tmp_path):
    search,backend=make(tmp_path,['not JSON',{'program':['add','x','y']}],task='scalar_add',max_calls=2)
    search.direct_search(repair=False)
    assert search.solution is not None
    assert all('rejection_feedback' not in p and 'previous_program' not in p for p in backend.calls)
    assert all(p['stage']=='direct' for p in backend.calls)


def test_completion_retry_receives_error_without_rewriting_prior_decisions(tmp_path):
    def correction(payload):
        assert payload['rejection_feedback'][-1]['status']=='format_error'
        assert payload['construction']==['add','x',['hole','R']]
        return {'program':['add','x','y']}
    search,_=make(tmp_path,['not JSON',correction],task='scalar_add',rollouts=2)
    state=search.submit(['add','x',['hole','R']])
    search.rollouts(state)
    assert state.completion_trials==2 and state.successful_completions==1
    assert search.solution is not None and search.solution.expr==['add','x','y']


def test_feedback_is_bounded_but_raw_generation_remains_in_log(tmp_path):
    raw='broken output '+('!'*3000)
    search,backend=make(tmp_path,[raw]*5,task='scalar_add',max_stalled_attempts=10)
    state=search.submit(['hole','R'])
    for _ in range(5):
        search.ask_expansion(state)
    feedback=backend.calls[-1]['rejection_feedback']
    assert len(feedback)==3 and all(len(r['rejected_response'])==2048 for r in feedback)
    assert all(r['response_truncated'] for r in feedback)
    assert all(e['raw']==raw for e in search.log.events if e['event']=='model_call')
    # All model calls, including diagnostics in later prompts, remain charged.
    calls=[e for e in search.log.events if e['event']=='model_call']
    assert search.inference.budget.tokens==sum(c['prompt_tokens']+c['generated_tokens'] for c in calls)
    assert calls[1]['prompt_tokens']>calls[0]['prompt_tokens']
    assert all(c['temperature']==.8 for c in calls)


def test_inference_does_not_reuse_stale_diagnostic_after_budget_failure(tmp_path):
    search,_=make(tmp_path,['not JSON'],task='scalar_add',max_calls=1)
    with pytest.raises(Invalid):
        search.inference.ask('direct',{'stage':'direct'},'program')
    assert search.inference.last_call['raw']=='not JSON'
    with pytest.raises(BudgetExhausted):
        search.inference.ask('direct',{'stage':'direct'},'program')
    assert search.inference.last_call is None


def test_report_exposes_raw_output_and_escapes_model_html(tmp_path):
    raw='</pre><script>alert("not executable")</script>'
    search,_=make(tmp_path,[raw],task='scalar_add',max_calls=1)
    search.graph_search()
    write_outputs(tmp_path,search,search.finish('graph'))
    report=(tmp_path/'report.html').read_text()
    assert '<script>' not in report and '&lt;script&gt;' in report
    assert 'Raw model response' in report and 'Exact system and user prompts' in report
    assert 'No task cases were evaluated.' in report
    saved=json.loads((tmp_path/'states.json').read_text())
    assert saved[0]['rejections'][0]['rejected_response']==raw


def test_invalid_stall_cap_fails_before_model_load(tmp_path,monkeypatch):
    import connection_search.cli as cli
    def forbidden(_):
        raise AssertionError('Model must not load')
    monkeypatch.setattr(cli,'load_backend',forbidden)
    assert cli.main(['run','--max-stalled-attempts','0','--out',str(tmp_path/'invalid')])==2


@pytest.mark.parametrize('index',[0,2,3,9])
def test_saved_bad_responses_are_never_silently_repaired(tmp_path,index):
    search,_=make(tmp_path,[RAW[index]],max_calls=1)
    search.graph_search()
    assert search.solution is None and len(search.graph.nodes)==1
    assert search.checker.case_evaluations==0
    rejection=next(e for e in search.log.events if e['event']=='rejected_transition')
    assert rejection['diagnostic']['rejected_response']==RAW[index]
