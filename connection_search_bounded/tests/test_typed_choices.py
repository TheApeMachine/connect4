"""Constrained-interface regressions. Hand-authored policies are not discovery."""
from __future__ import annotations
import copy
import json
import types
from pathlib import Path
import pytest
from connection_search.backend import Generation,HuggingFaceBackend
from connection_search.constrained import ChoiceTrie,generate_choice
from connection_search.dsl import *
from connection_search.enumerate import Enumerator,EnumerationLimits
from connection_search.inference import Budget,BudgetExhausted,Inference,Log
from connection_search.library import Library
from connection_search.search import Config,Search
from connection_search.tasks import TASKS,get_task
from connection_search.typed import menu
from connection_search.verify import Checker


class CharacterTokenizer:
    """Synthetic tokenizer units, not a claim about a pretrained vocabulary."""
    eos_token_id=0
    pad_token_id=1
    bos_token_id=2
    all_special_ids=[0,1,2]
    def encode(self,text,add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c)+3 for c in text]
    def decode(self,ids,skip_special_tokens=False):
        return ''.join(chr(int(i)-3) if int(i)>2 else ('' if skip_special_tokens else '<special>') for i in ids)


class ChoiceBackend:
    synthetic=True
    tokenizer=CharacterTokenizer()
    def __init__(self,policy):
        self.policy=policy
        self.calls=[]
    def count_prompt_tokens(self,system,user):
        return len((system+user).encode('utf8'))
    def generate(self,**kwargs):
        raise AssertionError('Typed runs must not silently use unconstrained generation')
    def generate_choice(self,*,system,user,seed,max_new_tokens,temperature,labels):
        payload=json.loads(user)
        label=self.policy(payload)
        self.calls.append(payload)
        return Generation(label,self.count_prompt_tokens(system,user),len(self.tokenizer.encode(label))+1,0.0)


def setup(tmp_path,backend,task='scalar_add',**kwargs):
    log=Log(tmp_path/'events.jsonl')
    b=Budget(max_calls=kwargs.pop('max_calls',64),max_tokens=kwargs.pop('max_tokens',1_000_000))
    inf=Inference(backend,b,log,max_new_tokens=kwargs.pop('max_new_tokens',32))
    cfg=Config(proposal_mode='typed',frames=1,rollouts=1,bridge_checks=0,**kwargs)
    return Search(get_task(task),inf,log,cfg)


def canonical(expr,env,renames=None):
    renames=renames or {}
    if isinstance(expr,str):
        return renames.get(expr,expr)
    if not isinstance(expr,list) or expr[0]=='hole':
        return expr
    if expr[0]=='fn':
        params=expr[1]
        n=len(env)
        names=[f'p{n}_{i}' for i in range(len(params))]
        while any(name in env for name in names):
            n+=1;names=[f'p{n}_{i}' for i in range(len(params))]
        new=[[name,p[1]] for name,p in zip(names,params)]
        return ['fn',new,canonical(expr[2],{**env,**dict(new)},
                                  {**renames,**dict(zip([p[0] for p in params],names))})]
    return [expr[0],*[canonical(c,env,renames) for c in expr[1:]]]


def target_policy(target):
    def select(payload):
        goal=at(target,tuple(payload['focus']['path']))
        def fits(term):
            if not extends(term,goal):
                return False
            try:
                return all(infer(at(goal,p),env)==t for p,e,env,t in walk_typed(term,payload['focus']['scope'])
                           if isinstance(e,list) and e[0]=='hole')
            except Invalid:
                return False
        compatible=[x for x in payload['legal_choices'] if fits(x['term'])]
        assert compatible, ('Target not reachable from this grammar menu',goal,payload['legal_choices'])
        return min(compatible,key=lambda x:(len(holes(x['term'])),-size(x['term'])))['id']
    return select


def test_trie_masks_every_prefix_and_forces_eos():
    tok=CharacterTokenizer();labels=['C001','C002','C010']
    trie=ChoiceTrie(tok,labels)
    for path,label in trie.paths.items():
        for i in range(len(path)):
            allowed=trie.allowed(path[:i],tok.eos_token_id)
            assert path[i] in allowed
            assert ord('F')+3 not in allowed
            assert 0 not in allowed
        assert trie.allowed(path,0)==[0]
        assert trie.selected(path+(0,),0)==label
    with pytest.raises(RuntimeError):
        trie.allowed([999],0)
    with pytest.raises(RuntimeError):
        trie.selected(next(iter(trie.paths)),0)


def test_trie_prefix_labels_can_end_or_continue():
    trie=ChoiceTrie(CharacterTokenizer(),['C0','C01'])
    path=tuple(CharacterTokenizer().encode('C0'))
    assert set(trie.allowed(path,0))=={0,ord('1')+3}


@pytest.mark.parametrize('labels',[[],['C0','C0']])
def test_invalid_choice_menus_fail(labels):
    with pytest.raises(ValueError):
        ChoiceTrie(CharacterTokenizer(),labels)


def test_lossy_tokenizer_is_rejected():
    tok=CharacterTokenizer()
    tok.decode=lambda *a,**k:'different'
    with pytest.raises(ValueError,match='round-trip'):
        ChoiceTrie(tok,['C000'])


@pytest.mark.parametrize('task_name',sorted(TASKS))
def test_every_offered_step_is_well_typed_and_scope_correct(task_name):
    t=TASKS[task_name]
    choices=menu(t.output,t.inputs,Library(),construction=['hole',t.output],path=())
    assert choices
    for expr in choices.values():
        validate_shape(expr)
        assert infer(expr,t.inputs)==t.output
        assert 'bs_' not in key(expr)


def test_callbacks_have_exact_arity_without_model_spelling():
    P=pair();env={'as_':seq(R),'bs':seq(R),'x0':R}
    grammar=Enumerator(env,seq(R),Library(),EnumerationLimits())
    accum=[t for t in grammar.one_step(seq(R),env) if isinstance(t,list) and t[0]=='accumulate']
    assert len(accum)==2
    assert any(e[1]==['hole',fn([R,P],R)] for e in accum)
    for term in accum:
        cb_type=term[1][1]
        callback=next(x for x in grammar.one_step(cb_type,env) if isinstance(x,list) and x[0]=='fn')
        assert len(callback[1])==2
        assert infer(callback,env)==cb_type


def test_bound_pair_projections_and_capture_avoidance():
    env={'p3_0':R,'p':pair(pair(),R),'x0':R}
    choices=menu(R,env,Library(),construction=['hole',R],path=())
    assert ['fst',['fst','p']] in choices.values()
    callbacks=menu(fn([R],R),env,Library(),construction=['hole',fn([R],R)],path=())
    callback=next(x for x in callbacks.values() if isinstance(x,list) and x[0]=='fn')
    assert callback[1][0][0]!='p3_0'
    local={**env,**dict(callback[1])}
    assert infer(callback,env)==fn([R],R)
    assert len(local)==len(env)+1


def test_menu_excludes_previous_constructions_not_all_uses_of_operator():
    env={'x':R,'y':R};hole=['hole',R]
    first=menu(R,env,Library(),construction=hole,path=())
    add=next(e for e in first.values() if isinstance(e,list) and e[0]=='add')
    after=menu(R,env,Library(),construction=hole,path=(),excluded=[key(add)])
    assert add not in after.values()
    # Reusing addition INSIDE the candidate remains legal.
    inside=menu(R,env,Library(),construction=add,path=(1,),excluded=[key(add)])
    assert any(isinstance(e,list) and e[0]=='add' for e in inside.values())


def test_typed_interface_emits_real_step_and_charges_exact_tokens(tmp_path):
    backend=ChoiceBackend(lambda p:next(x['id'] for x in p['legal_choices'] if x['term']==['add',['hole',R],['hole',R]]))
    s=setup(tmp_path,backend)
    root=s.submit(['hole',R])
    child=s.ask_expansion(root,independent=True)
    assert child.check.status=='partial_type_checked'
    assert child.expr==['add',['hole',R],['hole',R]]
    assert len(s.graph.edges)>0
    assert s.inference.budget.calls==1
    assert s.inference.budget.generated_tokens==5
    assert s.inference.last_call['decoding']=='typed_choice_trie'
    assert s.inference.last_call['max_new_tokens']==5
    assert 'Return exactly the requested JSON object.' not in s.inference.last_call['system']


def test_no_unconstrained_fallback_for_invalid_choice_backend(tmp_path):
    s=setup(tmp_path,ChoiceBackend(lambda p:'Fn'))
    with pytest.raises(RuntimeError,match='escaped'):
        s.ask_expansion(s.submit(['hole',R]))
    assert s.inference.budget.calls==1


def test_insufficient_token_budget_does_not_call_model(tmp_path):
    s=setup(tmp_path,ChoiceBackend(lambda p:p['legal_choices'][0]['id']),max_tokens=8)
    with pytest.raises(BudgetExhausted):
        s.ask_expansion(s.submit(['hole',R]))
    assert s.inference.budget.calls==0


def test_small_output_cap_does_not_remove_long_labels(tmp_path):
    s=setup(tmp_path,ChoiceBackend(lambda p:p['legal_choices'][0]['id']),max_new_tokens=4)
    with pytest.raises(BudgetExhausted):
        s.ask_expansion(s.submit(['hole',R]))
    assert s.inference.budget.calls==0


def test_graph_rollout_builds_affine_through_checkable_steps(tmp_path):
    from connection_search.diagnostic import AFFINE_PROGRAM
    target=canonical(AFFINE_PROGRAM,get_task('affine_prefix').inputs)
    s=setup(tmp_path,ChoiceBackend(target_policy(target)),task='affine_prefix')
    child=s.ask_expansion(s.submit(['hole',s.task.output]),independent=True)
    s.rollouts(child)
    result=s.finish('typed_fixture')
    assert result['status']=='passed_holdout_tests'
    assert s.solution.expr==target
    assert s.inference.budget.calls>10
    assert s.rollout_count==s.inference.budget.calls-1
    assert result['metrics']['constrained_choice_calls']==s.inference.budget.calls
    assert result['synthetic_backend'] and not result['model_inference_executed']
    assert all(x.check.status not in ('type_error','resource_limit') for x in s.states.values())
    assert child.completion_trials==1 and child.successful_completions==1
    assert result['search_check']['passed_cases']==28
    assert result['holdout_check']['passed_cases']==64


def test_sequential_math_still_fails_parallel_constraint(tmp_path):
    expr=['accumulate',['fn',[['s',R],['p',pair()]],['add',['mul',['fst','p'],'s'],['snd','p']]],'x0',['zip','as_','bs']]
    task=get_task('affine_prefix');target=canonical(expr,task.inputs)
    s=setup(tmp_path,ChoiceBackend(target_policy(target)),task='affine_prefix')
    child=s.ask_expansion(s.submit(['hole',task.output]),independent=True)
    s.rollouts(child)
    assert s.solution is None
    check=next(x.check for x in s.states.values() if x.expr==target)
    assert check.status=='constraint_counterexample' and check.evidence['output_correct']
    assert check.evaluated_cases>0


def test_insufficient_call_horizon_preserves_state_without_spending(tmp_path):
    from connection_search.diagnostic import AFFINE_PROGRAM
    target=canonical(AFFINE_PROGRAM,get_task('affine_prefix').inputs)
    s=setup(tmp_path,ChoiceBackend(target_policy(target)),task='affine_prefix',max_calls=3)
    child=s.ask_expansion(s.submit(['hole',s.task.output]),independent=True)
    # The old implementation spent both remaining calls then interrupted. The
    # new horizon check defers this trial before spending either call.
    s.rollouts(child)
    assert len(s.states)==2
    assert s.inference.budget.calls==1
    assert s.rollout_count==0 and child.completion_trials==0
    assert any(e['event']=='completion_deferred' for e in s.log.events)
    assert all(holes(x.expr) for x in s.states.values())


def test_typed_direct_and_repair_share_the_same_interface(tmp_path):
    for repair in (False,True):
        s=setup(tmp_path/str(repair),ChoiceBackend(target_policy(['add','x','y'])),max_calls=3)
        s.direct_search(repair=repair)
        result=s.finish('repair' if repair else 'direct')
        assert result['status']=='passed_holdout_tests'
        assert result['metrics']['constrained_choice_calls']==3
        assert not any(e['event']=='forward_bridge' for e in s.log.events)


def test_transformers_call_contract_and_illegal_logit_mask():
    """Exercises our HF call boundary with a deterministic stand-in, NOT HF itself."""
    torch=pytest.importorskip('torch')
    b=HuggingFaceBackend.__new__(HuggingFaceBackend)
    b.torch=torch;b.device=torch.device('cpu');b.tokenizer=CharacterTokenizer()
    b.tfm=types.SimpleNamespace(GenerationConfig=lambda **kw:types.SimpleNamespace(**kw))
    b._encode=lambda *a:{'input_ids':torch.tensor([[2,6,7]])}
    calls=[]
    class Model:
        config=types.SimpleNamespace(is_encoder_decoder=False)
        generation_config=types.SimpleNamespace(eos_token_id=[0,1])
        def generate(self,**kwargs):
            calls.append(kwargs)
            c=kwargs['generation_config']
            ids=kwargs['input_ids']
            for _ in range(c.max_new_tokens):
                allowed=kwargs['prefix_allowed_tokens_fn'](0,ids[0])
                logits=torch.zeros(256)
                logits[ord('F')+3]=1000  # strongest raw logit is deliberately illegal
                mask=torch.full_like(logits,float('-inf'))
                mask[allowed]=0
                nxt=int((logits+mask).argmax())
                ids=torch.cat([ids,torch.tensor([[nxt]])],dim=1)
                if nxt==c.eos_token_id:
                    break
            return ids
    b.model=Model()
    for seed in (1,2):
        gen=generate_choice(b,system='s',user='u',seed=seed,max_new_tokens=5,temperature=0,labels=['C001','C003'])
        assert gen.text=='C001' and gen.prompt_tokens==3 and gen.generated_tokens==5
    assert all('past_key_values' not in c for c in calls)
    assert all(c['generation_config'].forced_eos_token_id is None for c in calls)


def test_real_transformers_generate_with_random_weights_when_installed():
    """Integration only: random tiny GPT-2, no pretrained capability claim/download."""
    tfm=pytest.importorskip('transformers')
    torch=pytest.importorskip('torch')
    torch.manual_seed(1)
    b=HuggingFaceBackend.__new__(HuggingFaceBackend)
    b.torch=torch;b.tfm=tfm;b.device=torch.device('cpu');b.tokenizer=CharacterTokenizer()
    b.model=tfm.GPT2LMHeadModel(tfm.GPT2Config(vocab_size=256,n_positions=32,n_embd=16,n_layer=1,n_head=1,eos_token_id=0,bos_token_id=2))
    b.model.eval()
    b._encode=lambda *a:{'input_ids':torch.tensor([[2,6,7]]),'attention_mask':torch.ones((1,3),dtype=torch.long)}
    gen=generate_choice(b,system='s',user='u',seed=1,max_new_tokens=5,temperature=.8,labels=['C001','C003'])
    assert gen.text in ['C001','C003'] and gen.generated_tokens==5


def test_structural_horizon_defers_without_changing_prior_steps(tmp_path):
    from connection_search.diagnostic import AFFINE_PROGRAM
    target=canonical(AFFINE_PROGRAM,get_task('affine_prefix').inputs)
    s=setup(tmp_path,ChoiceBackend(target_policy(target)),task='affine_prefix',completion_steps=2)
    child=s.ask_expansion(s.submit(['hole',s.task.output]),independent=True)
    original=copy.deepcopy(child.expr)
    s.rollouts(child)
    assert s.solution is None and len(s.states)==2
    assert child.expr==original
    assert s.inference.budget.calls==1
    assert any(e['event']=='completion_unavailable' and e['status']=='completion_limit' for e in s.log.events)
    assert not child.rejections  # Controller budget limits are not model errors.


def test_cli_typed_flag_is_wired_not_merely_recorded(tmp_path,monkeypatch):
    from connection_search import cli
    backend=ChoiceBackend(target_policy(['add','x','y']))
    monkeypatch.setattr(cli,'load_backend',lambda a:backend)
    out=tmp_path/'run'
    assert cli.main(['run','--mode','direct','--proposal-mode','typed','--task','scalar_add',
                     '--max-calls','3','--token-budget','100000','--out',str(out)])==0
    result=json.loads((out/'summary.json').read_text())
    assert result['status']=='passed_holdout_tests'
    assert result['metrics']['constrained_choice_calls']==3
    assert 'Typed-choice decoding' in (out/'report.html').read_text()
    assert result['synthetic_backend']


def test_bad_completion_configuration_rejected_before_loading(tmp_path,monkeypatch):
    from connection_search import cli
    def forbidden(a):
        raise AssertionError('Do not load weights for invalid configuration')
    monkeypatch.setattr(cli,'load_backend',forbidden)
    assert cli.main(['run','--proposal-mode','typed','--completion-steps','0','--out',str(tmp_path/'bad')])==2


def test_input_fingerprint_and_constraints_unchanged_in_typed_run(tmp_path):
    task=get_task('scalar_add');before=task.fingerprint
    s=setup(tmp_path,ChoiceBackend(target_policy(['add','x','y'])),max_calls=3)
    s.direct_search()
    assert task.fingerprint==before
    assert s.finish('direct')['task_fingerprint']==before


def test_random_legal_walks_do_not_introduce_type_or_scope_errors():
    import random
    rng=random.Random(51)
    task=get_task('affine_prefix')
    for run in range(10):
        expression=['hole',task.output]
        for step in range(15):
            spots=[(p,e,env,t) for p,e,env,t in walk_typed(expression,task.inputs)
                   if isinstance(e,list) and e[0]=='hole']
            if not spots:
                break
            path,_,env,t=rng.choice(spots)
            choices=menu(t,env,Library(),construction=expression,path=path,seed=run*100+step)
            if not choices:
                break
            expression=replace(expression,path,rng.choice(list(choices.values())))
            validate_shape(expression)
            assert infer(expression,task.inputs)==task.output
