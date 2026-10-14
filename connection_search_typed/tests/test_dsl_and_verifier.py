from __future__ import annotations
import copy
import random
import pytest
from connection_search.dsl import *
from connection_search.algebra import prove_associative,scan_laws,Poly
from connection_search.tasks import get_task
from connection_search.verify import Checker
from connection_search.diagnostic import AFFINE_PROGRAM,AFFINE_COMBINE,AFFINE_APPLY,BAD_COMBINE,P

ADD=['fn',[['a',R],['b',R]],['add','a','b']]
MUL=['fn',[['a',R],['b',R]],['mul','a','b']]
SQUARE=['fn',[['v',R]],['mul','v','v']]
SUM=['scan',ADD,'xs']

@pytest.mark.parametrize('expr,env,typ',[
    (3,{},R),(['pair',1,2],{},pair()),(SUM,{'xs':seq(R)},seq(R)),
    (['hole',seq(pair())],{},seq(pair())),
    (AFFINE_PROGRAM,get_task('affine_prefix').inputs,seq(R)),
])
def test_typing(expr,env,typ):
    validate_shape(expr)
    assert infer(expr,env)==typ

@pytest.mark.parametrize('expr',[
    True,1.5,{},['add',1],['add',1,2,3],['eval','x'],['hole','NoType'],
    ['fn',[['x',R],['x',R]],'x'],['fn',[],0],['hole',['Fn',[],R]],2**129,
    ['fn',[['__x','bogus']],'__x'],['map',1,'xs']
])
def test_malformed_is_rejected(expr):
    c=Checker(get_task('prefix_sum')).check(expr)
    assert c.status in ('type_error','resource_limit')
    assert not c.passed


def test_lambda_scope_and_holes():
    e=['map',['fn',[['p',pair()]],['hole',R]],['zip','as_','bs']]
    info={p:(env,t) for p,_,env,t in walk_typed(e,get_task('affine_prefix').inputs)}
    assert info[(1,2)][0]['p']==pair()
    assert holes(e)==[(1,2)]
    assert free_vars(e)=={'as_','bs'}
    assert extends(e,replace(e,(1,2),['fst','p']))
    assert not extends(e,['map',['fn',[['q',pair()]],['fst','q']],['zip','as_','bs']])
    with pytest.raises(Invalid):
        at(e,(1,1)) # Parameter metadata is not an editable expression child.

@pytest.mark.parametrize('operation',[ADD,MUL,AFFINE_COMBINE])
def test_symbolic_associativity(operation):
    carrier=R if operation in (ADD,MUL) else pair()
    proof=prove_associative(operation,{},carrier)
    assert proof['status']=='proved_polynomial_identity'


def test_nonassociativity_has_replayable_witness():
    with pytest.raises(Invalid) as err:
        prove_associative(BAD_COMBINE,{},pair())
    assert err.value.kind=='law_counterexample'
    witness=err.value.evidence
    rt=Runtime(witness['modulus']);f=rt.eval(BAD_COMBINE,{})
    a,b,c=[rt.wrap(v,pair()) for v in witness['triple']]
    assert plain(rt.call(f,[rt.call(f,[a,b]),c]))==tuple(witness['left'])
    assert witness['left']!=witness['right']


def test_negative_proof_is_not_goal_test():
    bad=['scan',['fn',[['a',R],['b',R]],['sub','a','b']],'xs']
    c=Checker(get_task('prefix_sum')).check(bad)
    assert c.status=='law_counterexample'
    assert c.path==(1,)
    assert c.evaluated_cases==0


def test_unsupported_symbolic_operator_is_unknown_not_proved():
    # Carries a scalar through a sequence-valued closure; valid type, unsupported proof language.
    f=['fn',[['a',R],['b',R]],['apply',['fn',[['s',seq(R)]],['add','a','b']],'xs']]
    with pytest.raises(Invalid) as err:
        prove_associative(f,{'xs':seq(R)},R)
    assert err.value.kind=='law_unknown'


def test_partial_program_has_no_goal_evidence():
    c=Checker(get_task('prefix_sum')).check(['scan',['hole',fn([R,R],R)],'xs'])
    assert c.status=='partial_type_checked'
    assert c.evaluated_cases==0 and not c.passed

@pytest.mark.parametrize('name,expr',[
    ('prefix_sum',SUM),('prefix_product',['scan',MUL,'xs']),
    ('prefix_square_sum',['scan',ADD,['map',SQUARE,'xs']]),
    ('affine_prefix',AFFINE_PROGRAM),
    ('map_square',['map',SQUARE,'xs']),
    ('pair_energy',['add',['mul','x','x'],['mul','y','y']]),
])
def test_known_diagnostics_pass_independent_search_and_holdout(name,expr):
    checker=Checker(get_task(name))
    for split in ('search','holdout'):
        c=checker.check(expr,split)
        assert c.passed,c.to_dict()
        assert c.passed_cases==(28 if split=='search' else 64)


def test_affine_perturbed_precondition_does_not_silently_pass():
    c=Checker(get_task('affine_square_coefficient')).check(AFFINE_PROGRAM)
    assert c.status=='output_counterexample'


def test_affine_order_is_not_commutative():
    backward=copy.deepcopy(AFFINE_COMBINE)
    def swap(e):
        if e=='l':return 'r'
        if e=='r':return 'l'
        return [swap(v) for v in e] if isinstance(e,list) else e
    backward[2]=swap(backward[2])
    assert prove_associative(backward,{},pair())['status']=='proved_polynomial_identity'
    wrong=['map',AFFINE_APPLY,['scan',backward,['zip','as_','bs']]]
    assert Checker(get_task('affine_prefix')).check(wrong).status=='output_counterexample'


def test_sequential_recurrence_correct_but_rejected_for_span():
    body=['add',['mul',['fst','p'],'state'],['snd','p']]
    sequential=['accumulate',['fn',[['state',R],['p',pair()]],body],'x0',['zip','as_','bs']]
    c=Checker(get_task('affine_prefix')).check(sequential)
    assert c.status=='constraint_counterexample'
    assert c.evidence['output_correct']
    assert c.evidence['span']>c.evidence['maximum_span']


def test_scan_matches_sequential_under_noncommutative_composition():
    rng=random.Random(23)
    for modulus in (2,4,12,97,101):
        for n in (0,1,2,3,7,16,33,129):
            pairs=[(rng.randrange(modulus),rng.randrange(modulus)) for _ in range(n)]
            rt=Runtime(modulus)
            data=rt.wrap(pairs,seq(pair()))
            actual=plain(rt.eval(['scan',AFFINE_COMBINE,'ps'],{'ps':data}))
            expected=[]
            for a,b in pairs:
                if not expected:
                    expected.append((a,b))
                else:
                    u,v=expected[-1]
                    expected.append((a*u%modulus,(a*v+b)%modulus))
            assert actual==expected


def test_zip_mismatch_fails_explicit_precondition():
    rt=Runtime(5)
    with pytest.raises(Invalid) as err:
        rt.eval(['zip','a','b'],{'a':rt.wrap([1],seq(R)),'b':rt.wrap([],seq(R))})
    assert err.value.kind=='precondition_failed'


def test_runtime_resource_limit():
    rt=Runtime(5,step_limit=2)
    with pytest.raises(Limit):
        rt.eval(['mul',['add',1,2],3],{})


def test_task_specification_is_not_mutated_by_bad_program():
    task=get_task('affine_prefix');before=task.fingerprint
    Checker(task).check(['hole','R'])
    assert task.fingerprint==before
    p=task.public()
    assert 'holdout' not in key(p) and 'reference' not in p
    assert set(c.modulus for c in task.cases('holdout'))-set(c.modulus for c in task.cases('search'))


def test_exact_poly_normalization_not_float_equality():
    x=Poly.variable('x');y=Poly.variable('y')
    left=x.combine(y,'add').combine(x.combine(y,'sub'),'mul')
    right=x.combine(x,'mul').combine(y.combine(y,'mul'),'sub')
    assert left==right
