"""Reproduce local diagnostics. NO pretrained model is loaded by this script."""
from pathlib import Path
import copy
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from test_typed_choices import ChoiceBackend,target_policy,canonical,setup
from connection_search.diagnostic import AFFINE_PROGRAM
from connection_search.dsl import R,pair
from connection_search.inference import parse_reply,Log,Inference,Budget
from connection_search.replay import ReplayBackend
from connection_search.report import write_outputs
from connection_search.search import Config,Search
from connection_search.tasks import get_task
from connection_search.verify import Checker

out=ROOT/'validation/typed_choices'
fixture=json.loads((ROOT/'tests/fixtures/uploaded_affine_second_run.json').read_text())
task=get_task('affine_prefix')
call5=next(x for x in fixture['calls'] if x['call_number']==5)
original=parse_reply(call5['raw'],'replacement')
changed=copy.deepcopy(original)
changed[1][0]='fn'
checker=Checker(task)
audit={'source':fixture['source'],'source_sha256':fixture['source_sha256'],
       'original_call_number':5,'original_expression':original,
       'original_check':checker.check(original).to_dict(),
       'manual_diagnostic_edit':{'path':[1,0],'before':'Fn','after':'fn',
                               'production_automatic_repair':False},
       'edited_expression':changed,'edited_check':checker.check(changed).to_dict(),
       'interpretation':'Counterfactual diagnostic only. The edited output remains invalid under the original span bound.'}
(out/'uploaded_call5_audit.json').write_text(json.dumps(audit,indent=2)+'\n')

log=Log(out/'historical_json_replay/events.jsonl')
backend=ReplayBackend([(x['stage'],x['raw']) for x in fixture['calls']])
inf=Inference(backend,Budget(max_calls=32,max_tokens=1_000_000),log,max_new_tokens=8192)
search=Search(task,inf,log,Config())
search.graph_search()
summary=search.finish('historical_json_replay')
summary['diagnostic_note']='Historical replies replayed without alteration; synthetic prompt/token units, not a fresh Gemma run.'
assert summary['budget']['calls']==7 and summary['metrics']['case_evaluations']==0
write_outputs(out/'historical_json_replay',search,summary)

results={}
for name,expr in [('affine_constructibility',AFFINE_PROGRAM),('sequential_rejection',changed)]:
    target=canonical(expr,task.inputs)
    search=setup(out/name,ChoiceBackend(target_policy(target)),task='affine_prefix')
    child=search.ask_expansion(search.submit(['hole',task.output]),independent=True)
    search.rollouts(child)
    search.stop_reason='hand_authored_fixture_completed' if search.solution else 'hand_authored_fixture_rejected'
    summary=search.finish('typed_choice_fixture')
    summary['diagnostic_note']='HAND-AUTHORED target policy. Exercises grammar construction and checks; not learned model behavior or discovery.'
    summary['synthetic_token_units']='UTF-8 prompt bytes; character-ID choice tokens. Not pretrained tokenizer measurements.'
    write_outputs(out/name,search,summary)
    results[name]={k:summary[k] for k in ('status','stop_reason','metrics','budget','diagnostic_note')}
assert results['affine_constructibility']['status']=='passed_holdout_tests'
assert results['sequential_rejection']['status']=='not_solved'
(out/'diagnostic_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({name:{'status':r['status'],'calls':r['budget']['calls'],'cases':r['metrics']['case_evaluations']}
                 for name,r in results.items()},indent=2))
