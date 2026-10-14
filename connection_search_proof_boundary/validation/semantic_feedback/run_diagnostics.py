#!/usr/bin/env python3
"""Offline historical-state audit and synthetic branching test. No model discovery."""
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from connection_search.dsl import holes, extends
from connection_search.feasibility import Feasibility
from connection_search.report import write_outputs
from connection_search.tasks import get_task
from test_typed_choices import ChoiceBackend,setup

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
records=json.loads((ROOT/'tests/fixtures/uploaded_bounded_states.json').read_text())
task=get_task('affine_prefix')
f=Feasibility(task.inputs,task.output,task.public()['public_examples'])
rows=[]
for record in records:
    if holes(record['expression']):
        check=f.check(record['expression'])
        if check:
            rows.append({'calls_so_far':record['calls_so_far'],'state':record['state'],
                         'expression':record['expression'],'new_check':check.to_dict(),
                         'original_check':record['original_check']})
(out/'historical_state_audit.json').write_text(json.dumps({
    'source':'Uploaded bounded run, events(3).jsonl / states(3).json',
    'kind':'Offline analysis of historical expressions, NOT a rerun of Gemma',
    'partial_states_analyzed':f.calls,'refuted_partial_states':len(rows),
    'first_refutation':rows[0],'refutations':rows,'analysis':f.metrics()},indent=2)+'\n')

# Preserve exactly the first six selected terms from the uploaded trace.
# Afterwards use a seeded random selection, never a mathematical target policy.
prefix=[
 ['accumulate',['hole',['Fn',['R','R'],'R']],['hole','R'],['hole',['Seq','R']]],
 ['accumulate',['hole',['Fn',['R',['Pair','R','R']],'R']],['hole','R'],['hole',['Seq',['Pair','R','R']]]],
 ['accumulate',['hole',['Fn',['R',['Pair','R','R']],'R']],['hole','R'],['hole',['Seq',['Pair','R','R']]]],
 ['fn',[['p3_0','R'],['p3_1','R']],['hole','R']],
 ['mul',['hole','R'],['hole','R']], 'x0']
for enabled in (False,True):
    target=out/('pruning_on' if enabled else 'pruning_off');target.mkdir(exist_ok=True)
    rng=random.Random(2718)
    def choose(p):
        i=len(backend.calls)
        if i<len(prefix):
            matches=[x for x in p['legal_choices'] if x['term']==prefix[i]]
            assert matches,('Historical prefix diverged',i,p['focus'])
            return matches[0]['id']
        return rng.choice(p['legal_choices'])['id']
    backend=ChoiceBackend(choose)
    search=setup(target,backend,task='affine_prefix',max_calls=32,
                 completion_steps=32,partial_pruning=enabled)
    search.config.frames=3;search.config.bridge_checks=2
    search.graph_search();summary=search.finish('graph')
    summary['diagnostic']='Recorded first six choices, then seeded random legal policy; no trained model. Synthetic tokens are bytes, not Gemma tokens.'
    write_outputs(target,search,summary)
    print(target.name,summary['status'],summary['metrics'])
