"""Hand-authored regression examples. Imported only by the explicit demo command.
These fixtures are NOT part of any real model prompt or production proposal policy.
"""
from __future__ import annotations
import json
from pathlib import Path
from .dsl import R,fn,pair,seq
from .inference import Budget,Inference,Log
from .replay import ReplayBackend
from .report import write_outputs
from .search import Config,Search
from .tasks import get_task

P=pair()
AFFINE_COMBINE=['fn',[['l',P],['r',P]],
                ['pair',['mul',['fst','r'],['fst','l']],
                 ['add',['mul',['fst','r'],['snd','l']],['snd','r']]]]
AFFINE_APPLY=['fn',[['p',P]],['add',['mul',['fst','p'],'x0'],['snd','p']]]
AFFINE_SEQUENCE=['scan',AFFINE_COMBINE,['zip','as_','bs']]
AFFINE_PROGRAM=['map',AFFINE_APPLY,AFFINE_SEQUENCE]
BAD_COMBINE=['fn',[['l',P],['r',P]],['pair',['add',['fst','l'],['fst','r']],['sub',['snd','l'],['snd','r']]]]
BAD_PROGRAM=['map',AFFINE_APPLY,['scan',BAD_COMBINE,['zip','as_','bs']]]


def run_demo(out:Path):
    log=Log(out/'events.jsonl')
    backend=ReplayBackend([
        ('independent_frame',{'replacement':BAD_PROGRAM}),
        ('revise_decision',{'replacement':AFFINE_COMBINE}),
    ])
    inf=Inference(backend,Budget(max_calls=2,max_tokens=200000),log,max_new_tokens=4096)
    config=Config(frames=1,rollouts=0,exploration=1.0,bridge_checks=0)
    search=Search(get_task('affine_prefix'),inf,log,config)
    search.graph_search()
    summary=search.finish('graph_regression_replay')
    if summary['status']!='passed_holdout_tests' or search.revision_count!=1:
        raise AssertionError('Controller regression did not exercise the required failure/revision/success path')
    summary['diagnostic_note']='Hand-authored algebra fixture; proves controller behavior only. No model weights were loaded.'
    summary['synthetic_token_units']='UTF-8 bytes, only for deterministic budget-contract testing; not tokenizer measurements'
    write_outputs(out,search,summary)
    print('SYNTHETIC CONTROLLER REGRESSION, not a model discovery result.')
    print('Non-associative connection rejected with a numeric witness; focused replacement passed independent search and held-out cases.')
    print(str(out/'report.html'))
    return summary
