#!/usr/bin/env python3
"""Offline uploaded-run audit. Never runs a language model or submits a witness."""
from __future__ import annotations
import argparse, collections, copy, json, pathlib, sys, traceback

def main():
    p=argparse.ArgumentParser();p.add_argument('--package-root');p.add_argument('--out',required=True)
    a=p.parse_args()
    bundle=pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(pathlib.Path(a.package_root).resolve() if a.package_root else bundle))
    from connection_search.dsl import holes, infer, replace, validate_shape, walk_typed
    from connection_search.completion import ClosingPlanner
    from connection_search.feasibility import Feasibility
    from connection_search.graph import Graph
    from connection_search.library import Library
    from connection_search.tasks import get_task
    from connection_search.typed import menu
    from connection_search.verify import Check, Checker
    data=json.loads((bundle/'tests/fixtures/uploaded_proof_boundary_run.json').read_text())
    task=get_task('affine_prefix');lib=Library();e=data['boundary_expression']
    original=copy.deepcopy(e);validate_shape(e)
    path,cur,local,t=next(x for x in walk_typed(e,task.inputs)
                        if isinstance(x[1],list) and x[1][0]=='hole')
    choices=menu(t,local,lib,construction=e,path=path)
    boundary={'accepted_expression_shape_valid':True,'accepted_type':infer(e,task.inputs),
              'next_focus_path':list(path),'valid_alternatives_before_horizon':len(choices)}
    try:
        planner=ClosingPlanner(lib)
        accepted=planner.admissible(choices,e,path,task.inputs,18)
        boundary.update(outcome='continued',offered_after_horizon=len(accepted),
                        closure_steps=[planner.fill(replace(e,path,x),task.inputs).steps for x in accepted.values()])
    except Exception as exc:
        boundary.update(outcome='exception',error_type=type(exc).__name__,error=str(exc),
                        error_path=list(getattr(exc,'path',())),traceback=traceback.format_exc())
    boundary['accepted_expression_unchanged']=e==original
    checker=Checker(task)
    feasibility=Feasibility(task.inputs,task.output,task.public(1729)['public_examples'])
    rows=[]
    def stable(value):
        if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if k!='seconds'}
        if isinstance(value,(list,tuple)):return [stable(v) for v in value]
        return value
    for row in data['checked']:
        result=checker.check(row['expression'])
        if result.status=='partial_type_checked':
            abstract=feasibility.check(row['expression'])
            if abstract:
                abstract.certificates=result.certificates;result=abstract
        rows.append({'state':row['state'],'before':row['check']['status'],'after':result.status,
                     'unchanged_except_timing':stable(row['check'])==stable(result.to_dict())})
    report={'synthetic_model_run':False,'language_model_executed':False,
            'source':'tests/fixtures/uploaded_proof_boundary_run.json, derived from events(5).jsonl',
            'boundary':boundary,'checked_states':rows,
            'unchanged_states':sum(x['unchanged_except_timing'] for x in rows),
            'total_states':len(rows),
            'last_recorded_check':checker.check(data['checked'][-1]['expression']).to_dict()}
    try:
        row=data['calls'][8]
        g=Graph()
        for checked in data['checked']:
            if checked['calls_so_far']>=9:break
            g.record(checked['expression'],task.inputs,Check(**checked['check']))
        components=g.available(row['expected_type'],row['scope'])
        filtered=menu(row['expected_type'],row['scope'],lib,construction=row['construction'],
                      path=tuple(row['path']),components=components,application_policy='existing',
                      callables=g.available(None,row['scope'],functions_only=True))
        report['existing_policy_call9']={'old_selected_term':row['selected_term'],
            'old_selection_offered':row['selected_term'] in filtered.values(),
            'offered_terms':list(filtered.values()),'new_model_choice_known':False}
    except TypeError:
        report['existing_policy_call9']='Not available in baseline package'
    pathlib.Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checked_states','last_recorded_check')},indent=2))

if __name__=='__main__':main()
