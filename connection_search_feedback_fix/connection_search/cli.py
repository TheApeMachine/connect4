from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

from .backend import HuggingFaceBackend
from .dsl import Invalid
from .enumerate import EnumerationLimits,run_enumeration
from .inference import Budget,Inference,Log
from .library import Library
from .report import write_outputs
from .search import Config,Search
from .tasks import PROBES,TASKS,get_task
from .verify import Checker


def prepare(path):
    out=Path(path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing nonempty output directory: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    return out


def make_config(a):
    return Config(seed=a.seed,frames=a.frames,rollouts=a.rollouts,exploration=a.exploration,
                  max_states=a.max_states,bridge_checks=a.bridge_checks,backward=not a.no_backward,
                  revision=not a.no_revision,reuse=not a.no_reuse,max_checks=a.max_checks,
                  max_stalled_attempts=a.max_stalled_attempts)


def one_run(a,mode,out,backend,library,task_name=None):
    out=prepare(out)
    log=Log(out/'events.jsonl')
    config=make_config(a)
    task=get_task(task_name or a.task)
    inf=None
    if mode!='enum':
        inf=Inference(backend,Budget(a.max_calls,a.token_budget,a.model_seconds,a.wall_seconds),log,
                      a.seed,a.max_new_tokens,a.temperature)
    search=Search(task,inf,log,config,library)
    log.write('run_configuration',method=mode,task=task.public(a.seed),config=asdict(config),
              model=a.model if mode!='enum' else None,device=a.device if mode!='enum' else None,
              initial_library_fingerprint=library.fingerprint)
    enum_stats=None
    if mode=='graph':
        search.graph_search()
    elif mode in ('direct','repair'):
        search.direct_search(repair=mode=='repair')
    else:
        limits=EnumerationLimits(a.enum_max_size,a.enum_candidates,a.enum_bucket_cap,a.enum_expansions,a.enum_seconds)
        enum_stats=run_enumeration(search,limits)
    summary=search.finish(mode)
    summary['model']=a.model if mode!='enum' else None
    summary['device']=a.device if mode!='enum' else None
    if enum_stats:
        summary['enumeration']=enum_stats
    write_outputs(out,search,summary)
    print(f'{mode:7} {task.name:28} {summary["status"]:23} '
          f'calls={inf.budget.calls if inf else 0} cases={summary["metrics"]["case_evaluations"]} -> {out}/report.html',flush=True)
    return summary


def load_backend(a):
    return HuggingFaceBackend(a.model,a.device,a.dtype,a.loader,a.trust_remote_code)


def parser():
    p=argparse.ArgumentParser(description='Executable graph discovery: frozen local model + typed composition + independent checks')
    sub=p.add_subparsers(dest='command',required=True)
    def common(q):
        q.add_argument('--task',choices=sorted(TASKS),default='affine_prefix')
        q.add_argument('--model',default='google/gemma-4-12B-it')
        q.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto')
        q.add_argument('--dtype',choices=['auto','float16','bfloat16','float32'],default='auto')
        q.add_argument('--loader',choices=['auto','multimodal','causal'],default='auto')
        q.add_argument('--trust-remote-code',action='store_true')
        q.add_argument('--seed',type=int,default=1729)
        q.add_argument('--temperature',type=float,default=.8)
        q.add_argument('--max-calls',type=int,default=32)
        q.add_argument('--token-budget',type=int,default=65536,help='Hard combined prompt+generated token cap per method/task')
        q.add_argument('--max-new-tokens',type=int,default=1024)
        q.add_argument('--model-seconds',type=float,default=0,help='Synchronized model-call time cap, checked between calls; 0 disables')
        q.add_argument('--wall-seconds',type=float,default=0,help='Whole search time cap, checked between calls; 0 disables')
        q.add_argument('--frames',type=int,default=3)
        q.add_argument('--rollouts',type=int,default=1)
        q.add_argument('--exploration',type=float,default=.25)
        q.add_argument('--max-states',type=int,default=256)
        q.add_argument('--max-checks',type=int,default=4096)
        q.add_argument('--bridge-checks',type=int,default=2)
        q.add_argument('--max-stalled-attempts',type=int,default=4,
                       help='Close a graph state/focus after this many adaptive attempts without a new usable state; not a mathematical rejection')
        q.add_argument('--no-backward',action='store_true',help='Require hole-free replacements')
        q.add_argument('--no-revision',action='store_true',help='Do not revisit failed complete constructions')
        q.add_argument('--no-reuse',action='store_true',help='Disable cross-branch graph component bridging (starting library remains equal)')
        q.add_argument('--library-in',help='Explicit shared starting library; no automatic reuse from previous runs')
        q.add_argument('--enum-max-size',type=int,default=9)
        q.add_argument('--enum-candidates',type=int,default=2000)
        q.add_argument('--enum-bucket-cap',type=int,default=256)
        q.add_argument('--enum-expansions',type=int,default=50000)
        q.add_argument('--enum-seconds',type=float,default=20)
        q.add_argument('--out',required=True,help='New/empty directory; existing data is never overwritten')
    run=sub.add_parser('run');common(run)
    run.add_argument('--mode',choices=['graph','direct','repair','enum'],default='graph')
    compare=sub.add_parser('compare');common(compare)
    compare.add_argument('--modes',default='enum,direct,repair,graph')
    compare.add_argument('--tasks',help='Optional comma-separated task matrix; otherwise --task')
    compare.add_argument('--seeds',help='Optional comma-separated seed matrix; otherwise --seed')
    probe=sub.add_parser('probe');common(probe)
    audit=sub.add_parser('audit')
    audit.add_argument('--task',choices=sorted(TASKS),required=True)
    audit.add_argument('--program',required=True)
    audit.add_argument('--seed',type=int,default=1729)
    audit.add_argument('--out',required=True)
    demo=sub.add_parser('demo',help='Hand-authored controller regression, NEVER a model discovery run')
    demo.add_argument('--out',required=True)
    sub.add_parser('tasks')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    try:
        if a.command=='tasks':
            for task in TASKS.values():
                print(task.name+'\n  '+task.description)
            return 0
        if a.command=='audit':
            obj=json.loads(Path(a.program).read_text())
            expr=obj['program'] if isinstance(obj,dict) and 'program' in obj else obj
            check=Checker(get_task(a.task),a.seed)
            result={'search':check.check(expr).to_dict(),'holdout':check.check(expr,'holdout').to_dict(),
                    'universal_proof':False}
            out=prepare(a.out)
            (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2))
            return 0
        if a.command=='demo':
            from .diagnostic import run_demo
            run_demo(prepare(a.out))
            return 0
        # Fail invalid controller budgets before loading a large checkpoint.
        make_config(a)
        if a.max_new_tokens<1 or a.temperature<0:
            raise ValueError('Invalid generation settings')
        Budget(a.max_calls,a.token_budget,a.model_seconds,a.wall_seconds)
        EnumerationLimits(a.enum_max_size,a.enum_candidates,a.enum_bucket_cap,a.enum_expansions,a.enum_seconds)
        library=Library.load(a.library_in)
        if a.command=='run':
            backend=None if a.mode=='enum' else load_backend(a)
            one_run(a,a.mode,a.out,backend,library)
            return 0
        out=prepare(a.out)
        if a.command=='probe':
            backend=load_backend(a)
            results=[one_run(a,'direct',out/name,backend,library,name) for name in PROBES]
        else:
            modes=a.modes.split(',')
            if not modes or len(set(modes))!=len(modes) or not set(modes)<={'enum','direct','repair','graph'}:
                raise ValueError('modes must be unique enum,direct,repair,graph entries')
            names=a.tasks.split(',') if a.tasks else [a.task]
            seeds=[int(s) for s in a.seeds.split(',')] if a.seeds else [a.seed]
            if not names or not seeds or len(names)!=len(set(names)) or len(seeds)!=len(set(seeds)):
                raise ValueError('Task and seed matrix entries must be unique and nonempty')
            for name in names:
                get_task(name)
            backend=load_backend(a) if any(m!='enum' for m in modes) else None
            results=[]
            matrix=bool(a.tasks or a.seeds)
            for name in names:
                for seed in seeds:
                    run_args=copy.copy(a);run_args.task=name;run_args.seed=seed
                    order=list(modes)
                    # Shuffle order deterministically; equal, immutable starting-library snapshots.
                    random.Random(seed).shuffle(order)
                    for method in order:
                        dest=out/name/str(seed)/method if matrix else out/method
                        results.append(one_run(run_args,method,dest,backend,library))
        aggregate={}
        for result in results:
            row=aggregate.setdefault(result['method'],{'runs':0,'search_passes':0,'holdout_passes':0,
                    'model_seconds':0.0,'total_tokens':0,'end_to_end_seconds':0.0})
            row['runs']+=1
            row['search_passes']+=int(result['solution'] is not None)
            row['holdout_passes']+=int(result['status']=='passed_holdout_tests')
            row['model_seconds']+=(result.get('budget') or {}).get('model_seconds',0)
            row['total_tokens']+=(result.get('budget') or {}).get('total_tokens',0)
            row['end_to_end_seconds']+=result['metrics']['wall_seconds']
        report={'results':results,'aggregate':aggregate,'all_initial_libraries_equal':len({r['initial_library_fingerprint'] for r in results})==1,
                'model_loaded_once':backend is not None,'conclusion':'No aggregate superiority claim from a single task/seed.',
                'budget_note':'Token caps are exact; token counts are not FLOPs. Model-time caps can overshoot by one call. Compare observed synchronized time and end-to-end time; enumeration uses its own explicit CPU limits.'}
        (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        return 0
    except (ValueError,FileExistsError,RuntimeError,OSError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
