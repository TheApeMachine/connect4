"""Run from the project root. Synthetic controller diagnostics, not discovery."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tests'))
from test_typed_choices import ChoiceBackend,setup
from connection_search.dsl import holes
from connection_search.report import write_outputs


def main():
    out=Path(__file__).resolve().parent/'hostile_policy'
    out.mkdir(exist_ok=True)
    def choose(p):
        options=p['legal_choices']
        if p['stage']=='independent_frame':
            wanted='accumulate' if not backend.calls else 'reverse'
            return next(x['id'] for x in options if isinstance(x['term'],list) and x['term'][0]==wanted)
        growing=[x for x in options if holes(x['term'])]
        return (growing or options)[0]['id']
    backend=ChoiceBackend(choose)
    search=setup(out,backend,task='affine_prefix',completion_steps=32,max_calls=32)
    search.config.frames=2
    search.log.write('diagnostic_notice',synthetic=True,note='Hostile policy deliberately favors further construction. No model inference.')
    search.graph_search()
    result=search.finish('synthetic_liveness_diagnostic')
    write_outputs(out,search,result)
    print(json.dumps({'status':result['status'],'metrics':result['metrics']},indent=2))


if __name__=='__main__':
    main()
