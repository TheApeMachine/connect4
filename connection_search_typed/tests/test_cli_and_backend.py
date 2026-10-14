from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import types
import pytest
from connection_search.cli import main
from connection_search.backend import HuggingFaceBackend

ROOT=Path(__file__).resolve().parents[1]


def test_supplied_backend_classes_preserved_verbatim():
    text=(ROOT/'connection_search/backend.py').read_text()
    provenance=json.loads((ROOT/'PROVENANCE.json').read_text())
    for node in ast.parse(text).body:
        if isinstance(node,ast.ClassDef) and node.name in provenance['preserved_classes']:
            start=min([node.lineno]+[d.lineno for d in node.decorator_list])
            source='\n'.join(text.splitlines()[start-1:node.end_lineno])
            assert hashlib.sha256(source.encode()).hexdigest()==provenance['class_source_sha256'][node.name]


def test_cli_enum_matrix_without_model_dependency(tmp_path):
    out=tmp_path/'matrix'
    assert main(['compare','--modes','enum','--tasks','scalar_add,scalar_mul','--seeds','1,2',
                 '--enum-max-size','3','--enum-seconds','5','--out',str(out)])==0
    report=json.loads((out/'comparison.json').read_text())
    assert len(report['results'])==4
    assert report['all_initial_libraries_equal']
    assert all(r['status']=='passed_holdout_tests' for r in report['results'])
    assert all(not r['model_inference_executed'] for r in report['results'])


def test_cli_preserves_existing_output(tmp_path):
    out=tmp_path/'keep';out.mkdir();(out/'sentinel').write_text('unchanged')
    assert main(['run','--mode','enum','--task','scalar_add','--out',str(out)])==2
    assert (out/'sentinel').read_text()=='unchanged'


def test_cli_invalid_budget_rejected_without_loading(tmp_path,monkeypatch):
    import connection_search.cli as cli
    def forbidden(_):
        raise AssertionError('Must validate budget before loading weights')
    monkeypatch.setattr(cli,'load_backend',forbidden)
    assert main(['run','--max-calls','0','--out',str(tmp_path/'x')])==2


def test_cli_demo_is_labeled_synthetic(tmp_path):
    out=tmp_path/'demo'
    assert main(['demo','--out',str(out)])==0
    report=json.loads((out/'summary.json').read_text())
    assert report['status']=='passed_holdout_tests'
    assert report['synthetic_backend'] and not report['model_inference_executed']
    assert 'SYNTHETIC' in (out/'report.html').read_text()
    assert (out/'graph.dot').exists() and (out/'solution.json').exists()


def test_cli_audit_reexecutes_program(tmp_path):
    program=tmp_path/'p.json';program.write_text(json.dumps({'program':['add','x','y']}))
    out=tmp_path/'audit'
    assert main(['audit','--task','scalar_add','--program',str(program),'--out',str(out)])==0
    report=json.loads((out/'audit.json').read_text())
    assert report['holdout']['status']=='passed_holdout_tests' and not report['universal_proof']


def test_original_generate_path_has_no_cross_call_cache(monkeypatch):
    torch=pytest.importorskip('torch')
    backend=HuggingFaceBackend.__new__(HuggingFaceBackend)
    backend.torch=torch;backend.device=torch.device('cpu')
    backend.tokenizer=types.SimpleNamespace(eos_token_id=2,pad_token_id=1,
                                          decode=lambda ids,**_:','.join(str(x.item()) for x in ids))
    encoded=[]
    def encode(system,user):
        encoded.append((system,user))
        return {'input_ids':torch.tensor([[3,4,5]]),'attention_mask':torch.tensor([[1,1,1]])}
    backend._encode=encode
    calls=[]
    class Model:
        def generate(self,**kwargs):
            calls.append(kwargs)
            return torch.tensor([[3,4,5,9,2]])
    backend.model=Model()
    first=backend.generate(system='s1',user='u1',seed=1,max_new_tokens=5,temperature=.8)
    second=backend.generate(system='s2',user='u2',seed=2,max_new_tokens=5,temperature=0)
    assert first.prompt_tokens==3 and first.generated_tokens==2
    assert encoded==[('s1','u1'),('s2','u2')]
    assert all('past_key_values' not in c for c in calls)
    assert calls[0]['do_sample'] and not calls[1]['do_sample']
