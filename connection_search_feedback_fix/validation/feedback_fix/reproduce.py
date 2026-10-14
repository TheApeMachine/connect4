#!/usr/bin/env python3
"""Replay user-supplied outputs to test feedback/stall handling, NOT model discovery."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from connection_search.cli import prepare
from connection_search.inference import Budget, Inference, Log
from connection_search.replay import ReplayBackend
from connection_search.report import write_outputs
from connection_search.search import Config, Search
from connection_search.tasks import get_task


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,help='New output directory; existing runs are never overwritten')
    args=parser.parse_args()
    try:
        out=prepare(args.out)
        data=json.loads((ROOT/'tests/fixtures/uploaded_affine_failures.json').read_text())
        log=Log(out/'events.jsonl')
        backend=ReplayBackend(data['raw_replies'])
        inference=Inference(backend,Budget(max_calls=32,max_tokens=500000),log,max_new_tokens=4096)
        search=Search(get_task('affine_prefix'),inference,log,Config(frames=3,rollouts=0,bridge_checks=0))
        log.write('replay_configuration',source_sha256=data['source_sha256'],synthetic_backend=True,
                  note='Fixed historical replies; does not predict model responses to new prompts. Token units are synthetic bytes, not original Gemma tokens.')
        search.graph_search()
        summary=search.finish('graph')
        summary['replay_note']='Historical responses replayed unchanged. No new model inference; no discovery claim.'
        write_outputs(out,search,summary)
        print(json.dumps({'status':summary['status'],'stop_reason':summary['stop_reason'],
                          'calls':inference.budget.calls,'case_evaluations':search.checker.case_evaluations,
                          'synthetic_backend':True,'report':str(out/'report.html')},indent=2))
        return 0
    except (ValueError,RuntimeError,OSError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
