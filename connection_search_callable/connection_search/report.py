"""Dependency-free local report. Model strings are escaped; no remote assets."""
from __future__ import annotations
import html
import json
from pathlib import Path


def write_outputs(out:Path,search,summary):
    def dump(name,obj):
        (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    dump('summary.json',summary)
    dump('graph.json',search.graph.to_dict())
    dump('states.json',[{'id':s.id,'expression':s.expr,'parent':s.parent,'visits':s.visits,
                        'completion_trials':s.completion_trials,'successful_completions':s.successful_completions,
                        'censored_completions':s.censored_completions,
                        'archived':s.archived,'repair_turns':s.repair_turns,'rejections':s.rejections,
                        'stalled_attempts':[{'path':list(p),'attempts':n} for p,n in sorted(s.stalled_attempts.items())],
                        'check':s.check.to_dict()} for s in search.states.values()])
    search.library.save(out/'library.json')
    (out/'graph.dot').write_text(search.graph.dot())
    if summary.get('solution') is not None:
        dump('solution.json',{'program':summary['solution']})
    esc=lambda x:html.escape(str(x),quote=True)
    pre=lambda x:'<pre>'+esc(json.dumps(x,indent=2,ensure_ascii=False))+'</pre>'
    events=[]
    for row in search.log.events:
        view={k:v for k,v in row.items() if k not in ('system','user','raw')}
        detail=pre(view)
        if row['event']=='model_call':
            detail+='<h3>Raw model response</h3><pre>'+esc(row.get('raw',''))+'</pre>'
            detail+='<details><summary>Exact system and user prompts</summary><pre>'+esc(row.get('system',''))+'</pre><pre>'+esc(row.get('user',''))+'</pre></details>'
        events.append('<details><summary>'+esc(str(row['index'])+' · '+row['event'])+'</summary>'+detail+'</details>')
    h='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Connection search — run audit</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 24px;color:#182332;background:#f6f8fb}
h1{font-size:32px;letter-spacing:-.8px}h2{margin-top:36px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace;background:white;border:1px solid #dce2ea;padding:16px;border-radius:8px}
.note{padding:16px;background:#fff4d7;border-left:4px solid #a47d19}details{margin:8px 0;padding:10px;border:1px solid #dce2ea;background:white;border-radius:6px}summary{cursor:pointer;font-weight:600}a{color:#215a8a}.tiles{display:flex;gap:16px;flex-wrap:wrap}.tile{padding:16px;background:white;border:1px solid #dce2ea;min-width:170px}.number{display:block;font-size:25px;font-weight:700}
</style><h1>Executable connection search</h1>'''
    h+='<p>'+esc(summary['method']+' / '+summary['task']+' / '+summary.get('proposal_mode','json')+' proposals')+'</p>'
    h+='<p class="note">'+('SYNTHETIC CONTROLLER REGRESSION — hand-authored responses, not model discovery. ' if summary.get('synthetic_backend') else '')
    h+='Finite test success is not a theorem proof or a novelty claim. Circuit span is not measured parallel hardware speedup.</p>'
    m=summary['metrics'];budget=summary.get('budget') or {}
    h+='<div class="tiles">'+''.join('<div class="tile">'+esc(label)+'<span class="number">'+esc(value)+'</span></div>' for label,value in
                [('Status',summary['status']),('Model calls',budget.get('calls',0)),('Typed graph nodes',m['graph_nodes']),('Cases evaluated',m['case_evaluations'])])+'</div>'
    h+='<p>Stop reason: <strong>'+esc(summary['stop_reason'])+'</strong></p>'
    if m['case_evaluations']==0:
        h+='<p class="note">No task cases were evaluated. This run did not establish mathematical or search-method performance.</p>'
    if summary.get('proposal_mode')=='typed':
        h+='<p class="note">Typed-choice decoding: the model selects IDs for type-checked grammar steps. The controller assembles syntax; finite tests and algebraic-law checks still determine correctness. Each structural decision is a charged model call.</p>'
    if summary.get('typed_application_policy')=='existing':
        h+='<p class="note">Existing-callable proposal policy: apply steps use a function already present in scope, the library, or reusable graph components. New lambdas remain available; speculative unknown application heads are not offered. This is a search restriction, not a semantic equivalence proof. Programs execute unchanged.</p>'
    if summary.get('typed_completion_policy'):
        h+='<p class="note">Budget-aware completions: every offered rollout step has a known hole-free extension within the remaining call allocation. This is a syntax-budget witness, not mathematical evidence. Initial branches receive an expansion turn; interrupted/unknown trials are recorded separately.</p>'
        h+='<p>Observed completion trials: '+esc(m.get('observed_completion_trials',0))+'. Censored completions: '+esc(m.get('censored_completions',0))+'. Initial branches expanded: '+esc(m.get('initial_branches_expanded',0))+'.</p>'
    if summary.get('typed_completion_policy')=='constructive_global_and_rollout_horizons':
        h+='<p class="note">Global call horizon: adaptive expansions and revisions also require a known closure within the calls left. Deferred states are retained; deferral is not a mathematical refutation.</p>'
        h+='<p>Choices removed outside rollouts: '+esc(m.get('global_horizon_filtered_choices',0))+'. States deferred for the remaining horizon: '+esc(m.get('horizon_deferred_states',0))+'.</p>'
    if summary.get('proof_policy')=='polynomial_results_with_opaque_construction_values':
        h+='<p class="note">Polynomial laws describe callback results, conditional on successful construction. Unused sequence values may remain opaque in the algebra proof; the original program still undergoes strict runtime evaluation, nested scan-law checks and resource checks. Opaque outputs are not proofs.</p>'
    if summary.get('partial_feasibility_policy'):
        a=m.get('partial_analysis',{})
        h+='<p class="note">Partial feasibility: conservative abstract evaluation against public examples. Exclusion refutes every hole-only completion of that fixed construction; inclusion means unknown, not success. Abstract checks do not count as executed task cases.</p>'
        h+='<p>Partial refutations: '+esc(a.get('refutations',0))+'. Abstract example evaluations: '+esc(a.get('example_evaluations',0))+'. Revision calls: '+esc(m.get('revision_model_calls_attempted',0))+'. Calls with checker observations: '+esc(m.get('model_calls_with_observations',0))+'.</p>'
    h+='<h2>Frozen selected program</h2>'+pre(summary.get('solution'))
    h+='<h2>Result and accounting</h2>'+pre(summary)
    h+='<h2>Typed hypergraph</h2><p>Each hyperedge requires every ordered tail together. '
    h+='Inspect <a href="graph.json">graph.json</a> or render <a href="graph.dot">graph.dot</a> with Graphviz. Unchanged expressions retain content identities.</p>'
    h+='<h2>Search timeline</h2>'+''.join(events)
    h+='<p>Raw calls, prompts, invalid output and counterexamples: <a href="events.jsonl">events.jsonl</a>. '
    h+='Reusable transparent functions: <a href="library.json">library.json</a>.</p></html>'
    (out/'report.html').write_text(h)
