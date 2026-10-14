"""Persistent executable search, structural cuts, and bounded completion trials.
This is a stochastic graph search, NOT an exact SMC/posterior sampler.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,field
import json
import math
from pathlib import Path
import random
import time
from typing import Any
from .dsl import Invalid,at,digest,extends,holes,infer,key,replace,size,validate_shape,walk_typed
from .graph import Graph
from .inference import BudgetExhausted,Inference,Log
from .library import Library
from .tasks import Task
from .verify import Check,Checker,prompt_feedback


@dataclass
class Config:
    seed: int = 1729
    frames: int = 3
    rollouts: int = 1
    exploration: float = .25
    max_states: int = 256
    bridge_checks: int = 2
    backward: bool = True
    revision: bool = True
    reuse: bool = True
    max_checks: int = 4096
    max_stalled_attempts: int = 4
    proposal_mode: str = 'json'
    completion_steps: int = 32

    def __post_init__(self):
        if (self.frames<1 or self.rollouts<0 or not 0<=self.exploration<=1 or self.max_states<1
                or self.bridge_checks<0 or self.max_checks<1 or self.max_stalled_attempts<1
                or self.proposal_mode not in ('json','typed') or self.completion_steps<1):
            raise ValueError('Invalid search configuration')


@dataclass
class State:
    id: str
    expr: Any
    check: Check
    graph_root: str | None
    parent: str | None = None
    visits: int = 0
    completion_trials: int = 0
    successful_completions: int = 0
    archived: bool = False
    rejections: list[dict[str,Any]] = field(default_factory=list)
    stalled_attempts: dict[tuple[int,...],int] = field(default_factory=dict)


class Search:
    def __init__(self,task:Task,inference:Inference|None,log:Log,config:Config,library:Library|None=None):
        self.task,self.inference,self.log,self.config=task,inference,log,config
        self.library=library.clone() if library else Library()
        self.initial_library=self.library.fingerprint
        self.checker=Checker(task,config.seed)
        self.graph=Graph()
        self.rng=random.Random(config.seed)
        self.states={}
        self.solution:State|None=None
        self.started=time.perf_counter()
        self.fingerprint=task.fingerprint
        self.stop_reason='not_started'
        self.reuse_hits=0
        self.rollout_count=0
        self.revision_count=0
        self.check_count=0
        self.full_candidates=0

    def context(self):
        return {'task':self.task.public(self.config.seed),
                'reusable_functions':self.library.offered(),
                'reuse_syntax':'["use","lib_ID"] expands transparently to that fn; apply it with ["apply",["use","lib_ID"],ARGS...]'}

    def submit(self,expr,parent=None,focus=(),stage='proposal'):
        if self.task.fingerprint!=self.fingerprint:
            raise AssertionError('Immutable task was modified')
        if self.check_count>=self.config.max_checks:
            raise BudgetExhausted('verification_call_cap')
        try:
            expr=self.library.resolve(expr)
        except Invalid as exc:
            self.log.write('rejected_expression',stage=stage,message=str(exc),status=exc.kind,path=list(exc.path))
            return None
        eid=digest(expr)
        if eid in self.states:
            self.log.write('duplicate',stage=stage,state=eid,parent=parent.id if parent else None)
            return self.states[eid]
        self.check_count+=1
        check=self.checker.check(expr)
        if not holes(expr):
            self.full_candidates+=1
        self.log.write('checked',stage=stage,state=eid,parent=parent.id if parent else None,
                       replaced_path=list(focus),expression=expr,check=check.to_dict())
        root=None
        if check.status not in ('type_error','resource_limit'):
            root=self.graph.record(expr,self.task.inputs,check,parent.graph_root if parent else None,focus)
        state=State(eid,expr,check,root,parent.id if parent else None)
        self.states[eid]=state
        if check.passed:
            self.solution=state
        elif check.status=='type_error' or root is None:
            state.archived=True
        # Archive, don't destroy, the least sampled evidence-poor surplus states.
        live=[s for s in self.states.values() if not s.archived]
        if len(live)>self.config.max_states:
            candidates=[s for s in live if s.id!=eid and s.check.status!='partial_type_checked'] or live[:-1]
            if candidates:
                worst=max(candidates,key=lambda s:(s.visits,-s.successful_completions,size(s.expr)))
                worst.archived=True
                self.log.write('archived',state=worst.id,reason='frontier_capacity; expression and evidence retained')
        return state

    def select(self):
        eligible=[s for s in self.states.values() if not s.archived and
                  (holes(s.expr) or self.config.revision) and self.focus_paths(s)]
        if not eligible:
            stalled=any(any(n>=self.config.max_stalled_attempts for n in s.stalled_attempts.values())
                        for s in self.states.values() if not s.archived)
            raise BudgetExhausted('proposal_stalled' if stalled else 'frontier_exhausted')
        if self.rng.random()<self.config.exploration:
            least=min(s.visits for s in eligible)
            return self.rng.choice([s for s in eligible if s.visits==least])
        def value(s):
            # A bounded completion success rate under THIS proposal policy, not novelty.
            p=self.rng.betavariate(1+s.successful_completions,
                                  1+s.completion_trials-s.successful_completions)
            return p/math.sqrt(1+s.visits)
        return max(eligible,key=value)

    def focus_paths(self,state):
        """A retry limit closes a state/path for this run, never proves it impossible."""
        allowed=lambda p: state.stalled_attempts.get(p,0)<self.config.max_stalled_attempts
        if state.check.path and state.check.status!='partial_type_checked' and allowed(state.check.path):
            return [state.check.path]
        options=holes(state.expr)
        if options:
            return [p for p in options if allowed(p)]
        cuts=[p for p,e,_,_ in walk_typed(state.expr,self.task.inputs)
              if isinstance(e,list) and e[0] not in ('hole','fn')]
        return [p for p in dict.fromkeys([(),*cuts]) if allowed(p)]

    def focus(self,state):
        options=self.focus_paths(state)
        if not options:
            raise BudgetExhausted('proposal_stalled')
        if not holes(state.expr) and () in options and self.rng.random()<.3:
            path=()
        else:
            path=self.rng.choice(options)
        for p,e,env,t in walk_typed(state.expr,self.task.inputs):
            if p==path:
                return path,e,env,t
        raise AssertionError('Verifier returned a non-expression blame path')

    def rejection(self,exc,path=(),target='replacement'):
        """Bounded model-facing diagnostic; the full response stays in events.jsonl."""
        row=self.inference.last_call or {}
        raw=row.get('raw','')
        return {'focus_path':list(path),'target':target,'status':exc.kind,'message':str(exc),
                'error_path':list(exc.path),'error_path_relative_to':target,
                'evidence':exc.evidence,'call_index':row.get('index'),
                'rejected_response':raw[:2048],'response_truncated':len(raw)>2048}

    def remember_rejection(self,state,exc,path=(),target='replacement'):
        item=self.rejection(exc,path,target)
        state.rejections.append(item)
        # Independent proposal slots remain independent. Only adaptive calls consume this memory.
        state.rejections[:]=state.rejections[-16:]
        return item

    def rejection_feedback(self,state,path=(),target='replacement'):
        return [r for r in state.rejections if r['focus_path']==list(path) and r['target']==target][-3:]

    def no_progress(self,state,path,independent,reason):
        if independent:
            return
        count=state.stalled_attempts.get(path,0)+1
        state.stalled_attempts[path]=count
        self.log.write('proposal_no_progress',state=state.id,path=list(path),attempts=count,reason=reason)
        if count>=self.config.max_stalled_attempts:
            self.log.write('focus_stalled',state=state.id,path=list(path),attempts=count,
                           reason='Adaptive retry limit; not a proof that this construction is impossible.')
            print(f'  focus stalled after {count} unproductive adaptive attempts: {state.id} {list(path)}',flush=True)

    def typed_step(self,stage,payload,*,excluded=(),components=()):
        from .typed import menu
        focus=payload['focus']
        choices=menu(focus['expected_type'],focus['scope'],self.library,
                     construction=payload['construction'],path=tuple(focus['path']),
                     components=components,excluded=excluded,
                     seed=self.config.seed+self.inference.budget.calls*104729)
        return self.inference.ask(stage,payload,'replacement',choices=choices)

    def complete_typed(self,payload,partial,env,*,persist=None):
        """Fill only holes, one constrained structural decision per model call.

        Graph rollouts persist partial work even on budget/step interruption.
        Direct/repair controls use the same grammar but receive no graph reuse.
        """
        original=partial
        parent=persist
        for step in range(self.config.completion_steps):
            spots=[(p,e,local,t) for p,e,local,t in walk_typed(partial,env)
                   if isinstance(e,list) and e[0]=='hole']
            if not spots:
                return partial
            path,current,local,t=spots[0]
            stage=payload['stage']
            request={**payload,'construction':partial,'completion_step':step,
                     'focus':{'path':list(path),'expected_type':t,'scope':local,'current_term':current}}
            components=self.graph.available(t,local,exclude=current) if parent and self.config.reuse else []
            term=self.typed_step(stage,request,components=components)
            partial=replace(partial,path,term)
            if not extends(original,partial):
                raise AssertionError('Typed completion changed a previously fixed decision')
            validate_shape(partial)
            self.log.write('typed_completion_step',stage=stage,step=step,path=list(path),
                           expression=partial,remaining_holes=len(holes(partial)))
            if parent:
                parent=self.submit(partial,parent,path,'typed_completion_step')
                if parent and parent.check.status not in ('partial_type_checked','passed_search_tests'):
                    raise Invalid(parent.check.message,path=parent.check.path,kind=parent.check.status,
                                  evidence=parent.check.evidence)
                if parent and parent.check.passed:
                    return partial
            if not holes(partial):
                return partial
        raise Invalid('Typed completion reached its structural-step limit',kind='completion_limit',
                      evidence={'steps':self.config.completion_steps,'partial_expression':partial,
                                'not_a_mathematical_rejection':True})

    def ask_expansion(self,state,independent=False):
        path,current,env,t=self.focus(state)
        stage='independent_frame' if independent else ('expand_hole' if holes(state.expr) else 'revise_decision')
        if stage=='revise_decision':
            self.revision_count+=1
        available=self.graph.available(t,env,exclude=current) if self.config.reuse and not independent else []
        payload={**self.context(),'stage':stage,
                 'instruction':'Propose one executable structural choice at the focused path. '
                   +('Use typed holes for as-yet unresolved prerequisites; do not merely describe them. ' if self.config.backward
                     else 'The replacement must be hole-free. ')
                   +'Do not alter the existing construction outside this path. '
                   +'A goal counterexample does not locate causal blame; cuts without a local-law error are exploratory.',
                 'construction':state.expr,'focus':{'path':list(path),'expected_type':t,'scope':env,'current_term':current}}
        if not independent:
            payload['feedback']=prompt_feedback(state.check)
            payload['available_components']=available
            feedback=self.rejection_feedback(state,path)
            if feedback:
                payload['rejection_feedback']=feedback
                payload['instruction']+=' Correct the rejected proposal using its diagnostics, or choose a different construction. Do not repeat it unchanged.'
        state.visits+=1
        try:
            if self.config.proposal_mode=='typed':
                if self.config.backward:
                    excluded=() if independent else [key(s.expr) for s in self.states.values()]
                    term=self.typed_step(stage,payload,excluded=excluded,components=available)
                else:
                    term=self.complete_typed(payload,['hole',t],env)
            else:
                term=self.inference.ask(stage,payload,'replacement')
            term=self.library.resolve(term)
            if not self.config.backward and holes(term):
                raise Invalid('Backward-search ablation requires a hole-free replacement',kind='contract_error')
            actual_type=infer(term,env)
            if actual_type!=t:
                raise Invalid('Replacement does not satisfy focused type',evidence={'expected':t,'actual':actual_type})
            candidate=replace(state.expr,path,term)
            seen=digest(candidate) in self.states
            child=self.submit(candidate,state,path,stage)
            if seen or child is None or child.archived:
                why='Proposal already exists in the frontier' if seen else 'Proposal produced no usable state'
                self.remember_rejection(state,Invalid(why,kind='no_progress'),path)
                self.no_progress(state,path,independent,why)
                return None
            state.stalled_attempts[path]=0
            return child
        except Invalid as exc:
            diagnostic=self.remember_rejection(state,exc,path)
            self.log.write('rejected_transition',state=state.id,stage=stage,message=str(exc),status=exc.kind,
                           path=list(path),diagnostic=diagnostic)
            print(f'  rejected {exc.kind}: {exc}',flush=True)
            if exc.kind=='proposal_exhausted':
                state.stalled_attempts[path]=self.config.max_stalled_attempts
                self.log.write('typed_menu_exhausted',state=state.id,path=list(path),
                               note='No untried one-step grammar choices; not complete-language exhaustion')
            else:
                self.no_progress(state,path,independent,str(exc))
            return None

    def rollouts(self,state):
        if not state or not holes(state.expr) or state.check.status!='partial_type_checked':
            return
        for _ in range(self.config.rollouts):
            if self.solution:
                break
            payload={**self.context(),'stage':'bounded_completion',
                     'instruction':'Fill every hole to obtain a complete program. All non-hole syntax must be preserved EXACTLY; do not revise previous decisions.',
                     'construction':state.expr}
            feedback=self.rejection_feedback(state,target='program')
            if feedback:
                payload['rejection_feedback']=feedback
            calls_before=self.inference.budget.calls
            try:
                if self.config.proposal_mode=='typed':
                    full=self.complete_typed(payload,state.expr,self.task.inputs,persist=state)
                else:
                    full=self.inference.ask('bounded_completion',payload,'program')
                full=self.library.resolve(full)
                if holes(full) or not extends(state.expr,full):
                    raise Invalid('Completion changed an earlier choice or left holes',kind='contract_error')
                s=self.submit(full,state,(),'completion_trial')
                if s and s.check.passed:
                    state.successful_completions+=1
            except Invalid as exc:
                diagnostic=self.remember_rejection(state,exc,target='program')
                self.log.write('rejected_completion',state=state.id,status=exc.kind,message=str(exc),diagnostic=diagnostic)
            finally:
                if self.inference.budget.calls>calls_before:
                    self.rollout_count+=self.inference.budget.calls-calls_before
                    state.completion_trials+=1
            self.log.write('completion_evidence',state=state.id,trials=state.completion_trials,
                           successful=state.successful_completions,
                           interpretation='Observed completed-and-checked success under current model/prompt/budget, not a theorem value.')

    def bridges(self,state):
        if not self.config.reuse or not state or state.check.status!='partial_type_checked':
            return
        remaining=self.config.bridge_checks
        for path,term,env,t in walk_typed(state.expr,self.task.inputs):
            if not (isinstance(term,list) and term[0]=='hole'):
                continue
            choices=self.graph.available(t,env,exclude=term,limit=16)
            self.rng.shuffle(choices)
            for item in choices:
                if remaining<=0 or self.solution:
                    return
                candidate=replace(state.expr,path,item['term'])
                if digest(candidate) in self.states:
                    continue
                remaining-=1
                self.reuse_hits+=1
                self.log.write('forward_bridge',state=state.id,path=list(path),source_node=item['node_id'],
                               evidence=item['evidence'])
                self.submit(candidate,state,path,'forward_bridge')

    def graph_search(self):
        root=self.submit(['hole',self.task.output],stage='initial_goal')
        self.stop_reason='searching'
        try:
            # Independent first proposals receive no other frames or generated failure history.
            for _ in range(self.config.frames):
                s=self.ask_expansion(root,independent=True)
                if self.solution:
                    break
            while not self.solution:
                self.inference.budget.ensure()
                state=self.select()
                self.bridges(state)
                if self.solution:
                    break
                child=self.ask_expansion(state)
                if self.solution:
                    break
                self.bridges(child)
                if self.solution:
                    break
                self.rollouts(child)
            if self.solution:
                self.stop_reason='search_solution_frozen'
        except BudgetExhausted as exc:
            self.stop_reason=str(exc)

    def direct_search(self,repair=False):
        self.stop_reason='searching'
        previous=None
        previous_rejection=None
        try:
            while not self.solution:
                self.inference.budget.ensure()
                payload={**self.context(),'stage':'repair' if repair and (previous or previous_rejection) else 'direct',
                         'instruction':'Produce one complete executable program satisfying the original task and every constraint. No holes.'}
                if repair and previous:
                    payload['previous_program']=previous.expr
                    payload['checker_feedback']=prompt_feedback(previous.check)
                if repair and previous_rejection:
                    payload['rejection_feedback']=[previous_rejection]
                try:
                    if self.config.proposal_mode=='typed':
                        expr=self.complete_typed(payload,['hole',self.task.output],self.task.inputs)
                    else:
                        expr=self.inference.ask(payload['stage'],payload,'program')
                    expr=self.library.resolve(expr)
                    if holes(expr):
                        raise Invalid('Direct/repair candidates must be complete',kind='contract_error')
                    s=self.submit(expr,previous if repair else None,stage=payload['stage'])
                    if repair and s:
                        previous=s
                        previous_rejection=None
                except Invalid as exc:
                    previous_rejection=self.rejection(exc,target='program')
                    self.log.write('rejected_transition',stage=payload['stage'],message=str(exc),status=exc.kind,
                                   diagnostic=previous_rejection)
                    print(f'  rejected {exc.kind}: {exc}',flush=True)
            self.stop_reason='search_solution_frozen'
        except BudgetExhausted as exc:
            self.stop_reason=str(exc)

    def finish(self,mode):
        # Freeze selection before touching held-out tests. Never feed audit failure back to search.
        solution=self.solution
        audit=None
        if solution:
            self.log.write('selection_frozen',state=solution.id,expression=solution.expr)
            audit=self.checker.check(solution.expr,'holdout')
            self.log.write('holdout_audit',state=solution.id,check=audit.to_dict(),feedback_to_search=False)
            added=self.library.harvest(solution.expr,self.task.inputs,
                    {'task':self.task.name,'task_fingerprint':self.fingerprint,'program_id':solution.id},solution.check)
            self.log.write('library_harvest',ids=added,source_status=solution.check.status,
                           note='Evidence is source-program testing; imported entries are re-typechecked and expanded transparently.')
        budget=self.inference.budget.to_dict() if self.inference else None
        return {'schema':1,'method':mode,'task':self.task.name,'task_fingerprint':self.fingerprint,
                'status':'passed_holdout_tests' if audit and audit.passed else ('failed_holdout_tests' if audit else 'not_solved'),
                'stop_reason':self.stop_reason,'model_inference_executed':bool(self.inference and self.inference.budget.calls and not getattr(self.inference.backend,'synthetic',False)),
                'backend_generation_executed':bool(self.inference and self.inference.budget.calls),
                'synthetic_backend':bool(getattr(self.inference.backend,'synthetic',False)) if self.inference else False,
                'frozen_weights':True,'cross_call_chat_memory':False,'hidden_state_intervention':False,
                'proposal_mode':self.config.proposal_mode,
                'constrained_action_decoding':self.config.proposal_mode=='typed' and self.inference is not None,
                'proposal_language_note':('Bounded shared enumeration grammar, constants -1/0/1, unary apply, typed constructors and bound projections; no full-language completeness claim.' if self.config.proposal_mode=='typed' else 'Free-form JSON proposals in the full supported DSL.'),
                'config':asdict(self.config),'initial_library_fingerprint':self.initial_library,
                'solution':solution.expr if solution else None,'search_check':solution.check.to_dict() if solution else None,
                'holdout_check':audit.to_dict() if audit else None,'budget':budget,
                'metrics':{'wall_seconds':time.perf_counter()-self.started,'verification_seconds':self.checker.seconds,
                           'verification_calls':self.checker.calls,'case_evaluations':self.checker.case_evaluations,
                           'unique_states':len(self.states),'complete_candidates':self.full_candidates,
                           'forward_bridge_attempts':self.reuse_hits,'completion_model_calls_attempted':self.rollout_count,
                           'revision_model_calls_attempted':self.revision_count,'graph_nodes':len(self.graph.nodes),
                           'graph_hyperedges':len(self.graph.edges),
                           'rejected_transitions':sum(e['event']=='rejected_transition' for e in self.log.events),
                           'stalled_focuses':sum(e['event']=='focus_stalled' for e in self.log.events),
                           'constrained_choice_calls':sum(e['event']=='typed_choice' for e in self.log.events)},
                'claims':{'whole_program_formally_proved':False,'historically_novel':False,
                          'hardware_parallel_speedup_measured':False,'search_beats_controls':False,
                          'exact_smc_sampler':False}}
