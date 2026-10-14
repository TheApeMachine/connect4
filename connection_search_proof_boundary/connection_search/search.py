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
    partial_pruning: bool = True

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
    censored_completions: int = 0
    archived: bool = False
    repair_turns: int = 0
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
        self.initial_frame_ids=[]
        self.failure_observations=[]
        self.last_turn_was_revision=False
        self.horizon_deferrals=set()
        from .completion import ClosingPlanner
        self.closing_planner=ClosingPlanner(self.library,
            components=(lambda t,env:self.graph.available(t,env)) if config.reuse else None)
        from .feasibility import Feasibility
        self.feasibility=Feasibility(task.inputs,task.output,task.public(config.seed)['public_examples'])

    def partial_check(self,expr):
        # This additional check applies only to typed proposals. All typed
        # methods use the same public-example analysis; no holdout input is read.
        if self.config.proposal_mode=='typed' and self.config.partial_pruning and holes(expr):
            check=self.feasibility.check(expr)
            if check:
                self.log.write('partial_refutation',expression=expr,check=check.to_dict(),
                               concrete_test_cases=0,source='public_examples_only')
            return check
        return None

    @staticmethod
    def refuted(state):
        return state.check.status in ('output_counterexample','constraint_counterexample',
                                      'law_counterexample','partial_output_counterexample')

    @staticmethod
    def needs_revision(state):
        # A checker limitation needs a repair opportunity, not a false claim
        # that the candidate is disproved. Trial accounting remains censored.
        return Search.refuted(state) or state.check.status=='law_unknown'

    def observe_failure(self,state):
        if self.needs_revision(state):
            self.failure_observations.append({'state':state.id,'program':state.expr,
                                             'observation_kind':'refutation' if self.refuted(state) else 'checker_limitation',
                                             'checker_feedback':prompt_feedback(state.check)})
            self.failure_observations[:]=self.failure_observations[-3:]

    def eligible(self,state):
        if not (not state.archived and bool(self.focus_paths(state)) and
                (state.check.status=='partial_type_checked' or self.config.revision)):
            return False
        if (self.config.proposal_mode=='typed' and self.config.backward and self.inference
                and state.check.status=='partial_type_checked'):
            remaining=self.inference.budget.max_calls-self.inference.budget.calls
            plan=self.closing_planner.fill(state.expr,self.task.inputs)
            if plan is None or plan.steps>remaining:
                marker=(state.id,remaining)
                if marker not in self.horizon_deferrals:
                    self.horizon_deferrals.add(marker)
                    self.log.write('state_horizon_deferred',state=state.id,remaining_calls=remaining,
                                   known_closure_steps=plan.steps if plan else None,
                                   not_a_mathematical_rejection=True)
                return False
        return True

    def context(self,observations=False):
        result={'task':self.task.public(self.config.seed),
                'reusable_functions':self.library.offered(),
                'reuse_syntax':'["use","lib_ID"] expands transparently to that fn; apply it with ["apply",["use","lib_ID"],ARGS...]'}
        if observations and self.failure_observations:
            result['search_observations']=list(self.failure_observations[-2:])
            result['observation_scope']='Each observation applies to its shown construction, not every shared component. Refutations require changing a fixed decision. A checker_limitation is unknown, not a disproof.'
        return result

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
        if check.status=='partial_type_checked':
            abstract=self.partial_check(expr)
            if abstract:
                abstract.certificates=check.certificates
                check=abstract
        if not holes(expr):
            self.full_candidates+=1
        self.log.write('checked',stage=stage,state=eid,parent=parent.id if parent else None,
                       replaced_path=list(focus),expression=expr,check=check.to_dict())
        root=None
        if check.status not in ('type_error','resource_limit'):
            root=self.graph.record(expr,self.task.inputs,check,parent.graph_root if parent else None,focus)
        state=State(eid,expr,check,root,parent.id if parent else None)
        self.states[eid]=state
        self.observe_failure(state)
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

    def pending_initial_frames(self):
        return [self.states[eid] for eid in self.initial_frame_ids
                if self.states[eid].visits==0 and self.eligible(self.states[eid])]

    def select(self):
        # Retain initial-branch fairness. Afterwards alternate reserved first
        # repair turns with ordinary search, instead of hoping random selection
        # eventually revisits a failed program among many unvisited ancestors.
        if self.config.proposal_mode=='typed':
            pending=self.pending_initial_frames()
            if pending:
                self.last_turn_was_revision=self.needs_revision(pending[0])
                return pending[0]
        eligible=[s for s in self.states.values() if self.eligible(s)]
        if not eligible:
            stalled=any(any(n>=self.config.max_stalled_attempts for n in s.stalled_attempts.values())
                        for s in self.states.values() if not s.archived)
            raise BudgetExhausted('proposal_stalled' if stalled else 'frontier_exhausted')
        repairs=[s for s in eligible if self.needs_revision(s) and s.repair_turns==0]
        if repairs and not self.last_turn_was_revision:
            chosen=repairs[0]
            self.last_turn_was_revision=True
            self.log.write('repair_turn',state=chosen.id,status=chosen.check.status,
                           reason='First repair turn reserved after a refutation or checker limitation',
                           mathematical_refutation=self.refuted(chosen))
            return chosen
        if self.rng.random()<self.config.exploration:
            least=min(s.visits for s in eligible)
            chosen=self.rng.choice([s for s in eligible if s.visits==least])
        else:
            def value(s):
                p=self.rng.betavariate(1+s.successful_completions,
                                      1+s.completion_trials-s.successful_completions)
                return p/math.sqrt(1+s.visits)
            chosen=max(eligible,key=value)
        self.last_turn_was_revision=self.needs_revision(chosen)
        return chosen

    def focus_paths(self,state):
        """A retry limit closes a state/path for this run, never proves it impossible."""
        allowed=lambda p: state.stalled_attempts.get(p,0)<self.config.max_stalled_attempts
        if state.check.path and state.check.status!='partial_type_checked' and allowed(state.check.path):
            return [state.check.path]
        options=holes(state.expr)
        if options and state.check.status=='partial_type_checked':
            return [p for p in options if allowed(p)]
        # A refuted partial program cannot be repaired by filling its holes.
        # Include fixed leaves too: an incorrect selected variable is a decision.
        cuts=[p for p,e,_,_ in walk_typed(state.expr,self.task.inputs)
              if not (isinstance(e,list) and e[0]=='hole')]
        return [p for p in dict.fromkeys([(),*cuts]) if allowed(p)]

    def focus(self,state):
        options=self.focus_paths(state)
        if not options:
            raise BudgetExhausted('proposal_stalled')
        if state.check.status!='partial_type_checked' and () in options and self.rng.random()<.3:
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

    def typed_step(self,stage,payload,*,excluded=(),components=(),planner=None,remaining_steps=None,completion_env=None):
        from .typed import menu
        focus=payload['focus']
        choices=menu(focus['expected_type'],focus['scope'],self.library,
                     construction=payload['construction'],path=tuple(focus['path']),
                     components=components,excluded=excluded,
                     seed=self.config.seed+self.inference.budget.calls*104729)
        if planner is not None:
            before=len(choices)
            choices=planner.admissible(choices,payload['construction'],tuple(focus['path']),
                                       completion_env,remaining_steps)
            self.log.write('completion_horizon_filter',stage=stage,remaining_steps=remaining_steps,
                           offered_before=before,offered_after=len(choices),
                           policy='constructive_closure_witness; syntax budget only, not correctness')
            if not choices:
                raise Invalid('No construction with a known closure in the remaining rollout budget',
                              kind='completion_limit',evidence={'remaining_steps':remaining_steps,
                                      'not_a_mathematical_rejection':True})
        return self.inference.ask(stage,payload,'replacement',choices=choices)

    def complete_typed(self,payload,partial,env,*,persist=None,step_limit=None):
        """Fill only holes, one constrained structural decision per model call.

        Graph rollouts persist partial work even on budget/step interruption.
        Direct/repair controls use the same grammar but receive no graph reuse.
        """
        from .completion import ClosingPlanner
        self.inference.budget.ensure()
        original=partial
        parent=persist
        planner=ClosingPlanner(self.library,
            components=(lambda t,local:self.graph.available(t,local)) if persist and self.config.reuse else None)
        available_calls=self.inference.budget.max_calls-self.inference.budget.calls
        limit=min(self.config.completion_steps,available_calls,
                  self.config.completion_steps if step_limit is None else step_limit)
        plan=planner.fill(partial,env)
        if plan is None or plan.steps>limit:
            details={'step_limit':limit,'known_closure_steps':plan.steps if plan else None,
                     'not_a_mathematical_rejection':True}
            self.log.write('completion_deferred',stage=payload['stage'],**details)
            if persist is None:
                # Avoid a zero-call retry loop in direct/repair or no-backward mode.
                raise BudgetExhausted('typed_completion_horizon_unavailable')
            raise Invalid('No known closure fits this rollout allocation',kind='completion_limit',evidence=details)
        self.log.write('completion_started',stage=payload['stage'],state=persist.id if persist else None,
                       step_limit=limit,known_closure_steps=plan.steps)
        for step in range(limit):
            spots=[(p,e,local,t) for p,e,local,t in walk_typed(partial,env)
                   if isinstance(e,list) and e[0]=='hole']
            if not spots:
                return partial
            path,current,local,t=spots[0]
            stage=payload['stage']
            request={**payload,'construction':partial,'completion_step':step,
                     'remaining_completion_steps':limit-step,
                     'focus':{'path':list(path),'expected_type':t,'scope':local,'current_term':current}}
            components=self.graph.available(t,local,exclude=current) if parent and self.config.reuse else []
            term=self.typed_step(stage,request,components=components,planner=planner,
                                 remaining_steps=limit-step,completion_env=env)
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
            if not parent and env==self.task.inputs:
                abstract=self.partial_check(partial)
                if abstract:
                    raise Invalid(abstract.message,kind=abstract.status,
                                  evidence={**abstract.evidence,'partial_expression':partial})
            if not holes(partial):
                return partial
        raise Invalid('Typed completion reached its structural-step limit',kind='completion_limit',
                      evidence={'steps':limit,'partial_expression':partial,
                                'not_a_mathematical_rejection':True})

    def ask_expansion(self,state,independent=False):
        path,current,env,t=self.focus(state)
        stage='independent_frame' if independent else ('expand_hole' if state.check.status=='partial_type_checked' else 'revise_decision')
        if stage=='revise_decision':
            self.revision_count+=1
            state.repair_turns+=1
        available=self.graph.available(t,env,exclude=current) if self.config.reuse and not independent else []
        payload={**self.context(observations=not independent),'stage':stage,
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
                    remaining=self.inference.budget.max_calls-self.inference.budget.calls
                    term=self.typed_step(stage,payload,excluded=excluded,components=available,
                                         planner=self.closing_planner,remaining_steps=remaining,
                                         completion_env=self.task.inputs)
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
        if (self.config.proposal_mode=='typed' and not self.last_turn_was_revision
                and any(self.needs_revision(s) and s.repair_turns==0 and self.eligible(s)
                        for s in self.states.values())):
            # Retain this partial child, but do not let its multi-call rollout
            # consume the budget reserved for a known failure's first repair.
            self.log.write('repair_preempts_rollout',state=state.id,
                           reason='A refutation or checker limitation is waiting for its first revision turn')
            return
        for _ in range(self.config.rollouts):
            if self.solution:
                break
            payload={**self.context(observations=True),'stage':'bounded_completion',
                     'instruction':'Fill every hole to obtain a complete program. All non-hole syntax must be preserved EXACTLY; do not revise previous decisions.',
                     'construction':state.expr}
            feedback=self.rejection_feedback(state,target='program')
            if feedback:
                payload['rejection_feedback']=feedback
            calls_before=self.inference.budget.calls
            observed=self.config.proposal_mode!='typed'
            success=False
            reason='interrupted'
            try:
                if self.config.proposal_mode=='typed':
                    remaining=self.inference.budget.max_calls-calls_before
                    pending=[s for s in self.pending_initial_frames() if s.id!=state.id]
                    allocation=min(self.config.completion_steps,remaining//(1+len(pending)))
                    self.log.write('rollout_allocation',state=state.id,remaining_calls=remaining,
                                   pending_initial_branches=len(pending),allocated_calls=allocation)
                    full=self.complete_typed(payload,state.expr,self.task.inputs,persist=state,
                                             step_limit=allocation)
                else:
                    full=self.inference.ask('bounded_completion',payload,'program')
                full=self.library.resolve(full)
                if holes(full) or not extends(state.expr,full):
                    raise Invalid('Completion changed an earlier choice or left holes',kind='contract_error')
                s=self.submit(full,state,(),'completion_trial')
                observed=observed or bool(s and (s.check.passed or s.check.status in
                                     ('output_counterexample','constraint_counterexample','law_counterexample','partial_output_counterexample')))
                reason=s.check.status if s else 'no_check_result'
                if s and s.check.passed:
                    state.successful_completions+=1
                    success=True
            except Invalid as exc:
                # Unknown laws, resource limits and unfinished trials are NOT
                # observed mathematical failures and do not update the beta counts.
                observed=observed or exc.kind in ('output_counterexample','constraint_counterexample','law_counterexample','partial_output_counterexample')
                reason=exc.kind
                if self.config.proposal_mode=='typed' and exc.kind=='completion_limit':
                    # This is a controller budget decision, not a rejected model
                    # response. Do not attach a stale last_call to the diagnostic.
                    self.log.write('completion_unavailable',state=state.id,status=exc.kind,
                                   message=str(exc),evidence=exc.evidence,source='controller')
                else:
                    diagnostic=self.remember_rejection(state,exc,target='program')
                    self.log.write('rejected_completion',state=state.id,status=exc.kind,message=str(exc),diagnostic=diagnostic)
            except BudgetExhausted as exc:
                reason=str(exc)
                raise
            finally:
                spent=self.inference.budget.calls-calls_before
                if spent:
                    self.rollout_count+=spent
                    if observed:
                        state.completion_trials+=1
                    else:
                        state.censored_completions+=1
                    self.log.write('completion_evidence',state=state.id,trials=state.completion_trials,
                                   successful=state.successful_completions,censored=state.censored_completions,
                                   calls=spent,outcome='success' if success else ('failure' if observed else 'censored'),
                                   reason=reason,interpretation=('Concrete checker outcomes or abstract refutations; unfinished or unknown typed trials are censored, not negative evidence.'
                                       if self.config.proposal_mode=='typed' else 'Success per JSON completion attempt; invalid outputs count as failed attempts.'))

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
                if s and s.id not in self.initial_frame_ids:
                    self.initial_frame_ids.append(s.id)
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
                'partial_feasibility_policy':('public_example_residue_sets' if self.config.proposal_mode=='typed' and self.config.partial_pruning else None),
                'feedback_policy':'adaptive_observations_and_reserved_repair_turns',
                'proof_policy':'polynomial_results_with_opaque_construction_values',
                'typed_completion_policy':('constructive_global_and_rollout_horizons' if self.config.proposal_mode=='typed' else None),
                'config':asdict(self.config),'initial_library_fingerprint':self.initial_library,
                'solution':solution.expr if solution else None,'search_check':solution.check.to_dict() if solution else None,
                'holdout_check':audit.to_dict() if audit else None,'budget':budget,
                'metrics':{'wall_seconds':time.perf_counter()-self.started,'verification_seconds':self.checker.seconds,
                           'verification_calls':self.checker.calls,'case_evaluations':self.checker.case_evaluations,
                           'partial_analysis':self.feasibility.metrics(),
                           'model_calls_with_observations':sum(e['event']=='model_call' and 'search_observations' in json.loads(e['user']) for e in self.log.events),
                           'reserved_repair_turns':sum(e['event']=='repair_turn' for e in self.log.events),
                           'unique_states':len(self.states),'complete_candidates':self.full_candidates,
                           'forward_bridge_attempts':self.reuse_hits,'completion_model_calls_attempted':self.rollout_count,
                           'revision_model_calls_attempted':self.revision_count,'graph_nodes':len(self.graph.nodes),
                           'graph_hyperedges':len(self.graph.edges),
                           'rejected_transitions':sum(e['event']=='rejected_transition' for e in self.log.events),
                           'stalled_focuses':sum(e['event']=='focus_stalled' for e in self.log.events),
                           'constrained_choice_calls':sum(e['event']=='typed_choice' for e in self.log.events),
                           'observed_completion_trials':sum(s.completion_trials for s in self.states.values()),
                           'censored_completions':sum(s.censored_completions for s in self.states.values()),
                           'completion_horizon_filtered_choices':sum(e['offered_before']-e['offered_after'] for e in self.log.events if e['event']=='completion_horizon_filter'),
                           'global_horizon_filtered_choices':sum(e['offered_before']-e['offered_after'] for e in self.log.events if e['event']=='completion_horizon_filter' and e['stage']!='bounded_completion'),
                           'horizon_deferred_states':len({sid for sid,_ in self.horizon_deferrals}),
                           'initial_branches_expanded':sum(self.states[eid].visits>0 for eid in self.initial_frame_ids)},
                'claims':{'whole_program_formally_proved':False,'historically_novel':False,
                          'hardware_parallel_speedup_measured':False,'search_beats_controls':False,
                          'exact_smc_sampler':False}}
