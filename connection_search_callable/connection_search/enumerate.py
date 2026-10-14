"""Bounded model-free type-directed enumeration over the same executable DSL.

This is deliberately reported as bounded grammar search, not exhaustive search
of the entire language. Bucket/type/size/time limits are explicit in results.
No task-name dispatch, answer templates, or reference evaluator is used here.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from functools import lru_cache
from itertools import product
import json
import time
from .dsl import R,Invalid,algebraic,fn,holes,key,pair,seq,valid_type
from .algebra import symbolic,scall,symbol,serialize_poly
from .inference import BudgetExhausted


@dataclass
class EnumerationLimits:
    max_size:int=9
    max_candidates:int=2000
    bucket_cap:int=256
    max_expansions:int=50000
    seconds:float=20.0

    def __post_init__(self):
        if min(self.max_size,self.max_candidates,self.bucket_cap,self.max_expansions)<1 or self.seconds<=0:
            raise ValueError('Enumeration limits must be positive')


def splits(total,k):
    if k==1:
        if total>=1:
            yield (total,)
        return
    for first in range(1,total-k+2):
        for rest in splits(total-first,k-1):
            yield (first,)+rest


def scope_terms(env):
    """Existing names and bound pair projections, with their lexical types."""
    def descend(term,typ):
        yield term,typ
        if isinstance(typ,list) and typ[0]=='Pair':
            for op,child in zip(('fst','snd'),typ[1:]):
                yield from descend([op,term],child)
    for name,typ in env.items():
        yield from descend(name,typ)


class Enumerator:
    def __init__(self,inputs,output,library,limits:EnumerationLimits):
        self.inputs,self.output,self.library,self.limits=inputs,output,library,limits
        self.start=time.perf_counter()
        self.expansions=0
        self.truncated_buckets=0
        self.semantic_duplicates=0
        self.best_semantic_size={}
        carriers={key(R):R,key(pair()):pair()}
        def collect(t):
            if t==R:
                return
            if t[0]=='Pair':
                carriers[key(t)]=t
                collect(t[1]);collect(t[2])
            elif t[0]=='Seq':
                collect(t[1])
            elif t[0]=='Fn':
                for x in t[1]+[t[2]]:
                    collect(x)
        for t in list(inputs.values())+[output]:
            collect(t)
        self.carriers=list(carriers.values())
        self.values=self.carriers+[seq(t) for t in self.carriers]

    def tick(self):
        self.expansions+=1
        if self.expansions>self.limits.max_expansions:
            raise BudgetExhausted('enumeration_expansion_cap')
        if time.perf_counter()-self.start>=self.limits.seconds:
            raise BudgetExhausted('enumeration_time_cap')

    def productions(self,t):
        productions=[]
        if t==R:
            productions.extend((op,[R,R]) for op in ('add','sub','mul'))
        if isinstance(t,list) and t[0]=='Pair':
            productions.append(('pair',t[1:]))
        for p in self.carriers:
            if isinstance(p,list) and p[0]=='Pair':
                if p[1]==t:
                    productions.append(('fst',[p]))
                if p[2]==t:
                    productions.append(('snd',[p]))
        if isinstance(t,list) and t[0]=='Seq':
            elem=t[1]
            if algebraic(elem):
                productions.append(('scan',[fn([elem,elem],elem),t]))
            for a in self.carriers:
                productions.append(('map',[fn([a],elem),seq(a)]))
                productions.append(('accumulate',[fn([elem,a],elem),elem,seq(a)]))
            productions.append(('reverse',[t]))
            if isinstance(elem,list) and elem[0]=='Pair':
                productions.append(('zip',[seq(elem[1]),seq(elem[2])]))
        for a in self.values:
            productions.append(('apply',[fn([a],t),a]))
        # A valid target can have an invalid higher-order prerequisite: adding
        # an apply wrapper may cross the DSL's type-depth limit. Such a
        # production is unavailable, not an error in the current construction.
        return [(op,types) for op,types in productions if all(valid_type(ty) for ty in types)]

    def one_step(self,t,env,*,application_policy='unrestricted',callables=()):
        """One grammar constructor with typed prerequisites; no task/answer access.

        The production list is shared with the model-free enumerator. Bound
        projections are also offered as already-executable, transparent terms.
        Every result is independently shape/type checked by the caller.
        """
        if application_policy not in ('unrestricted','existing'):
            raise ValueError('Unknown typed application policy')
        if not valid_type(t):
            return
        scoped=list(scope_terms(env))
        for term,ty in scoped:
            if ty==t:
                yield term
        if t==R:
            yield from (-1,0,1)
        for entry in self.library.offered():
            if entry['type']==t:
                yield ['use',entry['id']]
        if isinstance(t,list) and t[0]=='Fn':
            base=len(env)
            names=[f'p{base}_{i}' for i in range(len(t[1]))]
            while any(name in env for name in names):
                base+=1
                names=[f'p{base}_{i}' for i in range(len(t[1]))]
            yield ['fn',list(map(list,zip(names,t[1]))),['hole',t[2]]]
        for op,types in self.productions(t):
            if application_policy=='existing' and op=='apply':
                continue
            yield [op,*[['hole',ty] for ty in types]]
        if application_policy=='existing':
            # Application heads must already exist in scope, the starting
            # library, or the scoped graph inventory. Arguments may be holes.
            # New lambdas are still constructed by the ordinary fn production.
            sources=scoped+[(entry['term'],entry['type']) for entry in self.library.offered()]
            sources.extend((item['term'],item['type']) for item in callables)
            for head,typ in sources:
                if (valid_type(typ) and isinstance(typ,list) and typ[0]=='Fn'
                        and len(typ[1])==1 and typ[2]==t and not holes(head)):
                    yield ['apply',head,['hole',typ[1][0]]]

    def semantic_key(self,e,t,env):
        # Exact polynomial equivalence pruning strengthens the model-free control.
        # Unsupported sequence behavior falls back to syntax, not guessed equality.
        try:
            resolved=self.library.resolve(e)
            syms={n:symbol(ty,'capture:'+n) for n,ty in env.items() if algebraic(ty)}
            fuel=[50000]
            value=symbolic(resolved,syms,fuel)
            if isinstance(t,list) and t[0]=='Fn':
                if not all(algebraic(ty) for ty in t[1]) or not algebraic(t[2]):
                    return 'syntax:'+key(e)
                value=scall(value,[symbol(ty,'arg:'+str(i)) for i,ty in enumerate(t[1])],fuel)
            elif not algebraic(t):
                return 'syntax:'+key(e)
            return 'polynomial:'+key(serialize_poly(value))
        except (Invalid,TypeError,AttributeError):
            return 'syntax:'+key(e)

    def programs(self):
        @lru_cache(maxsize=8192)
        def build(tj,ej,n):
            self.tick()
            t,env=json.loads(tj),json.loads(ej)
            found=[]
            seen=set()
            def add(e):
                k=self.semantic_key(e,t,env)
                global_key=(tj,ej,k)
                if k in seen or self.best_semantic_size.get(global_key,n)<n:
                    self.semantic_duplicates+=1
                    return True
                self.best_semantic_size[global_key]=min(n,self.best_semantic_size.get(global_key,n))
                seen.add(k);found.append(e)
                if len(found)>=self.limits.bucket_cap:
                    self.truncated_buckets+=1
                    return False
                return True
            if n==1:
                for name,ty in env.items():
                    if ty==t and not add(name):
                        return tuple(found)
                if t==R:
                    for c in (-1,0,1):
                        if not add(c):
                            return tuple(found)
                for entry in self.library.offered():
                    if entry['type']==t and not add(['use',entry['id']]):
                        return tuple(found)
                return tuple(found)
            if isinstance(t,list) and t[0]=='Fn':
                base=len(env)
                names=[f'p{base}_{i}' for i in range(len(t[1]))]
                while any(name in env for name in names):
                    base+=1;names=[f'p{base}_{i}' for i in range(len(t[1]))]
                params=list(map(list,zip(names,t[1])))
                local={**env,**dict(params)}
                for body in build(key(t[2]),key(local),n-1):
                    if not add(['fn',params,body]):
                        return tuple(found)
            productions=self.productions(t)
            for op,types in productions:
                for ns in splits(n-1,len(types)):
                    self.tick()
                    pools=[]
                    for ty,sz in zip(types,ns):
                        pool=build(key(ty),ej,sz)
                        if not pool:
                            break
                        pools.append(pool)
                    if len(pools)!=len(types):
                        continue
                    for args in product(*pools):
                        self.tick()
                        if not add([op,*args]):
                            return tuple(found)
            return tuple(found)
        for n in range(1,self.limits.max_size+1):
            for term in build(key(self.output),key(self.inputs),n):
                yield term

    def statistics(self):
        return {'limits':asdict(self.limits),'expansions':self.expansions,
                'truncated_buckets':self.truncated_buckets,'semantic_duplicates_pruned':self.semantic_duplicates,'carrier_types':self.carriers,
                'complete_language_enumeration':False,
                'note':'Exact polynomial normal forms prune algebraically equivalent terms. Constants -1,0,1; task carrier types plus Pair(R,R); arbitrary nesting is bounded by size. Function application production is unary; inline lambdas support operator-required arities.'}


def run_enumeration(search,limits):
    enum=Enumerator(search.task.inputs,search.task.output,search.library,limits)
    tried=0
    try:
        for expr in enum.programs():
            if tried>=limits.max_candidates:
                raise BudgetExhausted('enumeration_candidate_cap')
            search.submit(expr,stage='type_directed_enumeration')
            tried+=1
            if search.solution:
                search.stop_reason='search_solution_frozen'
                break
        else:
            search.stop_reason='bounded_grammar_exhausted'
    except BudgetExhausted as exc:
        search.stop_reason=str(exc)
    search.log.write('enumeration_statistics',**enum.statistics(),candidates_submitted=tried)
    return enum.statistics()
