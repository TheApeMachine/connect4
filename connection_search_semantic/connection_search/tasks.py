"""Immutable task specifications and independent, ordinary-loop reference evaluators.
Known-answer diagnostics, not a claim of novel mathematics or a novel benchmark.
The model sees the specification and small public examples, never reference code.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Any
from .dsl import R, pair, seq, digest


@dataclass(frozen=True)
class Case:
    inputs: dict[str,Any]
    modulus: int
    size: int


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    input_items: tuple
    output: Any
    parallel: bool = False

    @property
    def inputs(self) -> dict:
        return dict(self.input_items)

    @property
    def fingerprint(self) -> str:
        return digest([self.name,self.description,self.input_items,self.output,self.parallel])

    def span_bound(self,n: int) -> int | None:
        return 12*math.ceil(math.log2(n+1))+12 if self.parallel else None

    def reference(self,c: Case):
        """Only the verifier calls this; no reference-expression AST is provided."""
        d,m=c.inputs,c.modulus
        if self.name=='scalar_add':
            return (d['x']+d['y'])%m
        if self.name=='scalar_mul':
            return (d['x']*d['y'])%m
        if self.name=='pair_energy':
            return (d['x']**2+d['y']**2)%m
        if self.name=='pair_second':
            return d['p'][1]%m
        if self.name=='map_square':
            return [x*x%m for x in d['xs']]
        if self.name=='zip_inputs':
            return [(a%m,b%m) for a,b in zip(d['as_'],d['bs'])]
        out=[]
        acc=d.get('x0',1 if self.name=='prefix_product' else 0)%m
        if self.name.startswith('affine'):
            pairs=list(zip(d['as_'],d['bs']))
            if self.name=='affine_suffix':
                pairs.reverse()
            for a,b in pairs:
                if self.name=='affine_square_coefficient':
                    a=a*a
                acc=(a*acc+b)%m
                out.append(acc)
            if self.name=='affine_suffix':
                out.reverse()
            return out
        for x in d['xs']:
            if self.name=='prefix_product':
                acc=acc*x%m
            elif self.name=='prefix_square_sum':
                acc=(acc+x*x)%m
            elif self.name=='shifted_prefix_sum':
                acc=(acc+x)%m
            else:
                acc=(acc+x)%m
            out.append(acc)
        return out

    def cases(self, split: str, seed: int=1729, count: int | None=None) -> tuple[Case,...]:
        if split not in ('search','holdout'):
            raise ValueError('Unknown split')
        rng=random.Random(seed+(0 if split=='search' else 987_654_319))
        mods=[2,5,17,97,101] if split=='search' else [3,4,12,103,257,1009]
        lengths=[0,1,2,3,7,17,129] if split=='search' else [0,1,4,8,16,31,65,257]
        count=count or (28 if split=='search' else 64)
        cases=[]
        for i in range(count):
            n=lengths[i%len(lengths)]
            m=mods[(i//len(lengths))%len(mods)]
            # Include zeros, negatives, composite moduli and empty sequences.
            def val(t):
                if t==R:
                    return rng.randrange(-2*m,2*m+1)
                if t[0]=='Pair':
                    return tuple(val(tt) for tt in t[1:])
                return [val(t[1]) for _ in range(n)]
            data={name:val(t) for name,t in self.input_items}
            if i==0:
                for name,t in self.input_items:
                    if t==R:
                        data[name]=0
            cases.append(Case(data,m,n))
        return tuple(cases)

    def public(self,seed:int=1729) -> dict:
        examples=self.cases('search',seed,5)
        return {'task':self.name,'immutable_specification':self.description,
                'inputs':self.inputs,'output_type':self.output,
                'constraints':{'arithmetic':'exact modulo each case modulus',
                   'max_dependency_span':'12*ceil(log2(n+1))+12' if self.parallel else None,
                   'max_runtime_work':200000,
                   'note':'The span constraint is on the evaluated circuit, not measured GPU execution.'},
                'public_examples':[{'inputs':c.inputs,'modulus':c.modulus,
                                    'output':self.reference(c)} for c in examples[:4]]}


_TASKS=[
    Task('scalar_add','Return x+y modulo the case modulus.',(('x',R),('y',R)),R),
    Task('scalar_mul','Return x*y modulo the case modulus.',(('x',R),('y',R)),R),
    Task('pair_second','Return the second scalar of pair p.',(('p',pair()),),R),
    Task('map_square','Return every element of xs squared modulo the case modulus.',(('xs',seq(R)),),seq(R)),
    Task('zip_inputs','Return corresponding scalar pairs from equal-length as_ and bs.',(('as_',seq(R)),('bs',seq(R))),seq(pair())),
    Task('pair_energy','Return x*x+y*y modulo the case modulus.',(('x',R),('y',R)),R),
    Task('prefix_sum','Return all inclusive prefix sums of xs, modulo the case modulus. Empty input returns [].',(('xs',seq(R)),),seq(R),True),
    Task('prefix_product','Return all inclusive prefix products of xs, modulo the case modulus. Empty input returns [].',(('xs',seq(R)),),seq(R),True),
    Task('prefix_square_sum','Return all inclusive prefix sums of squared elements of xs, modulo the case modulus.',(('xs',seq(R)),),seq(R),True),
    Task('shifted_prefix_sum','Starting at x0, add each element of xs in order and return the state after every addition, modulo the case modulus.',(('xs',seq(R)),('x0',R)),seq(R),True),
    Task('affine_prefix','For equal-length as_ and bs, start with x=x0. For each position t in order set x=(as_[t]*x+bs[t]) modulo the case modulus. Return x after every update; empty input returns []. Meet the dependency-span bound.',(('as_',seq(R)),('bs',seq(R)),('x0',R)),seq(R),True),
    Task('affine_square_coefficient','For equal-length as_ and bs, start with x=x0. At position t set x=(as_[t]*as_[t]*x+bs[t]) modulo the case modulus. Return x after every update; empty input returns []. Meet the dependency-span bound.',(('as_',seq(R)),('bs',seq(R)),('x0',R)),seq(R),True),
    Task('affine_suffix','For equal-length as_ and bs, start with x=x0 at the right boundary. Visit positions from last to first, setting x=(as_[t]*x+bs[t]) modulo the case modulus. Return the resulting state at each ORIGINAL position, in original position order. Meet the dependency-span bound.',(('as_',seq(R)),('bs',seq(R)),('x0',R)),seq(R),True),
]
TASKS={t.name:t for t in _TASKS}
PROBES=('scalar_add','scalar_mul','pair_second','map_square','zip_inputs')


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError('Unknown task '+name+'; choose '+', '.join(TASKS)) from exc
