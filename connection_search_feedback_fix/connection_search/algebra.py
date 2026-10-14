"""Small exact polynomial normalizer for scan's associativity obligation.
Certificates are relative to this implementation, not Lean/kernel certificates.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Any
from .dsl import Closure, Expr, Invalid, Limit, Runtime, Value, children, holes, infer, key, plain, walk_typed, R

MAX_TERMS=4096
MAX_DEGREE=24


@dataclass(frozen=True)
class Poly:
    terms: dict[tuple[str,...],int]

    @staticmethod
    def constant(n: int) -> 'Poly':
        return Poly({():n} if n else {})

    @staticmethod
    def variable(name: str) -> 'Poly':
        return Poly({(name,):1})

    def combine(self,other: 'Poly',op: str) -> 'Poly':
        out=dict(self.terms) if op!='mul' else {}
        if op=='mul':
            if len(self.terms)*len(other.terms)>MAX_TERMS*4:
                raise Limit('Polynomial multiplication budget exceeded')
            for a,u in self.terms.items():
                for b,v in other.terms.items():
                    mono=tuple(sorted(a+b))
                    if len(mono)>MAX_DEGREE:
                        raise Limit('Polynomial degree limit exceeded')
                    out[mono]=out.get(mono,0)+u*v
        else:
            for m,c in other.terms.items():
                out[m]=out.get(m,0)+(c if op=='add' else -c)
        out={m:c for m,c in out.items() if c}
        if len(out)>MAX_TERMS or any(c.bit_length()>8192 for c in out.values()):
            raise Limit('Polynomial normal form limit exceeded')
        return Poly(out)


@dataclass
class SClosure:
    params:list
    body:Expr
    env:dict


def symbol(t: Any, name: str):
    if t==R:
        return Poly.variable(name)
    if isinstance(t,list) and t[0]=='Pair':
        return (symbol(t[1],name+'.0'),symbol(t[2],name+'.1'))
    raise Invalid('Associativity contains non-algebraic captures',kind='law_unknown')


def symbolic(e: Expr, env: dict, fuel: list[int]):
    fuel[0]-=1
    if fuel[0]<0:
        raise Limit('Symbolic evaluation budget exceeded')
    if type(e) is int:
        return Poly.constant(e)
    if isinstance(e,str):
        if e not in env:
            raise Invalid('Non-algebraic free variable in scan function: '+e,kind='law_unknown')
        return env[e]
    op=e[0]
    if op=='fn':
        return SClosure(e[1],e[2],dict(env))
    if op not in ('add','sub','mul','pair','fst','snd','apply'):
        raise Invalid('Unsupported operator in polynomial proof: '+op,kind='law_unknown')
    vals=[symbolic(c,env,fuel) for _,c in children(e)]
    if op in ('add','sub','mul'):
        return vals[0].combine(vals[1],op)
    if op=='pair':
        return tuple(vals)
    if op in ('fst','snd'):
        return vals[0][0 if op=='fst' else 1]
    return scall(vals[0],vals[1:],fuel)


def scall(f: SClosure,args: list,fuel: list[int]):
    if not isinstance(f,SClosure):
        raise Invalid('Opaque function cannot be proved by polynomial normalization',kind='law_unknown')
    return symbolic(f.body,{**f.env,**dict(zip([p[0] for p in f.params],args))},fuel)


def serialize_poly(p):
    if isinstance(p,Poly):
        return [[list(m),c] for m,c in sorted(p.terms.items())]
    return [serialize_poly(x) for x in p]


def random_value(t,rng):
    if t==R:
        return rng.randrange(101)
    return (random_value(t[1],rng),random_value(t[2],rng))


def prove_associative(term: Expr, env: dict, element_type: Any) -> dict:
    symbolic_env={}
    for name,t in env.items():
        try:
            symbolic_env[name]=symbol(t,'capture:'+name)
        except Invalid:
            pass
    fuel=[50_000]
    f=symbolic(term,symbolic_env,fuel)
    x,y,z=(symbol(element_type,n) for n in ('x','y','z'))
    left=scall(f,[scall(f,[x,y],fuel),z],fuel)
    right=scall(f,[x,scall(f,[y,z],fuel)],fuel)
    if left==right:
        return {'status':'proved_polynomial_identity','law':'associativity',
                'domain':'integer polynomial identity; therefore every modular instance',
                'normal_form':serialize_poly(left)}
    # A normal-form mismatch is not by itself a task-specific numeric counterexample.
    rng=random.Random(9173)
    for _ in range(64):
        try:
            rt=Runtime(101)
            values={n:rt.wrap(random_value(t,rng),t) for n,t in env.items() if n in symbolic_env}
            closure=rt.eval(term,values)
            triple=[random_value(element_type,rng) for _ in range(3)]
            a,b,c=[rt.wrap(v,element_type) for v in triple]
            l=plain(rt.call(closure,[rt.call(closure,[a,b]),c]))
            r=plain(rt.call(closure,[a,rt.call(closure,[b,c])]))
            if l!=r:
                raise Invalid('Scan function is not associative',kind='law_counterexample',
                              evidence={'modulus':101,'captures':{n:plain(v) for n,v in values.items()},
                                        'triple':triple,'left':l,'right':r})
        except Invalid as exc:
            if exc.kind=='law_counterexample':
                raise
            break
    raise Invalid('Associativity not proved by supported normalization',kind='law_unknown')


def scan_laws(expr: Expr, env: dict) -> list[dict]:
    proofs=[]
    for path,e,local,_ in walk_typed(expr,env):
        if isinstance(e,list) and e[0]=='scan' and not holes(e[1]):
            t=infer(e[2],local)
            try:
                proof=prove_associative(e[1],local,t[1])
            except Invalid as exc:
                raise Invalid(str(exc),path+(1,),exc.kind,exc.evidence) from exc
            proofs.append({'path':list(path),'function':e[1],**proof})
    return proofs
