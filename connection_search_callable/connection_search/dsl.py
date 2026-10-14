"""A bounded, typed, purely functional language over exact modular arithmetic.

There is no eval/exec, Python import, filesystem access, or generated host code.
A scan is a staged Hillis--Steele prefix circuit. The Python evaluator executes
that circuit serially while recording *dependency span*, not hardware speedup.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
import hashlib
import json

R = 'R'
Type = Any
Expr = Any
Path = tuple[int, ...]


def seq(t: Type) -> list:
    return ['Seq', t]


def pair(a: Type = R, b: Type = R) -> list:
    return ['Pair', a, b]


def fn(args: list, out: Type) -> list:
    return ['Fn', args, out]


def key(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(obj: Any) -> str:
    return hashlib.sha256(key(obj).encode()).hexdigest()[:20]


class Invalid(ValueError):
    def __init__(self, message: str, path: Path = (), kind: str = 'type_error', evidence: Any = None):
        super().__init__(message)
        self.path, self.kind, self.evidence = path, kind, evidence


class Limit(Invalid):
    def __init__(self, message: str, path: Path = ()):
        super().__init__(message, path, 'resource_limit')


def valid_type(t: Type, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if t == R:
        return True
    if not isinstance(t, list) or not t:
        return False
    if t[0] == 'Seq':
        return len(t) == 2 and valid_type(t[1], depth+1)
    if t[0] == 'Pair':
        return len(t) == 3 and all(valid_type(x, depth+1) for x in t[1:])
    if t[0] == 'Fn':
        return (len(t) == 3 and isinstance(t[1], list) and 0 < len(t[1]) <= 4
                and all(valid_type(x, depth+1) for x in t[1]) and valid_type(t[2], depth+1))
    return False


def algebraic(t: Type) -> bool:
    return t == R or (isinstance(t, list) and len(t) == 3 and t[0] == 'Pair'
                      and algebraic(t[1]) and algebraic(t[2]))


def children(e: Expr) -> list[tuple[int, Expr]]:
    if not isinstance(e, list) or not e:
        return []
    if e[0] == 'hole':
        return []
    if e[0] == 'fn':
        return [(2, e[2])] if len(e) == 3 else []
    return list(enumerate(e[1:], 1))


def size(e: Expr) -> int:
    return 1 + sum(size(c) for _, c in children(e))


def holes(e: Expr, path: Path = ()) -> list[Path]:
    if isinstance(e, list) and len(e) == 2 and e[0] == 'hole':
        return [path]
    return [p for i,c in children(e) for p in holes(c, path+(i,))]


def at(e: Expr, path: Path) -> Expr:
    for i in path:
        if not isinstance(e, list) or i not in dict(children(e)):
            raise Invalid('Path is not an expression child', path, 'bad_path')
        e = e[i]
    return e


def replace(e: Expr, path: Path, term: Expr) -> Expr:
    if not path:
        return term
    at(e, path[:1])
    out = list(e)
    out[path[0]] = replace(e[path[0]], path[1:], term)
    return out


def extends(partial: Expr, full: Expr) -> bool:
    """A rollout may fill holes, never silently change an earlier decision."""
    if isinstance(partial, list) and len(partial) == 2 and partial[0] == 'hole':
        return True
    if type(partial) is not type(full):
        return False
    if isinstance(partial, list):
        return len(partial) == len(full) and all(extends(a,b) for a,b in zip(partial,full))
    return partial == full


def validate_shape(e: Expr, max_nodes: int = 256, max_depth: int = 40) -> None:
    count = 0
    def walk(x: Expr, path: Path) -> None:
        nonlocal count
        count += 1
        if count > max_nodes or len(path) > max_depth:
            raise Limit('Expression size/depth limit exceeded', path)
        if type(x) is int:
            if x.bit_length() > 128:
                raise Limit('Integer literal exceeds 128 bits', path)
            return
        if isinstance(x, str):
            if not x.isidentifier() or len(x) > 64:
                raise Invalid('Invalid variable name', path)
            return
        if not isinstance(x, list) or not x or not isinstance(x[0], str):
            raise Invalid('Expression must be an integer, name, or operator array', path)
        arity = {'hole':1,'fn':2,'add':2,'sub':2,'mul':2,'pair':2,'fst':1,'snd':1,
                 'zip':2,'map':2,'scan':2,'accumulate':3,'reverse':1}
        op = x[0]
        if op == 'apply':
            if not 3 <= len(x) <= 6:
                raise Invalid('apply needs a function and 1 to 4 arguments', path)
        elif op not in arity or len(x) != arity[op]+1:
            raise Invalid('Unknown operator or wrong arity: '+op, path)
        if op == 'hole' and not valid_type(x[1]):
            raise Invalid('Malformed hole type', path)
        if op == 'fn':
            if not isinstance(x[1],list) or not 1 <= len(x[1]) <= 4:
                raise Invalid('fn needs 1 to 4 typed parameters', path)
            names = set()
            for p in x[1]:
                if (not isinstance(p,list) or len(p)!=2 or not isinstance(p[0],str)
                        or not p[0].isidentifier() or len(p[0])>64 or not valid_type(p[1])):
                    raise Invalid('Malformed parameter', path)
                if p[0] in names:
                    raise Invalid('Duplicate parameter', path)
                names.add(p[0])
        for i,c in children(x):
            walk(c,path+(i,))
    walk(e,())


def infer(e: Expr, env: dict[str,Type], path: Path = ()) -> Type:
    def error(s: str):
        raise Invalid(s,path)
    def check(cond: bool, s: str):
        if not cond:
            error(s)
    if type(e) is int:
        return R
    if isinstance(e,str):
        if e not in env:
            raise Invalid('Unbound variable: '+e,path,evidence={'in_scope_names':sorted(env)})
        return env[e]
    if not isinstance(e,list) or not e:
        error('Malformed expression')
    op=e[0]
    if op=='hole':
        return e[1]
    if op=='fn':
        local={**env, **dict(e[1])}
        return fn([p[1] for p in e[1]],infer(e[2],local,path+(2,)))
    ts=[infer(c,env,path+(i,)) for i,c in children(e)]
    if op in ('add','sub','mul'):
        check(ts==[R,R],op+' requires two scalars')
        return R
    if op=='pair':
        return pair(*ts)
    if op in ('fst','snd'):
        check(isinstance(ts[0],list) and ts[0][0]=='Pair',op+' requires a pair')
        return ts[0][1 if op=='fst' else 2]
    if op=='zip':
        check(all(isinstance(t,list) and t[0]=='Seq' for t in ts),'zip requires sequences')
        return seq(pair(ts[0][1],ts[1][1]))
    if op=='reverse':
        check(isinstance(ts[0],list) and ts[0][0]=='Seq','reverse requires a sequence')
        return ts[0]
    if op=='apply':
        f=ts[0]
        check(isinstance(f,list) and f[0]=='Fn','apply requires a function')
        check(ts[1:]==f[1],'apply argument types do not match')
        return f[2]
    if op in ('map','scan','accumulate'):
        f=ts[0]; s=ts[-1]
        check(isinstance(f,list) and f[0]=='Fn',op+' requires a function')
        check(isinstance(s,list) and s[0]=='Seq',op+' requires a sequence')
        elem=s[1]
        if op=='map':
            check(f[1]==[elem],'map function input does not match sequence element')
            return seq(f[2])
        if op=='scan':
            check(f==fn([elem,elem],elem),'scan requires a homogeneous binary function')
            check(algebraic(elem),'scan proof currently supports scalar/pair algebraic carriers')
            return s
        state=ts[1]
        expected=fn([state,elem],state)
        if f!=expected:
            raise Invalid('accumulate requires (state,element)->state',path,
                          evidence={'expected_callback_type':expected,'actual_callback_type':f})
        return seq(state)
    error('Unsupported operator: '+op)


def walk_typed(e: Expr, env: dict[str,Type], path: Path = ()):
    yield path,e,dict(env),infer(e,env,path)
    for i,c in children(e):
        local={**env,**dict(e[1])} if e[0]=='fn' and i==2 else env
        yield from walk_typed(c,local,path+(i,))


def free_vars(e: Expr, bound: frozenset[str] = frozenset()) -> set[str]:
    if isinstance(e,str):
        return set() if e in bound else {e}
    if not isinstance(e,list) or not e or e[0]=='hole':
        return set()
    if e[0]=='fn':
        return free_vars(e[2],bound|frozenset(p[0] for p in e[1]))
    return set().union(*(free_vars(c,bound) for _,c in children(e)))


@dataclass(frozen=True)
class Value:
    data: Any
    ready: int = 0


@dataclass
class Closure:
    params: list
    body: Expr
    env: dict[str,Value]


class Runtime:
    def __init__(self, modulus: int, step_limit: int=200_000):
        if type(modulus) is not int or modulus<2:
            raise ValueError('modulus must be an integer >=2')
        self.modulus,self.step_limit=modulus,step_limit
        self.work=0

    def tick(self, n: int=1, path: Path=()):
        self.work+=n
        if self.work>self.step_limit:
            raise Limit('Runtime work limit exceeded',path)

    def wrap(self,x: Any,t: Type) -> Value:
        if t==R:
            if type(x) is not int:
                raise Invalid('Input is not an integer',kind='input_error')
            return Value(x%self.modulus)
        if t[0]=='Pair':
            if not isinstance(x,(list,tuple)) or len(x)!=2:
                raise Invalid('Input is not a pair',kind='input_error')
            return Value(tuple(self.wrap(v,ty) for v,ty in zip(x,t[1:])))
        if t[0]=='Seq':
            if not isinstance(x,(list,tuple)) or len(x)>4096:
                raise Invalid('Input sequence missing or too long',kind='input_error')
            return Value([self.wrap(v,t[1]) for v in x])
        raise Invalid('External function inputs are unsupported',kind='input_error')

    def call(self,f: Value,args: list[Value],path: Path=()) -> Value:
        if not isinstance(f.data,Closure) or len(args)!=len(f.data.params):
            raise Invalid('Invalid function application',path,'runtime_error')
        c=f.data
        return self.eval(c.body,{**c.env,**dict(zip([p[0] for p in c.params],args))},path)

    def eval(self,e: Expr,env: dict[str,Value],path: Path=()) -> Value:
        self.tick(path=path)
        if type(e) is int:
            return Value(e%self.modulus)
        if isinstance(e,str):
            return env[e]
        op=e[0]
        if op=='hole':
            raise Invalid('Cannot execute a hole',path,'incomplete')
        if op=='fn':
            return Value(Closure(e[1],e[2],dict(env)))
        a=[self.eval(c,env,path+(i,)) for i,c in children(e)]
        ready=max((v.ready for v in a),default=0)
        if op in ('add','sub','mul'):
            u,v=(x.data for x in a)
            val=u+v if op=='add' else u-v if op=='sub' else u*v
            return Value(val%self.modulus,ready+1)
        if op=='pair':
            return Value(tuple(a),ready+1)
        if op in ('fst','snd'):
            v=a[0].data[0 if op=='fst' else 1]
            return Value(v.data,max(v.ready,ready)+1)
        if op=='zip':
            if len(a[0].data)!=len(a[1].data):
                raise Invalid('zip precondition failed: equal lengths required',path,'precondition_failed')
            self.tick(len(a[0].data),path)
            return Value([Value((x,y),max(x.ready,y.ready,ready)+1) for x,y in zip(a[0].data,a[1].data)],ready+1)
        if op=='reverse':
            self.tick(len(a[0].data),path)
            return Value(list(reversed(a[0].data)),ready+1)
        if op=='apply':
            return self.call(a[0],a[1:],path)
        if op=='map':
            vals=[self.call(a[0],[Value(v.data,max(v.ready,a[1].ready))],path) for v in a[1].data]
            return Value(vals,max([ready]+[v.ready for v in vals]))
        if op=='scan':
            vals=[Value(v.data,max(v.ready,a[1].ready)) for v in a[1].data]
            stride=1
            # Each round reads only the preceding round; order is left then right.
            while stride<len(vals):
                old=vals
                vals=[old[i] if i<stride else self.call(a[0],[old[i-stride],old[i]],path) for i in range(len(old))]
                stride*=2
            return Value(vals,max([ready]+[v.ready for v in vals]))
        if op=='accumulate':
            acc=a[1]; vals=[]
            for v in a[2].data:
                acc=self.call(a[0],[acc,Value(v.data,max(v.ready,a[2].ready))],path)
                vals.append(acc)
            return Value(vals,max([ready]+[v.ready for v in vals]))
        raise Invalid('Unhandled runtime operator',path,'runtime_error')


def plain(v: Value) -> Any:
    if isinstance(v.data,Closure):
        raise Invalid('Functions are not observable task outputs',kind='runtime_error')
    if isinstance(v.data,tuple):
        return tuple(plain(x) for x in v.data)
    if isinstance(v.data,list):
        return [plain(x) for x in v.data]
    return v.data


DSL_GUIDE = '''Typed JSON expression language (no Python):
Types: "R" = an exact scalar modulo the case's modulus; ["Pair",A,B]; ["Seq",T]; ["Fn",[A,...],B].
An expression is an integer literal, a bound variable name as a JSON string, or one of these arrays:
Names must match the supplied scope EXACTLY. A variable is "v", never ["v"].
["hole",TYPE] (unfinished part); ["fn",[[NAME,TYPE],...],BODY];
["add",A,B], ["sub",A,B], ["mul",A,B] (scalars);
["pair",A,B]; ["fst",PAIR]; ["snd",PAIR];
["zip",SEQ_A,SEQ_B] (equal lengths -> sequence of pairs);
["reverse",SEQ]; ["apply",FUNCTION,ARG,...];
["map",FUNCTION,SEQ] (unary function applied to every element);
["scan",FUNCTION,SEQ] (inclusive prefix of associative binary FUNCTION;
  FUNCTION(left_earlier,right_later); empty input -> []; function associativity
  must normalize to an exact polynomial identity over integer scalars/pairs);
["accumulate",FUNCTION,INITIAL,SEQ] (sequential prefix fold, FUNCTION(state,element)->state;
  returns states AFTER every element; empty input -> []; linear dependency span).
Callback types: map uses ["Fn",[E],O]; scan uses ["Fn",[E,E],E];
accumulate uses ["Fn",[S,E],S], where E is the sequence element type and S the initial state type.
A pair-valued element is ONE parameter, not two scalar parameters. Access it with fst/snd.
The evaluator uses exact modular arithmetic; floats, division and arbitrary host code do not exist.
scan is a logarithmic-round prefix circuit, map is elementwise parallel in the recorded cost model.
Measured work/span are interpreter circuit diagnostics, not hardware benchmark timings.
Keep expressions <=256 nodes. Use holes to make short, type-correct structural decisions.
A function can capture in-scope names. Task inputs and task constraints cannot be changed.
Return exactly the requested JSON object. No prose, markdown, self-scores or invented operators.'''
