"""Conservative partial evaluation against already-public input/output examples.

An abstract value encloses ALL values a well-typed hole filling can produce.
Only exclusion of a required output refutes a partial program. Inclusion is
UNKNOWN, never a pass. No reference implementation, target AST, model, or solver
is used here. Resource exhaustion also returns unknown.

Scalars are sets of residues; pairs are products; sequences are either exact
length vectors or arbitrary-length sequences of an enclosed element set.
Function holes return top, while known closures preserve lexical scope. This
intentionally loses correlations and can miss impossible programs.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Any
from .dsl import R, Invalid, fn, infer, seq, validate_shape
from .verify import Check


class UnknownAnalysis(Exception):
    """A limitation of analysis, not a rejection of the candidate."""


@dataclass(frozen=True)
class UnknownFunction:
    typ: Any


@dataclass(frozen=True)
class Closure:
    expression: Any
    values: dict
    types: dict
    typ: Any


@dataclass(frozen=True)
class Sequence:
    element_type: Any
    items: tuple | None
    element: Any  # Overapproximation used only when items is None.


class AbstractRuntime:
    def __init__(self, modulus: int, max_steps: int = 5000):
        if not 2 <= modulus <= 17:
            raise UnknownAnalysis('Residue domain supports moduli 2..17')
        self.modulus, self.max_steps, self.steps = modulus, max_steps, 0

    def tick(self):
        self.steps += 1
        if self.steps > self.max_steps:
            raise UnknownAnalysis('Abstract operation budget exhausted')

    def top(self, typ):
        self.tick()
        if typ == R:
            return frozenset(range(self.modulus))
        if typ[0] == 'Pair':
            return (self.top(typ[1]), self.top(typ[2]))
        if typ[0] == 'Seq':
            return Sequence(typ[1], None, self.top(typ[1]))
        if typ[0] == 'Fn':
            return UnknownFunction(typ)
        raise UnknownAnalysis('Unsupported type')

    def wrap(self, value, typ):
        self.tick()
        if typ == R:
            return frozenset((value % self.modulus,))
        if typ[0] == 'Pair':
            return tuple(self.wrap(v, t) for v, t in zip(value, typ[1:]))
        if typ[0] == 'Seq':
            return self.vector(typ[1], [self.wrap(v, typ[1]) for v in value])
        raise UnknownAnalysis('Function-valued public input is unsupported')

    def vector(self, typ, items):
        return Sequence(typ, tuple(items), None)

    def elements(self, value):
        if value.items is None:
            return value.element
        if not value.items:
            # Empty vectors have no elements. Top is a safe, imprecise summary.
            return self.top(value.element_type)
        out = value.items[0]
        for item in value.items[1:]:
            out = self.join(out, item, value.element_type)
        return out

    def join(self, left, right, typ):
        self.tick()
        if left == right:
            return left
        if typ == R:
            return left | right
        if typ[0] == 'Pair':
            return tuple(self.join(a, b, t) for a, b, t in zip(left, right, typ[1:]))
        if typ[0] == 'Seq':
            if left.items is not None and right.items is not None and len(left.items) == len(right.items):
                return self.vector(typ[1], [self.join(a, b, typ[1]) for a, b in zip(left.items, right.items)])
            return Sequence(typ[1], None, self.join(self.elements(left), self.elements(right), typ[1]))
        return self.top(typ)

    def call(self, function, arguments):
        self.tick()
        if isinstance(function, UnknownFunction):
            return self.top(function.typ[2])
        if not isinstance(function, Closure):
            raise UnknownAnalysis('Unknown callable representation')
        parameters = function.expression[1]
        return self.eval(function.expression[2],
                         {**function.values, **{p[0]: a for p, a in zip(parameters, arguments)}},
                         {**function.types, **dict(parameters)})

    def closed_states(self, initial, step, typ):
        """A post-fixpoint includes every finite number of unknown-length steps."""
        reachable = initial
        for _ in range(32):
            enlarged = self.join(reachable, step(reachable), typ)
            if enlarged == reachable:
                return reachable
            reachable = enlarged
        # Never reject on incomplete fixed-point iteration.
        return self.top(typ)

    def eval(self, expression, values, types):
        self.tick()
        if type(expression) is int:
            return frozenset((expression % self.modulus,))
        if isinstance(expression, str):
            return values[expression]
        op = expression[0]
        if op == 'hole':
            return self.top(expression[1])
        if op == 'fn':
            return Closure(expression, dict(values), dict(types), infer(expression, types))
        args = [self.eval(e, values, types) for e in expression[1:]]
        if op in ('add', 'sub', 'mul'):
            a, b = args
            combine = {'add': lambda x, y: x+y, 'sub': lambda x, y: x-y,
                       'mul': lambda x, y: x*y}[op]
            return frozenset(combine(x, y) % self.modulus for x in a for y in b)
        if op == 'pair':
            return tuple(args)
        if op in ('fst', 'snd'):
            return args[0][0 if op == 'fst' else 1]
        if op == 'apply':
            return self.call(args[0], args[1:])
        if op == 'reverse':
            s = args[0]
            return s if s.items is None else self.vector(s.element_type, reversed(s.items))
        if op == 'zip':
            a, b = args
            typ = ['Pair', a.element_type, b.element_type]
            if a.items is not None and b.items is not None:
                if len(a.items) != len(b.items):
                    # Concrete execution fails; unknown avoids attributing an output.
                    return self.top(seq(typ))
                return self.vector(typ, zip(a.items, b.items))
            if a.items is not None:
                return self.vector(typ, [(v, b.element) for v in a.items])
            if b.items is not None:
                return self.vector(typ, [(a.element, v) for v in b.items])
            return Sequence(typ, None, (a.element, b.element))
        if op == 'map':
            f, s = args
            typ = f.typ[2]
            if s.items is None:
                return Sequence(typ, None, self.call(f, [s.element]))
            return self.vector(typ, [self.call(f, [v]) for v in s.items])
        if op == 'accumulate':
            f, state, s = args
            typ = f.typ[2]
            if s.items is None:
                states = self.closed_states(state, lambda v: self.call(f, [v, s.element]), typ)
                # accumulate emits AFTER updates: initial is reachable but not
                # necessarily emitted. Applying one more step safely tightens it.
                return Sequence(typ, None, self.call(f, [states, s.element]))
            out = []
            for element in s.items:
                state = self.call(f, [state, element])
                out.append(state)
            return self.vector(typ, out)
        if op == 'scan':
            f, s = args
            if s.items is None:
                states = self.closed_states(s.element, lambda v: self.call(f, [v, v]), s.element_type)
                return Sequence(s.element_type, None, states)
            # Same prefix circuit as Runtime, so no assumed associativity here.
            out, distance = list(s.items), 1
            while distance < len(out):
                old = out
                out = [self.call(f, [old[i-distance], old[i]]) if i >= distance else old[i]
                       for i in range(len(old))]
                distance *= 2
            return self.vector(s.element_type, out)
        raise UnknownAnalysis('Unsupported operator')

    def contains(self, value, concrete, typ):
        self.tick()
        if typ == R:
            return concrete % self.modulus in value
        if typ[0] == 'Pair':
            return all(self.contains(v, c, t) for v, c, t in zip(value, concrete, typ[1:]))
        if typ[0] == 'Seq':
            if value.items is not None:
                return len(value.items) == len(concrete) and all(
                    self.contains(v, c, typ[1]) for v, c in zip(value.items, concrete))
            return all(self.contains(value.element, c, typ[1]) for c in concrete)
        return True

    def describe(self, value, typ):
        if typ == R:
            return {'possible_residues': sorted(value)}
        if typ[0] == 'Pair':
            return {'pair': [self.describe(v, t) for v, t in zip(value, typ[1:])]}
        if typ[0] == 'Seq':
            if value.items is None:
                return {'length': 'unknown', 'every_element': self.describe(value.element, typ[1])}
            return {'length': len(value.items), 'elements': [self.describe(v, typ[1]) for v in value.items[:8]],
                    'truncated': len(value.items) > 8}
        return {'function': 'not summarized'}


class Feasibility:
    def __init__(self, inputs, output, examples, max_steps=5000):
        self.inputs, self.output, self.examples = inputs, output, examples
        self.max_steps = max_steps
        self.calls = self.example_evaluations = self.refutations = self.unknown_limits = 0
        self.seconds = 0.0

    def check(self, expression):
        """Return a refutation or None (unknown), never a successful proof."""
        start = time.perf_counter()
        self.calls += 1
        try:
            validate_shape(expression)
            if infer(expression, self.inputs) != self.output:
                return None  # The ordinary type checker owns this diagnostic.
            for i, example in enumerate(self.examples):
                try:
                    rt = AbstractRuntime(example['modulus'], self.max_steps)
                    self.example_evaluations += 1
                    values = {n: rt.wrap(example['inputs'][n], t) for n, t in self.inputs.items()}
                    abstract = rt.eval(expression, values, self.inputs)
                    if not rt.contains(abstract, example['output'], self.output):
                        self.refutations += 1
                        return Check('partial_output_counterexample',
                            'No hole-only completion can produce this public output while preserving the fixed syntax.',
                            evidence={'public_example_index': i, 'inputs': example['inputs'],
                                      'modulus': example['modulus'], 'expected': example['output'],
                                      'abstract_output': rt.describe(abstract, self.output),
                                      'basis': 'conservative residue-set abstract execution',
                                      'scope': 'all well-typed hole fillings preserving fixed syntax',
                                      'repair': 'Change at least one fixed decision. No unique blame location is established.'},
                            seconds=time.perf_counter()-start)
                except (UnknownAnalysis, RecursionError):
                    self.unknown_limits += 1
        except Invalid:
            return None
        finally:
            self.seconds += time.perf_counter()-start
        return None

    def metrics(self):
        return {'calls': self.calls, 'example_evaluations': self.example_evaluations,
                'refutations': self.refutations, 'unknown_limits': self.unknown_limits, 'seconds': self.seconds,
                'evidence_source': 'public_examples_only', 'concrete_test_cases': 0}
