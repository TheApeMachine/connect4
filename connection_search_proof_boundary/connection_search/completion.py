"""Constructive call-budget checks for typed completions.

A plan is a *witness* of a hole-free extension, not a correct task solution,
not a lower bound, and not a completeness claim. Its steps use the existing
one-step grammar. Plans are never submitted or used as model answers.
"""
from __future__ import annotations
from dataclasses import dataclass
from .dsl import Invalid, holes, infer, key, replace, size, validate_shape, walk_typed
from .enumerate import Enumerator, EnumerationLimits


@dataclass(frozen=True)
class Plan:
    expression: object
    steps: int


class ClosingPlanner:
    """Small syntax-only inhabitation procedure; no Task or checker access.

    Uses terminals/projections, lambda introduction, pairs, zip, and map over
    an available sequence. This deliberate subset gives a conservative upper
    bound on known closure cost. A missing plan means unknown, not impossible.
    """
    def __init__(self, library, components=None):
        self.library = library
        # Optional graph lookup exposes only the same typed, hole-free terms
        # offered to this search path. It does not supply a task reference.
        self.components = components
        self.cache = {}

    def close(self, expected, scope):
        cache_key = (key(expected), key(scope))
        if cache_key in self.cache:
            return self.cache[cache_key]
        # Recursion follows output subtypes; the sentinel also guards cycles.
        self.cache[cache_key] = None
        grammar = Enumerator(scope, expected, self.library, EnumerationLimits())
        choices = list(grammar.one_step(expected, scope))
        if self.components is not None:
            choices.extend(item['term'] for item in self.components(expected, scope))
        candidates = []
        for candidate in choices:
            term = self.library.resolve(candidate)
            if not holes(term):
                candidates.append(Plan(term, 1))
        if not candidates:
            for term in choices:
                if not isinstance(term, list):
                    continue
                op = term[0]
                # Avoid unbounded synthesis inside a feasibility test. Other
                # operators remain available to the model; only the witness
                # builder uses this structurally decreasing subset.
                if op not in ('fn', 'pair', 'zip', 'map'):
                    continue
                if op == 'map':
                    # Witness maps only over an existing sequence, not another
                    # recursively invented map. Its callback may be constructed.
                    seq_type = term[2][1]
                    sources = []
                    for source in grammar.one_step(seq_type, scope):
                        source = self.library.resolve(source)
                        if not holes(source):
                            sources.append(source)
                    if not sources:
                        continue
                    function = self.close(term[1][1], scope)
                    if function is not None:
                        source = min(sources, key=lambda e: (size(e), key(e)))
                        candidates.append(Plan(['map', function.expression, source],
                                               2 + function.steps))
                    continue
                full = term
                calls = 1
                for path, expr, local, typ in walk_typed(term, scope):
                    if not (isinstance(expr, list) and expr[0] == 'hole'):
                        continue
                    child = self.close(typ, local)
                    if child is None:
                        break
                    full = replace(full, path, child.expression)
                    calls += child.steps
                else:
                    candidates.append(Plan(full, calls))
        valid = []
        for plan in candidates:
            try:
                validate_shape(plan.expression)
                if not holes(plan.expression) and infer(plan.expression, scope) == expected:
                    valid.append(plan)
            except Invalid:
                pass
        result = min(valid, key=lambda p: (p.steps, size(p.expression), key(p.expression)), default=None)
        self.cache[cache_key] = result
        return result

    def fill(self, partial, scope):
        """Return a checked hole-free witness and its additional selection cost."""
        # Components can grow between decisions. A stale inhabitation cost must
        # not hide a newly reusable one-call completion.
        if self.components is not None:
            self.cache.clear()
        full = partial
        calls = 0
        for path, expr, local, typ in walk_typed(partial, scope):
            if not (isinstance(expr, list) and expr[0] == 'hole'):
                continue
            child = self.close(typ, local)
            if child is None:
                return None
            full = replace(full, path, child.expression)
            calls += child.steps
        try:
            validate_shape(full)
            if holes(full) or infer(full, scope) != infer(partial, scope):
                return None
        except Invalid:
            return None
        return Plan(full, calls)

    def admissible(self, choices, partial, path, scope, remaining_steps):
        """Keep choices with a witness fitting AFTER this charged selection."""
        result = {}
        for label, term in choices.items():
            plan = self.fill(replace(partial, path, term), scope)
            if plan is not None and plan.steps <= remaining_steps - 1:
                result[label] = term
        return result
