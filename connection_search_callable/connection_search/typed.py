"""Type-checked structural menus built from the existing enumeration grammar.

No task names, reference evaluators, affine identities, or repaired model output
occur in this module. A menu is a bounded grammar frontier, not all mathematics.
"""
from __future__ import annotations
import random
from .dsl import Invalid,infer,key,replace,validate_shape
from .enumerate import Enumerator,EnumerationLimits


def menu(expected,scope,library,*,construction,path,components=(),excluded=(),seed=0,application_policy='unrestricted',callables=()):
    grammar=Enumerator(scope,expected,library,EnumerationLimits())
    blocked=set(excluded)
    proposals=list(grammar.one_step(expected,scope,application_policy=application_policy,callables=callables))
    proposals.extend(item['term'] for item in components)
    found={}
    for proposal in proposals:
        try:
            term=library.resolve(proposal)
            validate_shape(term)
            if infer(term,scope)!=expected:
                continue
            candidate=replace(construction,path,term)
            validate_shape(candidate)
            if key(candidate) in blocked or key(term)==key(_at(construction,path)):
                continue
            found.setdefault(key(term),term)
        except Invalid:
            continue
    terms=list(found.values())
    # Labels/positions do not permanently favor one operator. The entire mapping
    # is logged in the prompt, along with the explicit application policy.
    random.Random(seed).shuffle(terms)
    return {f'C{i:03d}':term for i,term in enumerate(terms)}


def _at(construction,path):
    from .dsl import at
    return at(construction,path)
