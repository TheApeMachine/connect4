"""Independent verification. A search-test pass is never named a theorem proof."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import time
from typing import Any
from .dsl import Invalid, Runtime, holes, infer, plain, validate_shape
from .algebra import scan_laws
from .tasks import Task


@dataclass
class Check:
    status: str
    message: str = ''
    path: tuple = ()
    evidence: Any = None
    certificates: list = field(default_factory=list)
    passed_cases: int = 0
    evaluated_cases: int = 0
    max_span: int = 0
    max_work: int = 0
    seconds: float = 0.0

    @property
    def passed(self):
        return self.status in ('passed_search_tests','passed_holdout_tests')

    def to_dict(self):
        return asdict(self)


class Checker:
    def __init__(self,task:Task,seed:int=1729):
        self.task,self.seed=task,seed
        self.calls=0
        self.case_evaluations=0
        self.seconds=0.0

    def check(self,expr,split='search') -> Check:
        start=time.perf_counter()
        result=Check('unexamined')
        self.calls+=1
        try:
            validate_shape(expr)
            t=infer(expr,self.task.inputs)
            if t!=self.task.output:
                raise Invalid('Program output type does not match immutable task output')
            result.certificates=scan_laws(expr,self.task.inputs)
            if holes(expr):
                result.status='partial_type_checked'
                result.message='Typed holes remain; this is not an executed task solution.'
                return result
            for i,case in enumerate(self.task.cases(split,self.seed)):
                result.evaluated_cases+=1
                self.case_evaluations+=1
                rt=Runtime(case.modulus)
                inputs={n:rt.wrap(case.inputs[n],t) for n,t in self.task.input_items}
                value=rt.eval(expr,inputs)
                actual=plain(value)
                expected=self.task.reference(case)
                result.max_span=max(result.max_span,value.ready)
                result.max_work=max(result.max_work,rt.work)
                if actual!=expected:
                    raise Invalid('Output differs from independent reference',kind='output_counterexample',
                                  evidence={'case_index':i,'inputs':case.inputs,'modulus':case.modulus,
                                            'expected':expected,'actual':actual})
                bound=self.task.span_bound(case.size)
                if bound is not None and value.ready>bound:
                    raise Invalid('Correct output but dependency-span constraint failed',kind='constraint_counterexample',
                                  evidence={'case_index':i,'inputs':case.inputs,'modulus':case.modulus,
                                            'n':case.size,'span':value.ready,'maximum_span':bound,
                                            'output_correct':True})
                result.passed_cases+=1
            result.status='passed_'+split+'_tests'
            result.message='Passed specified finite tests; whole-program universal correctness and historical novelty are unproved.'
        except Invalid as exc:
            result.status,result.message,result.path,result.evidence=exc.kind,str(exc),exc.path,exc.evidence
        finally:
            result.seconds=time.perf_counter()-start
            self.seconds+=result.seconds
        return result


def prompt_feedback(c:Check) -> dict:
    """Keep full counterexamples in logs; shorten large arrays only for model context."""
    d=c.to_dict()
    d.pop('certificates',None)
    def shorten(v):
        if isinstance(v,list) and len(v)>12:
            return {'truncated':True,'length':len(v),'first':v[:8],'last':v[-2:]}
        if isinstance(v,dict):
            return {k:shorten(x) for k,x in v.items()}
        return v
    return shorten(d)
