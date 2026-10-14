"""Fresh-call inference, strict output contracts, and auditable compute accounting."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import time
from typing import Any
from .dsl import Invalid


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    max_calls: int = 32
    max_tokens: int = 65536
    max_model_seconds: float = 0.0
    max_wall_seconds: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    model_seconds: float = 0.0
    preflight_seconds: float = 0.0
    started: float = 0.0

    def __post_init__(self):
        if self.max_calls<1 or self.max_tokens<1 or self.max_model_seconds<0 or self.max_wall_seconds<0:
            raise ValueError('Invalid budget')
        self.started=time.perf_counter()

    @property
    def tokens(self):
        return self.prompt_tokens+self.generated_tokens

    def ensure(self):
        if self.calls>=self.max_calls:
            raise BudgetExhausted('model_call_cap')
        if self.tokens>=self.max_tokens:
            raise BudgetExhausted('total_token_cap')
        if self.max_model_seconds and self.model_seconds>=self.max_model_seconds:
            raise BudgetExhausted('model_time_cap')
        self.ensure_wall()

    def ensure_wall(self):
        if self.max_wall_seconds and time.perf_counter()-self.started>=self.max_wall_seconds:
            raise BudgetExhausted('wall_time_cap')

    def to_dict(self):
        d=asdict(self)
        d.pop('started')
        d['total_tokens']=self.tokens
        d['elapsed_seconds']=time.perf_counter()-self.started
        d['time_cap_semantics']='Checked between operations; at most one indivisible model call may overshoot.'
        d['token_cap_semantics']='Exact preflight reservation of prompt + maximum output tokens.'
        d['equal_tokens_are_equal_flops']=False
        return d


class Log:
    def __init__(self,path:Path):
        self.path=path
        path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text('')
        self.events=[]

    def write(self,event:str,**fields):
        row={'index':len(self.events),'event':event,**fields}
        self.events.append(row)
        with self.path.open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        return row


def parse_reply(text:str,field:str) -> Any:
    """A whole JSON root only; do not salvage a nested object from broken output."""
    text=text.strip()
    if text.startswith('```') and text.endswith('```'):
        lines=text.splitlines()
        text='\n'.join(lines[1:-1]).strip()
    try:
        obj=json.loads(text)
    except (ValueError,RecursionError) as exc:
        raise Invalid('Expected one complete JSON object: '+str(exc),kind='format_error') from exc
    if not isinstance(obj,dict) or set(obj)!={field}:
        raise Invalid('Expected exactly one field: '+field,kind='format_error')
    return obj[field]


class Inference:
    def __init__(self,backend,budget:Budget,log:Log,seed=1729,max_new_tokens=1024,temperature=.8):
        if max_new_tokens<1 or temperature<0:
            raise ValueError('Invalid generation settings')
        self.backend,self.budget,self.log=backend,budget,log
        self.seed,self.max_new_tokens,self.temperature=seed,max_new_tokens,temperature
        self.last_call=None

    def count_prompt(self,system,user):
        if hasattr(self.backend,'_encode'):
            return int(self.backend._encode(system,user)['input_ids'].shape[-1])
        if hasattr(self.backend,'count_prompt_tokens'):
            return self.backend.count_prompt_tokens(system,user)
        raise TypeError('Backend must expose exact prompt counting; no character/token guesses for real runs')

    def synchronize(self):
        device=getattr(self.backend,'device',None)
        torch=getattr(self.backend,'torch',None)
        if torch is None or device is None:
            return
        if device.type=='cuda':
            torch.cuda.synchronize()
        elif device.type=='mps':
            torch.mps.synchronize()

    def ask(self,stage:str,payload:dict,field:str,*,choices=None):
        from .dsl import DSL_GUIDE
        self.last_call=None
        b=self.budget
        b.ensure()
        trie=None
        if choices is not None:
            from .constrained import ChoiceTrie
            if not choices:
                raise Invalid('No untried legal structural choices at this focus',kind='proposal_exhausted')
            payload={**payload,'legal_choices':[{'id':k,'term':v} for k,v in choices.items()]}
            payload['instruction']='Choose the legal structural step most useful for the task. Return its ID only. Unresolved holes are future decisions, not promises of correctness.'
            system=DSL_GUIDE.rsplit('Return exactly the requested JSON object.',1)[0]
            system+='\nReturn one legal_choices ID only, for example C000. No JSON, prose, or code.'
            trie=ChoiceTrie(self.backend.tokenizer,list(choices))
        else:
            system=DSL_GUIDE+'\n'+('Reply with exactly {"'+field+'": EXPRESSION}.')
        user=json.dumps(payload,separators=(',',':'),ensure_ascii=False)
        start=time.perf_counter()
        n=self.count_prompt(system,user)
        b.preflight_seconds+=time.perf_counter()-start
        allowed=min(self.max_new_tokens,b.max_tokens-b.tokens-n)
        if allowed<(trie.max_tokens if trie else 32):
            raise BudgetExhausted('insufficient_tokens_for_prompt_and_output')
        if trie:
            allowed=trie.max_tokens
        b.ensure()
        seed=self.seed+b.calls*104729
        b.calls+=1
        self.synchronize()
        start=time.perf_counter()
        try:
            kwargs=dict(system=system,user=user,seed=seed,
                        max_new_tokens=allowed,temperature=self.temperature)
            if choices is None:
                gen=self.backend.generate(**kwargs)
            elif hasattr(self.backend,'generate_choice'):
                gen=self.backend.generate_choice(**kwargs,labels=list(choices))
            else:
                from .constrained import generate_choice
                gen=generate_choice(self.backend,**kwargs,labels=list(choices),trie=trie)
            self.synchronize()
        except Exception as exc:
            b.model_seconds+=time.perf_counter()-start
            # Output token count on a failed inference is unknown. Reserve the cap and abort.
            b.prompt_tokens+=n
            b.generated_tokens+=allowed
            self.log.write('model_error',stage=stage,seed=seed,system=system,user=user,
                           error=repr(exc),tokens_charged_conservatively=n+allowed)
            raise RuntimeError('Model inference failed; no automatic backend/precision fallback. See events.jsonl') from exc
        elapsed=time.perf_counter()-start
        b.model_seconds+=elapsed
        b.prompt_tokens+=gen.prompt_tokens
        b.generated_tokens+=gen.generated_tokens
        if gen.prompt_tokens!=n or gen.generated_tokens>allowed:
            self.log.write('backend_contract_error',expected_prompt=n,actual_prompt=gen.prompt_tokens,
                           allowed_output=allowed,actual_output=gen.generated_tokens)
            raise RuntimeError('Backend token accounting violates its preflight contract')
        print(f'{stage:20} call={b.calls:3} prompt={n:5} output={gen.generated_tokens:4} seconds={elapsed:.2f}',flush=True)
        row=self.log.write('model_call',call_number=b.calls,stage=stage,seed=seed,system=system,user=user,raw=gen.text,
                          prompt_tokens=n,generated_tokens=gen.generated_tokens,max_new_tokens=allowed,
                          synchronized_model_seconds=elapsed,backend_reported_seconds=gen.seconds,
                          temperature=self.temperature,
                          decoding='typed_choice_trie' if choices is not None else 'unconstrained_json')
        self.last_call=row
        if choices is not None:
            if gen.text not in choices:
                self.log.write('backend_contract_error',message='Constrained backend returned an unknown choice ID',raw=gen.text)
                raise RuntimeError('Constrained backend escaped the legal menu; no repair/fallback')
            self.log.write('typed_choice',call_index=row['index'],selected=gen.text,
                           term=choices[gen.text],choice_count=len(choices),
                           syntax_and_type_checked=True,mathematical_correctness_established=False)
            return choices[gen.text]
        try:
            return parse_reply(gen.text,field)
        except Invalid as exc:
            self.log.write('rejected_format',call_index=row['index'],message=str(exc))
            raise
