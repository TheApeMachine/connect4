"""Deterministic controller-test backend. NOT an LLM and NOT a discovery result."""
from __future__ import annotations
import json
from .backend import Generation


class ReplayBackend:
    synthetic=True
    def __init__(self,replies):
        self.replies=list(replies)
        self.calls=[]

    def count_prompt_tokens(self,system,user):
        # Synthetic byte units, expressly not tokenizer/inference FLOP measurements.
        return len((system+user).encode())

    def generate(self,*,system,user,seed,max_new_tokens,temperature):
        payload=json.loads(user)
        self.calls.append(payload)
        if not self.replies:
            raise RuntimeError('Replay fixture exhausted')
        item=self.replies.pop(0)
        if isinstance(item,tuple):
            stage,response=item
            if payload['stage']!=stage:
                raise AssertionError(f'Replay expected {stage}, got {payload["stage"]}')
        else:
            response=item
        if callable(response):
            response=response(payload)
        raw=response if isinstance(response,str) else json.dumps(response,separators=(',',':'))
        # A long synthetic response obeys the same truncation contract.
        raw=raw.encode()[:max_new_tokens].decode(errors='ignore')
        return Generation(raw,self.count_prompt_tokens(system,user),len(raw.encode()),0.0)
