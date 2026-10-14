"""Finite-token-trie decoding for an explicit, pre-checked action menu.

This is local constrained sampling, not an exact conditional distribution over
whole programs. The existing loader and unconstrained generate() are untouched.
No invalid output is repaired or interpreted as a different selected action.
"""
from __future__ import annotations
import random
import inspect
import time
from .backend import Generation


class ChoiceTrie:
    def __init__(self,tokenizer,labels):
        if not labels or len(labels)!=len(set(labels)):
            raise ValueError('Choice labels must be unique and nonempty')
        self.paths={}
        self.next_tokens={}
        special=set(getattr(tokenizer,'all_special_ids',[]))
        for label in labels:
            ids=tuple(tokenizer.encode(label,add_special_tokens=False))
            if not ids or any(i in special for i in ids):
                raise ValueError('Choice label has empty or special-token encoding: '+label)
            if tokenizer.decode(list(ids),skip_special_tokens=False).strip()!=label:
                raise ValueError('Tokenizer does not round-trip choice label: '+label)
            if ids in self.paths:
                raise ValueError('Two choices have the same token sequence')
            self.paths[ids]=label
            for i,token in enumerate(ids):
                self.next_tokens.setdefault(ids[:i],set()).add(token)
        self.max_tokens=max(map(len,self.paths))+1  # includes a forced terminal EOS

    def allowed(self,prefix,eos):
        prefix=tuple(prefix)
        allowed=set(self.next_tokens.get(prefix,()))
        if prefix in self.paths:
            allowed.add(eos)
        if not allowed:
            raise RuntimeError('Constrained decoder left the legal token trie')
        return sorted(allowed)

    def selected(self,tokens,eos):
        tokens=tuple(tokens)
        if not tokens or tokens[-1]!=eos or tokens[:-1] not in self.paths:
            raise RuntimeError('Constrained generation did not finish one exact action label plus EOS')
        return self.paths[tokens[:-1]]


def generate_choice(backend,*,system,user,seed,max_new_tokens,temperature,labels,trie=None):
    """Use the already-loaded HF model and its original prompt encoder.

    Fresh generation configuration prevents inherited stop strings, forced
    tokens, or beam settings from contradicting the finite action grammar.
    All output is checked again after decoding; there is no free-text fallback.
    """
    torch=backend.torch
    trie=trie or ChoiceTrie(backend.tokenizer,labels)
    if max_new_tokens<trie.max_tokens:
        raise ValueError('Not enough output budget for every offered action and EOS')
    if getattr(backend.model.config,'is_encoder_decoder',False):
        raise ValueError('Typed choices currently require a decoder-only text generation head')
    inputs=backend._encode(system,user)
    if 'past_key_values' in inputs or inputs['input_ids'].shape[0]!=1:
        raise ValueError('Typed choices require a fresh, single-item prompt')
    prompt_len=int(inputs['input_ids'].shape[-1])
    eos=getattr(backend.model.generation_config,'eos_token_id',None)
    if isinstance(eos,(list,tuple)):
        eos=eos[0] if eos else None
    if eos is None:
        eos=backend.tokenizer.eos_token_id
    if not isinstance(eos,int):
        raise ValueError('Tokenizer/model must define an EOS token')
    if any(eos in path for path in trie.paths):
        raise ValueError('EOS occurs inside an action label')
    pad=backend.tokenizer.pad_token_id
    if pad is None:
        pad=eos
    torch.manual_seed(seed)
    random.seed(seed)
    if backend.device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
    elif backend.device.type=='mps':
        torch.mps.manual_seed(seed)
    def allowed(batch_id,ids):
        if batch_id!=0:
            raise RuntimeError('Unexpected batch in constrained generation')
        return trie.allowed(ids[prompt_len:].tolist(),eos)
    config=backend.tfm.GenerationConfig(
        max_new_tokens=trie.max_tokens,min_new_tokens=0,
        do_sample=temperature>0,temperature=temperature if temperature>0 else 1.0,
        top_k=0,top_p=1.0,typical_p=1.0,num_beams=1,num_return_sequences=1,
        use_cache=True,eos_token_id=eos,pad_token_id=pad,
        bos_token_id=getattr(backend.tokenizer,'bos_token_id',None),
        forced_bos_token_id=None,forced_eos_token_id=None,
        repetition_penalty=1.0,renormalize_logits=True,
        return_dict_in_generate=False)
    extra={}
    if 'use_model_defaults' in inspect.signature(backend.model.generate).parameters:
        extra['use_model_defaults']=False
    start=time.perf_counter()
    with torch.inference_mode():
        out=backend.model.generate(**inputs,generation_config=config,
                                   prefix_allowed_tokens_fn=allowed,**extra)
    ids=out.sequences[0] if hasattr(out,'sequences') else out[0]
    tokens=ids[prompt_len:].tolist()
    selected=trie.selected(tokens,eos)
    raw=backend.tokenizer.decode(tokens,skip_special_tokens=True).strip()
    if raw!=selected:
        raise RuntimeError('Decoded action text does not match selected token path')
    return Generation(raw,prompt_len,len(tokens),time.perf_counter()-start)
