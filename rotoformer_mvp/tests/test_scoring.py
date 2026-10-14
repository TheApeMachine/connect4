from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from rotoformer.scoring import choose_candidate, score_prompt, CandidateScore


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 11, width: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.lm_head = nn.Linear(width, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        logits = self.lm_head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


def test_score_prompt_returns_finite_metrics() -> None:
    torch.manual_seed(11)
    model = TinyCausalLM()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    metrics = score_prompt(model, inputs)
    assert metrics["prompt_nll"] > 0
    assert metrics["perplexity"] > 1
    assert metrics["last_token_entropy"] > 0
    assert metrics["latency_seconds"] >= 0


def _score(k: int, nll: float, entropy: float, margin: float) -> CandidateScore:
    return CandidateScore(
        k=k,
        multiplier=1,
        translation=0,
        prompt_nll=nll,
        perplexity=1.0,
        last_token_entropy=entropy,
        top2_margin=margin,
        latency_seconds=0.0,
    )


def test_choose_candidate_objectives() -> None:
    values = [_score(0, 2.0, 1.0, 0.1), _score(3, 1.0, 2.0, 0.8)]
    assert choose_candidate(values, "prompt_nll").k == 3
    assert choose_candidate(values, "entropy").k == 0
    assert choose_candidate(values, "margin").k == 3
