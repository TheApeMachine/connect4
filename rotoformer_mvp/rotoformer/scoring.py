"""Inference-time scoring and brute-force rotation search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Iterable, Literal, Mapping

import torch
from torch import Tensor, nn

from .adapter import RotoformerAdapter

Objective = Literal["prompt_nll", "entropy", "margin"]


@dataclass
class CandidateScore:
    k: int
    multiplier: int
    translation: int
    prompt_nll: float
    perplexity: float
    last_token_entropy: float
    top2_margin: float
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_logits(outputs: Any) -> Tensor:
    logits = getattr(outputs, "logits", None)
    if torch.is_tensor(logits):
        return logits
    if isinstance(outputs, tuple) and outputs and torch.is_tensor(outputs[0]):
        return outputs[0]
    raise TypeError("model output did not expose causal-LM logits")


def _extract_loss(outputs: Any) -> Tensor | None:
    loss = getattr(outputs, "loss", None)
    if torch.is_tensor(loss):
        return loss
    return None


def _manual_causal_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.shape[1] < 2:
        raise ValueError("prompt scoring requires at least two tokens")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _last_positions(attention_mask: Tensor | None, sequence_length: int, device: torch.device) -> Tensor:
    if attention_mask is None:
        return torch.full((1,), sequence_length - 1, dtype=torch.long, device=device)
    return attention_mask.long().sum(dim=-1).clamp_min(1) - 1


def score_prompt(model: nn.Module, inputs: Mapping[str, Tensor]) -> dict[str, float]:
    """Score one prompt without generating.

    ``prompt_nll`` is teacher-forced next-token loss over the prompt itself. The
    entropy and margin are measured at the final non-padding prompt position.
    """

    if "input_ids" not in inputs:
        raise ValueError("inputs must contain input_ids")

    labels = inputs["input_ids"].clone()
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)

    forward_inputs = dict(inputs)
    forward_inputs["labels"] = labels
    forward_inputs["use_cache"] = False

    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**forward_inputs)
    latency = time.perf_counter() - started

    logits = _extract_logits(outputs)
    loss = _extract_loss(outputs)
    if loss is None:
        loss = _manual_causal_loss(logits, labels)

    batch = logits.shape[0]
    positions = _last_positions(attention_mask, logits.shape[1], logits.device)
    if positions.numel() == 1 and batch > 1:
        positions = positions.expand(batch)
    row = torch.arange(batch, device=logits.device)
    last_logits = logits[row, positions]

    log_probs = torch.log_softmax(last_logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    top2 = torch.topk(last_logits.float(), k=2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).mean()

    nll = float(loss.detach().float().cpu())
    perplexity = math.exp(min(20.0, nll))
    return {
        "prompt_nll": nll,
        "perplexity": perplexity,
        "last_token_entropy": float(entropy.detach().cpu()),
        "top2_margin": float(margin.detach().cpu()),
        "latency_seconds": latency,
    }


def sweep_rotations(
    model: nn.Module,
    adapter: RotoformerAdapter,
    inputs: Mapping[str, Tensor],
    rotations: Iterable[int],
    *,
    progress: bool = True,
) -> list[CandidateScore]:
    candidates = [int(k) for k in rotations]
    if not candidates:
        raise ValueError("rotation sweep is empty")

    results: list[CandidateScore] = []
    total = len(candidates)
    for index, k in enumerate(candidates, start=1):
        adapter.set_rotation(k)
        metrics = score_prompt(model, inputs)
        spec = adapter.current_spec
        result = CandidateScore(
            k=k,
            multiplier=spec.multiplier,
            translation=spec.translation,
            prompt_nll=metrics["prompt_nll"],
            perplexity=metrics["perplexity"],
            last_token_entropy=metrics["last_token_entropy"],
            top2_margin=metrics["top2_margin"],
            latency_seconds=metrics["latency_seconds"],
        )
        results.append(result)
        if progress:
            print(
                f"[{index:>3}/{total}] k={k:<5} a={result.multiplier:<4} "
                f"nll={result.prompt_nll:.6f} entropy={result.last_token_entropy:.5f} "
                f"margin={result.top2_margin:.5f}",
                flush=True,
            )
    return results


def choose_candidate(results: Iterable[CandidateScore], objective: Objective) -> CandidateScore:
    values = list(results)
    if not values:
        raise ValueError("no candidate scores available")
    if objective == "prompt_nll":
        return min(values, key=lambda item: (item.prompt_nll, item.k))
    if objective == "entropy":
        return min(values, key=lambda item: (item.last_token_entropy, item.k))
    if objective == "margin":
        return max(values, key=lambda item: (item.top2_margin, -item.k))
    raise ValueError(f"unsupported objective: {objective!r}")
