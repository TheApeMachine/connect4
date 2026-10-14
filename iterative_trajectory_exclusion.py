#!/usr/bin/env python3
"""Iterative trajectory exclusion: a falsifiable activation-level search experiment.

Same prompt, same weights, three paired arms:
  baseline : independent sampling with no activation modifications
  inhibit  : gate a bounded subtractive activation update towards the nearest
             previously visited, prompt-centered state (in this arm's history)
  random   : random direction using the inhibitor's realized magnitude at each
             matched forward step and layer (explicit replay schedule and audit)

Round zero is identical in all arms and provides the initial memory. All arms
use the same sampling seed each round, and repeat the *same* original prompt.
The technique is exploratory: hidden-state proximity is NOT semantic identity.

Install: pip install -U torch transformers safetensors accelerate
Example:
  python iterative_trajectory_exclusion.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --rounds 8 --max-new-tokens 220 --out runs/first
  python iterative_trajectory_exclusion.py --seed-policy fixed --rounds 8

Stored results: runs.jsonl, summary.json, candidates.md, traces/*.pt, config.json.
A .pt trace stores tensor states; only open trusted trace files with torch.load.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_TASK = (
    "Propose ONE concrete, falsifiable new idea for improving machine-learning "
    "systems by recombining already established mechanisms or results. "
    "Explain the constituent mechanisms, the precise interaction that makes "
    "the proposal work, a small experiment to test it, and the most likely "
    "reason it might fail. Do not make up experimental outcomes or claim "
    "verified historical novelty."
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    for name in (
        "model.layers", "model.model.layers", "transformer.h", "model.transformer.h",
        "model.language_model.layers", "model.language_model.model.layers",
        "language_model.model.layers", "gpt_neox.layers",
    ):
        obj: Any = model
        try:
            for key in name.split("."):
                obj = getattr(obj, key)
        except AttributeError:
            continue
        if isinstance(obj, (torch.nn.ModuleList, list, tuple)) and obj:
            return list(obj)
    raise RuntimeError(
        "Cannot find decoder layers. Use a standard text-only HF causal LM "
        "(Qwen2.5, Llama, SmolLM2, etc.), or extend decoder_layers()."
    )


def resolve_device(request: str) -> torch.device:
    if request != "auto":
        dev = torch.device(request)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    return dev


def choose_layers(layer_count: int, fractions: str) -> list[int]:
    vals = [float(x) for x in fractions.split(",") if x.strip()]
    if not vals or any(not 0 <= f <= 1 for f in vals):
        raise ValueError("--layers must be a comma-separated list of fractions in [0,1]")
    return sorted({max(0, min(layer_count - 1, round(f * (layer_count - 1)))) for f in vals})


def get_hidden(out: Any):
    """Hidden + output rebuilder for standard transformer block conventions."""
    if torch.is_tensor(out):
        return out, lambda h: h
    if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
        return out[0], lambda h: (h, *out[1:])
    if isinstance(out, list) and out and torch.is_tensor(out[0]):
        return out[0], lambda h: [h, *out[1:]]
    raise RuntimeError(f"Unsupported decoder block output type: {type(out).__name__}")


@dataclasses.dataclass
class Trace:
    arm: str
    round: int
    seed: int
    text: str
    states: dict[int, torch.Tensor]  # CPU [forward steps, hidden width], after any edit
    tokens: list[int]
    metrics: dict[str, Any]
    file: str = ""


class TrajectoryBank:
    """Per-arm rolling memory of states, prompt-centered at each layer.

    Each stored vector is a *token state*, not an averaged text embedding. When
    sampling, find the nearest previous visited state in the chosen layer, and
    subtract a bounded step in that reference direction. This is deliberately
    an activation-space proxy rather than a semantic-idea detector.
    """

    def __init__(self, max_runs: int, max_tokens_per_run: int, min_step: int = 1):
        self.items: collections.deque[Trace] = collections.deque(maxlen=max_runs)
        self.max_tokens_per_run = max_tokens_per_run
        self.min_step = min_step

    def add(self, trace: Trace) -> None:
        self.items.append(trace)

    def compiled(
        self, anchor: dict[int, torch.Tensor], device: torch.device, dtype: torch.dtype
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        result = {}
        for layer, base in anchor.items():
            pieces, positions = [], []
            for item in self.items:
                seq = item.states[layer][self.min_step:].float()
                if seq.numel() == 0:
                    continue
                if len(seq) > self.max_tokens_per_run:
                    indices = torch.linspace(0, len(seq) - 1, self.max_tokens_per_run).long()
                    positions.append(indices + self.min_step)
                    seq = seq[indices]
                else:
                    positions.append(torch.arange(self.min_step, self.min_step + len(seq)))
                relative = seq - base.float().unsqueeze(0)
                norms = relative.norm(dim=-1)
                valid = norms > 1e-5
                if valid.any():
                    pieces.append(F.normalize(relative[valid], dim=-1))
                    positions[-1] = positions[-1][valid]
                else:
                    positions.pop()
            if pieces:
                result[layer] = (
                    torch.cat(pieces).to(device=device, dtype=dtype),
                    torch.cat(positions).to(device=device),
                )
        return result


def rescale_rms(h: torch.Tensor, new_h: torch.Tensor) -> torch.Tensor:
    rms_old = h.float().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    rms_new = new_h.float().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    return new_h * (rms_old / rms_new).to(new_h.dtype)


def apply_bounded_perturbation(
    h: torch.Tensor, unit_direction: torch.Tensor, ratio: torch.Tensor, preserve_rms: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return edited state and actual ||delta||/||state||, post-RMS.

    Two correction steps help match the requested magnitude even when the
    direction has a large radial component in the residual stream.
    """
    size = h.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    delta = unit_direction * (ratio * size).to(h.dtype)
    for _ in range(2):
        candidate = h - delta
        if preserve_rms:
            candidate = rescale_rms(h, candidate)
        actual = (candidate.float() - h.float()).norm(dim=-1, keepdim=True)
        wanted = ratio.float() * size
        factor = torch.where(wanted > 1e-7, wanted / actual.clamp_min(1e-7), 0.0)
        delta = delta * factor.to(delta.dtype)
    updated = h - delta
    if preserve_rms:
        updated = rescale_rms(h, updated)
    achieved = (updated.float() - h.float()).norm(dim=-1, keepdim=True) / size
    return updated, achieved.squeeze(-1)


class TraceIntervention:
    def __init__(
        self, layers: list[torch.nn.Module], selected: list[int],
        arm: str, memory: dict[int, tuple[torch.Tensor, torch.Tensor]],
        anchor: dict[int, torch.Tensor], threshold: float, strength: float,
        cap: float, token_window: int, rng_seed: int, preserve_rms: bool = True,
        magnitude_schedule: dict[tuple[int, int], float] | None = None,
    ):
        self.layers = layers
        self.selected = selected
        self.arm = arm
        self.memory = memory
        self.anchor = anchor
        self.threshold = threshold
        self.strength = strength
        self.cap = cap
        self.token_window = token_window
        self.preserve_rms = preserve_rms
        self.magnitude_schedule = magnitude_schedule
        self.step = -1
        self.records: dict[int, list[torch.Tensor]] = {i: [] for i in selected}
        self.diagnostics: list[dict[str, Any]] = []  # device tensors until flush
        self._flushed: list[dict[str, Any]] | None = None
        self.handles = []
        # random generator is created lazily from the activation device;
        # anchors are stored on CPU regardless of inference backend.
        self.random_generator = None
        self.rng_seed = rng_seed

    def _random_direction(self, h: torch.Tensor) -> torch.Tensor:
        if self.random_generator is None:
            self.random_generator = torch.Generator(device=h.device)
            self.random_generator.manual_seed(self.rng_seed)
        return torch.randn(h.shape, dtype=h.dtype, device=h.device,
                           generator=self.random_generator)

    def _hook(self, index: int):
        def hook(_module, _args, output):
            hidden, replace = get_hidden(output)
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError("This initial implementation supports batch size one.")
            if index == self.selected[0]:
                self.step += 1
            # The first forward processes the shared prompt, and yields an anchor.
            # Steering starts on subsequent decode passes.
            step = self.step
            h = hidden[:, -1, :]
            if index not in self.anchor:
                self.anchor[index] = h[0].detach().float().cpu().clone()
            applied = False
            closest = None
            target = None
            actual = None
            projection_drop = None
            if self.arm != "baseline" and step > 0:
                direction = None
                ratio = None
                if self.arm == "random" and self.magnitude_schedule is not None:
                    # Replay the *inhibitor's realized post-RMS magnitude* for this
                    # token-step/layer. Extra random-continuation steps get no dose.
                    amount = min(self.cap, max(0.0, self.magnitude_schedule.get((step, index), 0.0)))
                    if amount > 0:
                        ratio = torch.tensor([[amount]], device=h.device, dtype=torch.float32)
                        direction = self._random_direction(h)
                elif index in self.memory:
                    previous, positions = self.memory[index]
                    center = self.anchor[index].to(device=h.device, dtype=h.dtype)
                    delta = h - center.unsqueeze(0)
                    # Avoid treating an identical prompt anchor as a new idea.
                    normalized = F.normalize(delta.float(), dim=-1)
                    sims = (previous.float() @ normalized[0]).flatten()
                    if self.token_window >= 0:
                        sims = sims.masked_fill((positions - step).abs() > self.token_window, -2.0)
                    best_sim, argmax = torch.max(sims, dim=0)
                    nearest = previous[argmax].unsqueeze(0)
                    gate = ((best_sim - self.threshold) / (1.0 - self.threshold)).clamp(0, 1)
                    ratio = (gate * self.strength).clamp(max=self.cap).reshape(1, 1)
                    closest = best_sim.detach()
                    if self.arm == "inhibit":
                        direction = nearest
                    else:
                        direction = self._random_direction(h)
                if direction is not None and ratio is not None:
                    direction = F.normalize(direction.float(), dim=-1).to(h.dtype)
                    new_h, achieved = apply_bounded_perturbation(
                        h, direction, ratio, self.preserve_rms
                    )
                    if self.arm == "inhibit" and closest is not None:
                        # Normalized reduction in activation projection onto
                        # the remembered path. Angular cosine alone is unsuitable:
                        # shrinking a vector radially does not change its cosine.
                        center = self.anchor[index].to(device=h.device, dtype=h.dtype)
                        projection_drop = (
                            ((h.float() - center.float().unsqueeze(0)) * direction.float()).sum(-1)
                            - ((new_h.float() - center.float().unsqueeze(0)) * direction.float()).sum(-1)
                        ).mean() / h.float().norm(dim=-1).mean().clamp_min(1e-6)
                    changed = hidden.clone()
                    changed[:, -1, :] = new_h
                    hidden = changed
                    output = replace(hidden)
                    target = ratio.detach().mean()
                    actual = achieved.detach().mean()
            # Keep traces on the accelerator during generation. Copy them in
            # bulk afterwards: .cpu() per layer/token causes severe MPS stalls.
            self.records[index].append(hidden[0, -1].detach().clone())
            self.diagnostics.append({
                "layer": index, "step": step,
                "closest_cosine": closest, "requested_delta_ratio": target,
                "actual_delta_ratio": actual, "projection_drop_ratio": projection_drop,
            })
            return output
        return hook

    def __enter__(self):
        for i in self.selected:
            self.handles.append(self.layers[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()
        self.handles = []

    def states(self) -> dict[int, torch.Tensor]:
        return {i: torch.stack(v).float().cpu() for i, v in self.records.items()}

    def diagnostics_cpu(self) -> list[dict[str, Any]]:
        if self._flushed is not None:
            return self._flushed
        # One scalar transfer rather than hundreds of per-token synchronizations.
        positions = []
        scalars = []
        for i, item in enumerate(self.diagnostics):
            for name in ("closest_cosine", "requested_delta_ratio", "actual_delta_ratio",
                         "projection_drop_ratio"):
                value = item[name]
                if value is not None:
                    positions.append((i, name))
                    scalars.append(value.reshape(()).float())
        scalars_cpu = torch.stack(scalars).detach().cpu().tolist() if scalars else []
        rows = [{"layer": int(item["layer"]), "step": int(item["step"]),
                 "closest_cosine": None, "requested_delta_ratio": None,
                 "actual_delta_ratio": None, "projection_drop_ratio": None,
                 "applied": False}
                for item in self.diagnostics]
        for (i, name), value in zip(positions, scalars_cpu):
            rows[i][name] = float(value)
        for row in rows:
            row["applied"] = bool((row["actual_delta_ratio"] or 0.0) > 1e-7)
        self._flushed = rows
        return rows

    def diagnostics_summary(self) -> dict[str, float | int | None]:
        applied = [x for x in self.diagnostics_cpu() if x["applied"]]
        if not applied:
            return {"forward_steps": max(0, self.step + 1), "edits": 0,
                    "mean_similarity": None, "mean_actual_delta_ratio": 0.0,
                    "max_actual_delta_ratio": 0.0, "mean_projection_drop_ratio": None}
        similarities = [x["closest_cosine"] for x in applied if x["closest_cosine"] is not None]
        drops = [x["projection_drop_ratio"] for x in applied if x["projection_drop_ratio"] is not None]
        return {
            "forward_steps": self.step + 1,
            "edits": len(applied),
            "mean_similarity": (sum(similarities) / len(similarities)) if similarities else None,
            "mean_projection_drop_ratio": (sum(drops) / len(drops)) if drops else None,
            "mean_actual_delta_ratio": sum(x["actual_delta_ratio"] for x in applied) / len(applied),
            "max_actual_delta_ratio": max(x["actual_delta_ratio"] for x in applied),
        }


def candidate_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
    return {tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1))}


def candidate_quality_checks(text: str, eos_reached: bool, max_tokens_hit: bool) -> dict[str, Any]:
    """Cheap observable diagnostics, explicitly NOT semantic or scientific quality."""
    words = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
    return {
        "chars": len(text), "words": len(words),
        "repeated_bigram_fraction": round(1 - len(candidate_ngrams(text, 2)) / max(1, len(words)-1), 4),
        "replacement_characters": text.count("\ufffd"),
        "eos_reached": eos_reached,
        "max_tokens_hit": max_tokens_hit,
    }


def novelty_metrics(text: str, earlier: list[str]) -> dict[str, Any]:
    target = candidate_ngrams(text)
    prior_sets = [candidate_ngrams(x) for x in earlier]
    known = set().union(*prior_sets) if prior_sets else set()
    max_jaccard = max((len(target & old) / max(1, len(target | old)) for old in prior_sets), default=0)
    return {
        "new_trigrams": len(target - known),
        "new_trigram_fraction": round(len(target - known) / max(1, len(target)), 4),
        "max_previous_trigram_jaccard": round(max_jaccard, 4),
        "exact_repeat": any(text.strip() == x.strip() for x in earlier),
    }


def encode_prompt(tokenizer, prompt: str, use_chat: bool) -> dict[str, torch.Tensor]:
    if use_chat and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
    return tokenizer(prompt, return_tensors="pt")


def decode_output(tokenizer, continuation: torch.Tensor) -> str:
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def run_generation(
    model, tokenizer, inputs: dict[str, torch.Tensor], layers: list[torch.nn.Module],
    selected: list[int], arm: str, round_id: int, seed: int,
    bank: TrajectoryBank, anchor_cpu: dict[int, torch.Tensor], args,
    magnitude_schedule: dict[tuple[int, int], float] | None = None,
) -> tuple[Trace, list[dict[str, Any]]]:
    seed_all(seed)
    model_device = inputs["input_ids"].device
    # Anchor itself stays on CPU between runs; hooks convert it as needed.
    cache = bank.compiled(anchor_cpu, model_device, next(model.parameters()).dtype) if anchor_cpu else {}
    with TraceIntervention(
        layers, selected, arm, cache, anchor_cpu, args.similarity_threshold,
        args.strength, args.max_delta_ratio, args.step_window, seed + 1234567,
        preserve_rms=not args.no_preserve_rms,
        magnitude_schedule=magnitude_schedule,
    ) as probe:
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                **({"temperature": args.temperature, "top_p": args.top_p}
                   if args.temperature > 0 else {}),
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
    ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
    continuation = ids[inputs["input_ids"].shape[1]:].detach().cpu()
    text = decode_output(tokenizer, continuation)
    eos_values = tokenizer.eos_token_id
    eos_ids = [eos_values] if isinstance(eos_values, int) else list(eos_values or [])
    eos_reached = bool(eos_ids and any(int(t) in eos_ids for t in continuation.tolist()))
    diagnostics = probe.diagnostics_summary()
    trace = Trace(
        arm=arm, round=round_id, seed=seed, text=text,
        states=probe.states(), tokens=continuation.tolist(),
        metrics={**candidate_quality_checks(text, eos_reached, len(continuation) >= args.max_new_tokens),
                 **diagnostics},
    )
    return trace, probe.diagnostics_cpu()


def save_trace(output_dir: Path, trace: Trace, diagnostics: list[dict[str, Any]]) -> str:
    fname = f"{trace.arm}-{trace.round:02d}.pt"
    filepath = output_dir / "traces" / fname
    torch.save({
        "arm": trace.arm, "round": trace.round, "seed": trace.seed,
        "text": trace.text, "states": trace.states, "token_ids": trace.tokens,
        "intervention_diagnostics": diagnostics,
    }, filepath)
    return str(filepath.relative_to(output_dir))


def pretty_md(rows: list[dict[str, Any]]) -> str:
    out = ["# Iterative trajectory exclusion candidates", "",
           "Every arm replays the identical original prompt. The first run is shared.",
           "Read scores as lexical/behavioral diagnostics, NOT as idea quality.", ""]
    for row in rows:
        out += [f"## {row['arm']} / round {row['round']} / seed {row['seed']}",
                f"Trace: `{row['trace_file']}` · edits: {row['edits']} · "
                f"new trigram fraction: {row['new_trigram_fraction']:.3f} · "
                f"max overlap: {row['max_previous_trigram_jaccard']:.3f}", "",
                row["text"], "", "---", ""]
    return "\n".join(out)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ("baseline", "inhibit", "random"):
        group = [r for r in rows if r["arm"] == arm and r["round"] > 0]
        if not group:
            continue
        result[arm] = {
            "additional_rounds": len(group),
            "exact_repeat_rate": round(sum(int(r["exact_repeat"]) for r in group) / len(group), 4),
            "avg_new_trigram_fraction": round(sum(r["new_trigram_fraction"] for r in group) / len(group), 4),
            "avg_max_previous_trigram_jaccard": round(
                sum(r["max_previous_trigram_jaccard"] for r in group) / len(group), 4),
            "avg_response_words": round(sum(r["words"] for r in group) / len(group), 1),
            "avg_edits": round(sum(r["edits"] for r in group) / len(group), 1),
            "avg_projection_drop_ratio": (
                round(sum(r["mean_projection_drop_ratio"] for r in group
                          if r["mean_projection_drop_ratio"] is not None)
                      / len([r for r in group if r["mean_projection_drop_ratio"] is not None]), 6)
                if any(r["mean_projection_drop_ratio"] is not None for r in group) else None),
            "avg_repeated_bigram_fraction": round(
                sum(r["repeated_bigram_fraction"] for r in group) / len(group), 4),
        }
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--out", default="trajectory_exclusion_run")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--rounds", type=int, default=8, help="generations per arm, including shared seed run")
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--seed-policy", choices=["varied", "fixed"], default="varied")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--layers", default="0.40,0.60,0.80", help="normalized decoder depths")
    p.add_argument("--strength", type=float, default=0.018, help="max proposed intervention as fraction of state norm")
    p.add_argument("--max-delta-ratio", type=float, default=0.025)
    p.add_argument("--similarity-threshold", type=float, default=0.75)
    p.add_argument("--step-window", type=int, default=-1, help="nearest bank tokens within +/- N decode steps; -1 means any step")
    p.add_argument("--bank-runs", type=int, default=8)
    p.add_argument("--bank-tokens-per-run", type=int, default=80)
    p.add_argument("--no-preserve-rms", action="store_true")
    p.add_argument("--plain-prompt", action="store_true", help="ignore chat template")
    args = p.parse_args(argv)
    if args.rounds < 2 or args.max_new_tokens < 8 or args.bank_runs < 1 or args.bank_tokens_per_run < 1:
        p.error("rounds >= 2, max-new-tokens >= 8, bank-runs >= 1, bank-tokens >= 1 required")
    if not 0 <= args.strength <= args.max_delta_ratio <= 0.2:
        p.error("require 0 <= strength <= max-delta-ratio <= 0.2")
    if not 0 <= args.similarity_threshold < 1:
        p.error("similarity-threshold must be in [0,1)")
    if not 0 <= args.top_p <= 1 or args.temperature < 0:
        p.error("invalid sampling settings")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Late import makes --help and unit testing possible without transformers installed.
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install dependencies: pip install -U torch transformers accelerate safetensors") from exc

    output_dir = Path(args.out)
    if (output_dir / "runs.jsonl").exists():
        raise SystemExit(f"Refusing to append to existing run: {output_dir / 'runs.jsonl'} "
                         "(choose a new --out directory)")
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = resolve_device(args.device)
    dtype = (torch.float32 if device.type == "cpu" else
             torch.float16 if device.type == "mps" else torch.bfloat16)
    if args.dtype != "auto":
        dtype = getattr(torch, args.dtype)
    print(f"Loading {args.model} on {device} ({dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad token nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()
    layers = decoder_layers(model)
    selected = choose_layers(len(layers), args.layers)
    print(f"Selected decoder layers: {selected}", flush=True)
    encoded = encode_prompt(tokenizer, args.task, not args.plain_prompt)
    inputs = {k: v.to(device) for k, v in encoded.items() if torch.is_tensor(v)}
    anchor_cpu: dict[int, torch.Tensor] = {}
    banks = {arm: TrajectoryBank(args.bank_runs, args.bank_tokens_per_run)
             for arm in ("inhibit", "random")}
    history: dict[str, list[str]] = {arm: [] for arm in ("baseline", "inhibit", "random")}
    rows: list[dict[str, Any]] = []
    logfile = output_dir / "runs.jsonl"

    def record(trace: Trace, diagnostics, shared_file: str | None = None):
        trace.file = shared_file or save_trace(output_dir, trace, diagnostics)
        before = history[trace.arm]
        metrics = {**trace.metrics, **novelty_metrics(trace.text, before)}
        row = {"arm": trace.arm, "round": trace.round, "seed": trace.seed,
               "trace_file": trace.file, "text": trace.text, **metrics}
        rows.append(row)
        history[trace.arm].append(trace.text)
        with logfile.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{trace.arm:>8} {trace.round:2} | {len(trace.tokens):3} tokens | "
              f"{row['edits']:4} edits | novel-3={row['new_trigram_fraction']:.2f} "
              f"| overlap={row['max_previous_trigram_jaccard']:.2f} | "
              f"{trace.text[:95].replace(chr(10), ' ')}", flush=True)
        return row

    try:
        for n in range(args.rounds):
            seed = args.seed if args.seed_policy == "fixed" else args.seed + n
            baseline, base_diag = run_generation(
                model, tokenizer, inputs, layers, selected, "baseline", n, seed,
                TrajectoryBank(1, 1), anchor_cpu, args,
            )
            record(baseline, base_diag)
            if n == 0:
                for arm in ("inhibit", "random"):
                    shared = dataclasses.replace(baseline, arm=arm)
                    record(shared, [], shared_file=baseline.file)
                    banks[arm].add(shared)
                continue
            for arm in ("inhibit", "random"):
                # Inhibit first; then the random arm replays its exact perturbation
                # magnitude schedule. This avoids confounding direction with dose.
                schedule = ({(int(d["step"]), int(d["layer"])): float(d["actual_delta_ratio"] or 0.0)
                             for d in inhibit_diagnostics} if arm == "random" else None)
                trial, diag = run_generation(
                    model, tokenizer, inputs, layers, selected, arm, n, seed,
                    banks[arm], anchor_cpu, args, magnitude_schedule=schedule,
                )
                if arm == "inhibit":
                    inhibit_diagnostics = diag
                if arm == "random":
                    actual_schedule = {(int(d["step"]), int(d["layer"])):
                                       float(d["actual_delta_ratio"] or 0.0) for d in diag}
                    covered = set(schedule) & set(actual_schedule)
                    errors = [abs(actual_schedule[k] - schedule[k]) for k in covered]
                    trial.metrics["matched_dose_coverage"] = round(
                        len(covered) / max(1, len(schedule)), 4)
                    trial.metrics["max_absolute_dose_error"] = max(errors, default=0.0)
                    if trial.metrics["matched_dose_coverage"] < .95 or trial.metrics["max_absolute_dose_error"] > .003:
                        print("WARNING: random-control magnitude replay failed audit", file=sys.stderr)
                record(trial, diag)
                banks[arm].add(trial)
    finally:
        (output_dir / "candidates.md").write_text(pretty_md(rows), encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
    print("\nSUMMARY (exploration diagnostics only):")
    print(json.dumps(summarize(rows), indent=2))
    print(f"\nSaved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
