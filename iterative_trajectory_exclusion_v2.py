#!/usr/bin/env python3
"""Iterative trajectory exclusion v2: ordered-window activation search.

Research question: when a frozen LLM repeatedly sees the SAME prompt and seed,
can repelling it from previously generated ordered activation trajectories
reliably reveal different, useful solution paths?

Arms, run sequentially with identical prompt and paired sampling seeds:
    baseline : no steering (but measure proximity to excluded history)
    inhibit  : compare recent hidden-state WINDOWS (states + transitions) with
               windows from its earlier outputs; subtract a bounded matched
               transition direction when the ordered-window score is high
    shuffled : same realized inhibitor perturbation dose, but directional
               targets are matched to TIME-SHUFFLED versions of the very same
               inhibitor history (temporal-order ablation)
    random   : same realized dose, but random directions (noise control)

Round 0 is one common baseline sample and initializes inhibitor history.
Later rounds append ONLY the inhibitor's own trajectories to the shared bank.
The two controls read the exact same pre-round history as the inhibitor.
All four arms replay the exact same input tokens. v2 defaults to fixed seed.

The matching score is a weighted average of (a) cosine similarity of
prompt-centered state sequences and (b) cosine similarity of state-to-state
transitions. WINDOWS preserve temporal order; reference windows are contiguous
in the ORIGINAL sequence, and only then subsampled for bounded memory.

This is an activation-level EXPERIMENT, not semantic identification of ideas.
Changing a hidden-state path does not establish creativity or usefulness.

Install: pip install -U torch transformers accelerate safetensors
Gemma 4 needs a current Transformers version; its preferred loader is
AutoModelForMultimodalLM + AutoProcessor. Run:

  python iterative_trajectory_exclusion_v2.py \
    --model google/gemma-4-12B-it --device mps --rounds 8 \
    --max-new-tokens 800 --out runs/gemma_v2

  python iterative_trajectory_exclusion_v2.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --rounds 4 \
    --max-new-tokens 320 --out runs/qwen_v2

Output: config.json, runs.jsonl, candidates.md, summary.json, traces/*.pt.
Only torch.load .pt traces from trusted sources.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping

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
ARMS = ("baseline", "inhibit", "shuffled", "random")


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch, "mps", None) is not None:
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_layers(n: int, fractions: str) -> list[int]:
    values = [float(v.strip()) for v in fractions.split(",") if v.strip()]
    if not values or any(not 0 <= v <= 1 for v in values):
        raise ValueError("--layers must be comma-separated normalized depths in [0,1]")
    return sorted(set(max(0, min(n - 1, round((n - 1) * v))) for v in values))


def decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    for path in (
        "model.layers", "model.model.layers", "model.language_model.layers",
        "model.language_model.model.layers", "language_model.layers",
        "language_model.model.layers", "transformer.h", "model.transformer.h",
        "gpt_neox.layers", "decoder.layers",
    ):
        target: Any = model
        try:
            for part in path.split("."):
                target = getattr(target, part)
        except AttributeError:
            continue
        if isinstance(target, (torch.nn.ModuleList, list, tuple)) and len(target):
            return list(target)
    # Safest fallback is a ModuleList with transformer-like block type names.
    options = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module):
            score = sum(k in type(module[0]).__name__.lower() for k in ("decoder", "layer", "block"))
            options.append((score, len(module), name, list(module)))
    if options:
        best = max(options, key=lambda row: row[:2])
        if best[0] > 0:
            return best[3]
    raise RuntimeError("Cannot locate decoder blocks for activation hooks")


def hidden_output(value: Any):
    """Extract block hidden tensor and its safe replacement constructor."""
    if isinstance(value, torch.Tensor):
        return value, lambda h: h
    if isinstance(value, tuple) and len(value) and isinstance(value[0], torch.Tensor):
        return value[0], lambda h: (h, *value[1:])
    if isinstance(value, list) and len(value) and isinstance(value[0], torch.Tensor):
        return value[0], lambda h: [h, *value[1:]]
    raise TypeError(f"Unsupported hooked layer output: {type(value).__name__}")


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def preserve_state_rms(previous: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    original = previous.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    new = next_state.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    return next_state * (original / new).to(next_state.dtype)


def bounded_edit(
    state: torch.Tensor, direction: torch.Tensor, target_ratio: torch.Tensor,
    preserve_rms: bool, max_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edit the last position, approximately matching post-RMS ||delta||/||state||."""
    size = state.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ratio = target_ratio.float().clamp(0, max_ratio).reshape(-1, 1)
    # For RMS-preserved edits, radial perturbations get scaled away. Project
    # ALL arms onto the tangent space of the current state. Otherwise a
    # matched-dose shuffled direction can be nearly radial and falsely fail
    # the control audit while having the same requested magnitude.
    radial = state.float() / size
    tangential = normalize(direction) - (normalize(direction) * radial).sum(-1, keepdim=True) * radial
    fallback = torch.roll(radial, 1, dims=-1)
    fallback = fallback - (fallback * radial).sum(-1, keepdim=True) * radial
    tangential = torch.where(tangential.norm(dim=-1, keepdim=True) > 1e-5,
                             tangential, fallback)
    direction = normalize(tangential)
    delta = direction.to(state.dtype) * (ratio * size).to(state.dtype)
    for _ in range(3):
        candidate = state - delta
        if preserve_rms:
            candidate = preserve_state_rms(state, candidate)
        actual = (state.float() - candidate.float()).norm(dim=-1, keepdim=True)
        correction = torch.where(ratio > 1e-8, ratio * size / actual.clamp_min(1e-8), 0.0)
        delta = delta * correction.to(delta.dtype)
    candidate = state - delta
    if preserve_rms:
        candidate = preserve_state_rms(state, candidate)
    achieved = (state.float() - candidate.float()).norm(dim=-1) / size.squeeze(-1)
    return candidate, achieved


@dataclasses.dataclass
class Trace:
    arm: str
    round: int
    seed: int
    text: str
    states: dict[int, torch.Tensor]  # [forward steps, width], post-intervention
    tokens: list[int]
    metrics: dict[str, Any]
    file: str = ""


@dataclasses.dataclass
class ReferenceWindows:
    states: torch.Tensor      # [N, W * hidden_dim], each vector normalized separately
    transitions: torch.Tensor # [N, (W-1) * hidden_dim], normalized transitions
    directions: torch.Tensor  # [N, hidden_dim] terminal displacement directions
    positions: torch.Tensor   # [N] generation forward-step of window's terminal state

    def __len__(self) -> int:
        return self.directions.shape[0]


class TrajectoryBank:
    """Rolling ordered histories; store full runs, compile contiguous windows."""

    def __init__(self, max_runs: int, windows_per_run: int, window: int):
        self.traces: collections.deque[Trace] = collections.deque(maxlen=max_runs)
        self.windows_per_run = windows_per_run
        self.window = window

    def add(self, trace: Trace) -> None:
        self.traces.append(trace)

    def compile(
        self, anchors: Mapping[int, torch.Tensor], device: torch.device,
        dtype: torch.dtype, shuffled: bool = False, shuffle_seed: int = 1234,
    ) -> dict[int, ReferenceWindows]:
        out: dict[int, ReferenceWindows] = {}
        for layer, anchor in anchors.items():
            state_chunks, transition_chunks, directions, positions = [], [], [], []
            for trace_index, trace in enumerate(self.traces):
                # Step 0 is prefill's final prompt state. Only use decode steps.
                sequence = trace.states[layer][1:].float().cpu()
                if len(sequence) < self.window:
                    continue
                if shuffled:
                    g = torch.Generator(device="cpu").manual_seed(
                        shuffle_seed + trace_index * 104729 + layer * 9176
                    )
                    sequence = sequence[torch.randperm(len(sequence), generator=g)]
                total_windows = len(sequence) - self.window + 1
                n_take = min(total_windows, self.windows_per_run)
                # Select contiguous windows at evenly spaced START points.
                starts = torch.linspace(0, total_windows - 1, n_take).long().unique()
                frames = torch.stack([sequence[i:i + self.window] for i in starts.tolist()])
                if not shuffled:
                    # Full trace index = [original step 1 + start ... + window - 1].
                    positions.append(starts + self.window)
                else:
                    positions.append(starts + self.window)
                relative = normalize(frames - anchor.float().cpu().reshape(1, 1, -1))
                velocity = normalize(frames[:, 1:] - frames[:, :-1])
                displacement = normalize(frames[:, -1] - frames[:, -2])
                state_chunks.append(relative.reshape(len(frames), -1))
                transition_chunks.append(velocity.reshape(len(frames), -1))
                directions.append(displacement)
            if state_chunks:
                out[layer] = ReferenceWindows(
                    states=torch.cat(state_chunks).to(device=device, dtype=dtype),
                    transitions=torch.cat(transition_chunks).to(device=device, dtype=dtype),
                    directions=torch.cat(directions).to(device=device, dtype=dtype),
                    positions=torch.cat(positions).to(device=device),
                )
        return out


def make_query(history: list[torch.Tensor], now: torch.Tensor, anchor: torch.Tensor,
               window: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    # history includes the prompt anchor at its first element.
    if len(history) < window:
        return None
    tail = torch.stack(history[-(window - 1):] + [now.reshape(-1)], dim=0)
    state = normalize(tail - anchor.reshape(1, -1)).reshape(-1)
    velocity = normalize(tail[1:] - tail[:-1]).reshape(-1)
    return state, velocity


def match_window(refs: ReferenceWindows, query: tuple[torch.Tensor, torch.Tensor],
                 transition_weight: float, current_step: int, step_window: int,
                 window: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qstates, qtrans = query
    dtype = refs.states.dtype
    score = (
        (1 - transition_weight) * (refs.states @ qstates.to(dtype)) / window
        + transition_weight * (refs.transitions @ qtrans.to(dtype)) / (window - 1)
    ).float()
    if step_window >= 0:
        score = score.masked_fill((refs.positions - current_step).abs() > step_window, -2.0)
    best, best_index = score.max(dim=0)
    return best, best_index, refs.directions[best_index].reshape(1, -1)


def score_at_reference(
    refs: ReferenceWindows, index: torch.Tensor,
    query: tuple[torch.Tensor, torch.Tensor], transition_weight: float,
    window: int,
) -> torch.Tensor:
    states, transitions = query
    return (
        (1-transition_weight) * (refs.states[index].float() @ states.float()) / window
        + transition_weight * (refs.transitions[index].float() @ transitions.float()) / (window-1)
    )


class TrajectoryProbe:
    """Generation hooks that record post-edit states and evaluate path similarity."""

    def __init__(
        self, layers: list[torch.nn.Module], selected: list[int], arm: str,
        true_bank: dict[int, ReferenceWindows],
        shuffled_bank: dict[int, ReferenceWindows],
        anchors: dict[int, torch.Tensor], args: argparse.Namespace,
        seed: int, dose_schedule: Mapping[tuple[int, int], float] | None = None,
    ):
        self.layers, self.selected, self.arm = layers, selected, arm
        self.true_bank, self.shuffled_bank = true_bank, shuffled_bank
        self.anchors, self.args = anchors, args
        self.seed = seed
        self.dose_schedule = dose_schedule or {}
        self.step = -1
        self.states: dict[int, list[torch.Tensor]] = {i: [] for i in selected}
        self.diag: list[dict[str, Any]] = []
        self.handles = []
        self.random_generators: dict[str, torch.Generator] = {}
        self._diag_cpu: list[dict[str, Any]] | None = None

    def _random(self, h: torch.Tensor) -> torch.Tensor:
        key = str(h.device)
        if key not in self.random_generators:
            self.random_generators[key] = torch.Generator(device=h.device)
            self.random_generators[key].manual_seed(self.seed + 778899)
        return torch.randn(h.shape, generator=self.random_generators[key],
                           device=h.device, dtype=h.dtype)

    def _hook(self, layer: int):
        def hook(_module: Any, _args: tuple[Any, ...], value: Any) -> Any:
            hidden, rebuild = hidden_output(value)
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError("This experiment requires decoder output [1, seq, hidden]")
            if layer == self.selected[0]:
                self.step += 1
            step = self.step
            h = hidden[:, -1, :]
            if layer not in self.anchors:
                self.anchors[layer] = h[0].detach().float().cpu().clone()
            anchor = self.anchors[layer].to(device=h.device, dtype=h.dtype)

            query = make_query(self.states[layer], h[0], anchor, self.args.window)
            true_score = None
            reference_index = None
            chosen_direction = None
            shuffled_score = None
            gate = None
            before_score = None
            after_score = None
            requested = None
            actual = None
            if step > 0 and query is not None and layer in self.true_bank:
                true_ref = self.true_bank[layer]
                true_score, reference_index, nearest_direction = match_window(
                    true_ref, query, self.args.transition_weight, step,
                    self.args.step_window, self.args.window)
                before_score = true_score.detach()
                if self.arm == "inhibit":
                    gate = ((true_score - self.args.similarity_threshold)
                            / (1 - self.args.similarity_threshold)).clamp(0, 1)
                    requested = (gate * self.args.strength).clamp(max=self.args.max_delta_ratio)
                    chosen_direction = nearest_direction
                elif self.arm == "shuffled" and layer in self.shuffled_bank:
                    shuffled_score, _, chosen_direction = match_window(
                        self.shuffled_bank[layer], query, self.args.transition_weight,
                        step, self.args.step_window, self.args.window)
            if self.arm in ("random", "shuffled") and step > 0:
                requested = torch.tensor(
                    self.dose_schedule.get((step, layer), 0.0),
                    device=h.device, dtype=torch.float32).clamp(0, self.args.max_delta_ratio)
                if self.arm == "random":
                    chosen_direction = self._random(h)
            if requested is not None and chosen_direction is not None:
                new_h, achieved = bounded_edit(
                    h, chosen_direction, requested.reshape(1, 1),
                    not self.args.no_preserve_rms, self.args.max_delta_ratio,
                )
                # Avoid cloning full prefill tensors: only the last output position
                # needs to be changed. We steer only AFTER prefill.
                changed = hidden.clone()
                changed[:, -1, :] = new_h
                hidden = changed
                value = rebuild(hidden)
                actual = achieved.detach().mean()
                if layer in self.true_bank and reference_index is not None:
                    post_query = make_query(
                        self.states[layer], new_h[0], anchor, self.args.window)
                    if post_query is not None:
                        after_score = score_at_reference(
                            self.true_bank[layer], reference_index, post_query,
                            self.args.transition_weight, self.args.window).detach()
            self.states[layer].append(hidden[0, -1, :].detach().clone())
            self.diag.append({
                "layer": layer, "step": step,
                "true_similarity": before_score,
                "shuffled_similarity": shuffled_score,
                "post_similarity_same_reference": after_score,
                "target_ratio": requested.detach() if requested is not None else None,
                "actual_ratio": actual,
            })
            return value
        return hook

    def __enter__(self) -> 'TrajectoryProbe':
        for i in self.selected:
            self.handles.append(self.layers[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *_: Any) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def trajectory_cpu(self) -> dict[int, torch.Tensor]:
        return {i: torch.stack(history).detach().float().cpu() for i, history in self.states.items()}

    def diagnostics_cpu(self) -> list[dict[str, Any]]:
        if self._diag_cpu is not None:
            return self._diag_cpu
        names = ("true_similarity", "shuffled_similarity", "post_similarity_same_reference",
                 "target_ratio", "actual_ratio")
        scalars = []
        indexing = []
        for n, row in enumerate(self.diag):
            for field in names:
                value = row[field]
                if value is not None:
                    indexing.append((n, field))
                    scalars.append(value.float().reshape(()))
        as_cpu = torch.stack(scalars).detach().cpu().tolist() if scalars else []
        result = [{"layer": row["layer"], "step": row["step"],
                   **{field: None for field in names}} for row in self.diag]
        for (i, field), v in zip(indexing, as_cpu):
            result[i][field] = float(v)
        for row in result:
            row["applied"] = bool((row["actual_ratio"] or 0.0) > 1e-7)
        self._diag_cpu = result
        return result

    def summarize(self) -> dict[str, Any]:
        rows = self.diagnostics_cpu()
        edits = [r for r in rows if r["applied"]]
        compare = [r for r in rows if r["true_similarity"] is not None]
        matched = [r for r in compare if r["true_similarity"] >= self.args.similarity_threshold]
        changes = [r["true_similarity"] - r["post_similarity_same_reference"]
                   for r in rows if r["post_similarity_same_reference"] is not None]
        def avg(vals):
            return float(sum(vals) / len(vals)) if vals else None
        return {
            "forward_steps": self.step + 1,
            "edits": len(edits),
            "evaluated_windows": len(compare),
            "matched_windows": len(matched),
            "mean_true_path_similarity": avg([r["true_similarity"] for r in compare]),
            "max_true_path_similarity": max([r["true_similarity"] for r in compare], default=None),
            "mean_similarity_drop_same_reference": avg(changes),
            "mean_actual_delta_ratio": avg([r["actual_ratio"] for r in edits]) or 0.0,
            "max_actual_delta_ratio": max([r["actual_ratio"] for r in edits], default=0.0),
            "mean_shuffled_similarity": avg([r["shuffled_similarity"] for r in rows
                                             if r["shuffled_similarity"] is not None]),
        }


def word_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower(), re.UNICODE)
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def textual_metrics(text: str, previous: list[str]) -> dict[str, Any]:
    words = re.findall(r"\w+", text.lower(), re.UNICODE)
    cur = word_ngrams(text)
    prior = [word_ngrams(t) for t in previous]
    all_prior = set().union(*prior) if prior else set()
    return {
        "words": len(words), "characters": len(text),
        "unique_trigram_fraction": round(len(cur-all_prior) / max(1, len(cur)), 5),
        "max_previous_trigram_jaccard": round(max(
            (len(cur & old) / max(1, len(cur | old)) for old in prior), default=0.0), 5),
        "exact_repeat": any(text.strip() == v.strip() for v in previous),
        "repeated_bigram_fraction": round(1-len(word_ngrams(text, 2)) / max(1,len(words)-1), 5),
    }


def first_difference(left: list[int], right: list[int]) -> int | None:
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def encode_inputs(processor: Any, tokenizer: Any, task: str, chat: bool,
                  multimodal: bool) -> dict[str, torch.Tensor]:
    if chat:
        if multimodal:
            messages = [{"role": "user", "content": [{"type": "text", "text": task}]}]
            if hasattr(processor, "apply_chat_template"):
                try:
                    obj = processor.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt")
                    if "input_ids" in obj:
                        return {k: v for k, v in obj.items() if torch.is_tensor(v)}
                except Exception as exc:
                    print(f"WARNING: processor chat template failed ({exc}); "
                          "trying tokenizer chat template", file=sys.stderr)
        if getattr(tokenizer, "chat_template", None):
            obj = tokenizer.apply_chat_template(
                [{"role": "user", "content": task}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt", return_dict=True)
            return {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if multimodal:
            raise RuntimeError("No working Gemma 4 chat template; upgrade Transformers "
                               "or explicitly use --plain-prompt")
    return tokenizer(task, return_tensors="pt")


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, list[torch.nn.Module], dict[str, torch.Tensor]]:
    device = resolve_device(args.device)
    if device.type == "mps":
        # PyTorch MPS compiles and caches unary/cast shaders in an internal hash table
        # (MetalShaderLibrary) without thread locks. Transformers' default multi-threaded
        # weight materialization (ThreadPoolExecutor) races when copying safetensors
        # slices to MPS buffers, causing a SIGSEGV in libtorch_cpu.dylib
        # (__hash_table::__do_rehash). Enforce sequential materialization into MPS buffers.
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("Install torch transformers accelerate safetensors") from exc
    model_id = args.model
    gemma4 = "gemma-4" in model_id.lower()
    loader = "multimodal" if args.loader == "auto" and gemma4 else (
        "causal" if args.loader == "auto" else args.loader)
    dtype = (
        torch.float32 if device.type == "cpu" else
        torch.float16 if device.type == "mps" else torch.bfloat16
    ) if args.dtype == "auto" else getattr(torch, args.dtype)
    print(f"Loading {model_id} using {loader} on {device} ({dtype})", flush=True)
    processor = None
    if loader == "multimodal":
        model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_class is None:
            raise RuntimeError("Upgrade transformers: AutoModelForMultimodalLM is unavailable")
        processor = transformers.AutoProcessor.from_pretrained(model_id)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    else:
        model_class = transformers.AutoModelForCausalLM
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer must have pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"dtype": dtype}
    placement = args.load_placement
    if placement == "auto":
        placement = "direct" if device.type in ("mps", "cuda") else "staged"
    if placement == "direct" and device.type != "cpu":
        kwargs["device_map"] = "mps" if device.type == "mps" else "auto"
    model = model_class.from_pretrained(model_id, **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()
    layers = decoder_layers(model)
    encoded = encode_inputs(processor, tokenizer, args.task, not args.plain_prompt,
                            loader == "multimodal")
    if "input_ids" not in encoded:
        raise RuntimeError("Prompt encoding did not produce input_ids")
    embeddings = model.get_input_embeddings()
    input_device = embeddings.weight.device if embeddings is not None else device
    if input_device.type == "meta":
        input_device = device
    inputs = {k: v.to(input_device) for k, v in encoded.items() if torch.is_tensor(v)}
    return model, tokenizer, processor, layers, inputs


def run_generation(
    model: Any, tokenizer: Any, inputs: dict[str, torch.Tensor],
    layers: list[torch.nn.Module], selected: list[int], arm: str,
    round_id: int, seed: int, bank: TrajectoryBank,
    anchors: dict[int, torch.Tensor], args: argparse.Namespace,
    dose_schedule: Mapping[tuple[int, int], float] | None = None,
) -> tuple[Trace, list[dict[str, Any]]]:
    seed_all(seed)
    device = inputs["input_ids"].device
    # Compile old bank ONCE, not once per forward pass. Cast query to bank dtype.
    dtype = torch.float16 if device.type == "mps" else (torch.bfloat16 if device.type == "cuda" else torch.float32)
    true_bank = bank.compile(anchors, device, dtype, shuffled=False)
    shuffled_bank = (bank.compile(anchors, device, dtype, shuffled=True,
                                  shuffle_seed=args.seed + round_id * 2027)
                     if arm == "shuffled" else {})
    with TrajectoryProbe(layers, selected, arm, true_bank, shuffled_bank,
                         anchors, args, seed, dose_schedule) as probe:
        with torch.inference_mode():
            kw = {"max_new_tokens": args.max_new_tokens,
                  "do_sample": args.temperature > 0,
                  "num_beams": 1,
                  "pad_token_id": tokenizer.pad_token_id,
                  "use_cache": True}
            if args.temperature > 0:
                kw.update(temperature=args.temperature, top_p=args.top_p)
            result = model.generate(**inputs, **kw)
    ids = result.sequences[0] if hasattr(result, "sequences") else result[0]
    generated = ids[inputs["input_ids"].shape[-1]:].detach().cpu().tolist()
    eos_ids = tokenizer.eos_token_id
    allowed = set([eos_ids] if isinstance(eos_ids, int) else (eos_ids or []))
    generation_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    allowed.update([generation_eos] if isinstance(generation_eos, int) else (generation_eos or []))
    reached_eos = any(t in allowed for t in generated)
    trace = Trace(
        arm=arm, round=round_id, seed=seed,
        text=tokenizer.decode(generated, skip_special_tokens=True).strip(),
        states=probe.trajectory_cpu(), tokens=generated,
        metrics={**probe.summarize(), "tokens_generated": len(generated),
                 "eos_reached": reached_eos,
                 "max_tokens_hit": len(generated) >= args.max_new_tokens},
    )
    return trace, probe.diagnostics_cpu()


def save_trace(out: Path, trace: Trace, diagnostics: list[dict[str, Any]]) -> str:
    path = out / "traces" / f"{trace.arm}-{trace.round:02}.pt"
    torch.save({"states": trace.states, "token_ids": trace.tokens,
                "diagnostics": diagnostics, "seed": trace.seed,
                "arm": trace.arm, "round": trace.round}, path)
    return str(path.relative_to(out))


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm and r["round"] > 0]
        if not subset:
            continue
        def avg(key: str) -> float | None:
            vals = [float(v) for r in subset if (v := r.get(key)) is not None]
            return round(sum(vals)/len(vals), 5) if vals else None
        result[arm] = {
            "n": len(subset),
            "avg_unique_trigram_fraction": avg("unique_trigram_fraction"),
            "avg_mean_true_path_similarity": avg("mean_true_path_similarity"),
            "avg_max_true_path_similarity": avg("max_true_path_similarity"),
            "avg_similarity_drop_same_reference": avg("mean_similarity_drop_same_reference"),
            "avg_edits": avg("edits"),
            "avg_words": avg("words"),
            "completion_rate": round(sum(r["eos_reached"] for r in subset)/len(subset), 5),
            "mean_first_token_difference_from_baseline": avg("first_token_difference_from_baseline"),
        }
    return result


def render_candidates(rows: list[dict[str, Any]]) -> str:
    out = ["# Iterative trajectory exclusion — v2 candidates", "",
           "All arms share the original prompt and paired seeds; round 0 is shared.",
           "Scores are trajectory/lexical diagnostics, not measures of scientific quality.", ""]
    for r in rows:
        out += [f"## {r['arm']} / round {r['round']} / seed {r['seed']}",
                (f"Trace: `{r['trace_file']}` · edits: {r['edits']} · "
                 f"mean path sim: {r['mean_true_path_similarity']} · "
                 f"first difference vs baseline: {r['first_token_difference_from_baseline']}"),
                "", r["text"], "", "---", ""]
    return "\n".join(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--out", default="trajectory_exclusion_v2_run")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--loader", choices=["auto", "causal", "multimodal"], default="auto")
    p.add_argument("--load-placement", choices=["auto", "direct", "staged"], default="auto")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=800)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--seed-policy", choices=["fixed", "varied"], default="fixed")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--layers", default="0.40,0.60,0.80")
    p.add_argument("--window", type=int, default=3, help="contiguous decode steps per matched path window")
    p.add_argument("--transition-weight", type=float, default=0.5, help="weight of transition-vs-state similarity")
    p.add_argument("--strength", type=float, default=0.018)
    p.add_argument("--max-delta-ratio", type=float, default=0.025)
    p.add_argument("--similarity-threshold", type=float, default=0.75)
    p.add_argument("--step-window", type=int, default=-1,
                   help="optional +/- forward-step alignment constraint; -1 unrestricted")
    p.add_argument("--bank-runs", type=int, default=8)
    p.add_argument("--bank-windows-per-run", type=int, default=48)
    p.add_argument("--no-preserve-rms", action="store_true")
    p.add_argument("--plain-prompt", action="store_true")
    args = p.parse_args(argv)
    if args.rounds < 2 or args.max_new_tokens < args.window + 2:
        p.error("rounds >=2 and max-new-tokens >= window+2 required")
    if args.window < 2 or args.window > 16 or args.bank_windows_per_run < 1 or args.bank_runs < 1:
        p.error("window must be 2..16 and bank sizes must be positive")
    if not 0 <= args.transition_weight <= 1:
        p.error("transition-weight must be in [0,1]")
    if not 0 <= args.strength <= args.max_delta_ratio <= .2:
        p.error("require 0 <= strength <= max-delta-ratio <= 0.2")
    if not -1 <= args.similarity_threshold < 1:
        p.error("similarity-threshold must be in [-1,1)")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        p.error("invalid decoding settings")
    return args


def main(argv: list[str] | None = None, runtime: Any | None = None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    if (out / "runs.jsonl").exists():
        raise SystemExit(f"Refusing to overwrite prior run at {out / 'runs.jsonl'}")
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if runtime is None:
        model, tokenizer, _processor, layers, inputs = load_runtime(args)
    else:
        # Test injection only. No production workflow depends on this branch.
        model, tokenizer, layers, inputs = runtime
    selected = choose_layers(len(layers), args.layers)
    print(f"Selected decoder layers: {selected}", flush=True)
    anchors: dict[int, torch.Tensor] = {}
    bank = TrajectoryBank(args.bank_runs, args.bank_windows_per_run, args.window)
    histories = {a: [] for a in ARMS}
    rows: list[dict[str, Any]] = []
    logfile = out / "runs.jsonl"

    def record(t: Trace, d: list[dict[str, Any]], baseline_tokens: list[int],
               shared_path: str | None = None) -> dict[str, Any]:
        t.file = shared_path or save_trace(out, t, d)
        stats = textual_metrics(t.text, histories[t.arm])
        diff = first_difference(baseline_tokens, t.tokens)
        row = {"arm": t.arm, "round": t.round, "seed": t.seed,
               "trace_file": t.file, "text": t.text, **t.metrics, **stats,
               "first_token_difference_from_baseline": diff}
        rows.append(row)
        histories[t.arm].append(t.text)
        with logfile.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{t.arm:>9} {t.round:2} | {len(t.tokens):4} tokens | "
              f"{row['edits']:4} edits | path={row['mean_true_path_similarity']} | "
              f"first-diff={diff} | {t.text[:95].replace(chr(10), ' ')}", flush=True)
        return row

    try:
        for round_id in range(args.rounds):
            seed = args.seed if args.seed_policy == "fixed" else args.seed + round_id
            baseline, bdiag = run_generation(
                model, tokenizer, inputs, layers, selected, "baseline",
                round_id, seed, bank, anchors, args)
            record(baseline, bdiag, baseline.tokens)
            if round_id == 0:
                for arm in ARMS[1:]:
                    shared = dataclasses.replace(baseline, arm=arm)
                    record(shared, [], baseline.tokens, shared_path=baseline.file)
                bank.add(baseline)
                continue
            # All arms read the SAME bank; append inhibited path only after controls.
            inhibit, idiag = run_generation(
                model, tokenizer, inputs, layers, selected, "inhibit",
                round_id, seed, bank, anchors, args)
            record(inhibit, idiag, baseline.tokens)
            schedule = {(d["step"], d["layer"]): float(d["actual_ratio"] or 0)
                        for d in idiag}
            for arm in ("shuffled", "random"):
                candidate, diag = run_generation(
                    model, tokenizer, inputs, layers, selected, arm,
                    round_id, seed, bank, anchors, args, dose_schedule=schedule)
                realized = {(d["step"], d["layer"]): float(d["actual_ratio"] or 0)
                            for d in diag}
                common = schedule.keys() & realized.keys()
                # Audit exact step/layer post-RMS magnitude matching.
                errors = [abs(schedule[k] - realized[k]) for k in common]
                candidate.metrics["dose_coverage"] = len(common) / max(1, len(schedule))
                candidate.metrics["max_absolute_dose_error"] = max(errors, default=0)
                if candidate.metrics["dose_coverage"] < .95 or max(errors, default=0) > .003:
                    print(f"WARNING: {arm} dose replay mismatch", file=sys.stderr)
                record(candidate, diag, baseline.tokens)
            bank.add(inhibit)
    finally:
        (out / "summary.json").write_text(json.dumps(arm_summary(rows), indent=2), encoding="utf-8")
        (out / "candidates.md").write_text(render_candidates(rows), encoding="utf-8")
    print("\nExploration diagnostics (NOT idea quality):")
    print(json.dumps(arm_summary(rows), indent=2))
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
