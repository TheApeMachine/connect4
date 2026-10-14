#!/usr/bin/env python3
"""
Latent Composition Search — Counterbalanced Evaluation Edition
==============================================================

A single-file experiment harness for comparing two activation-level search methods
across local/open-weight transformer models.

  1. Contrastive Trajectory Inhibition (CTI)
     - Uses a fully crossed, seed-paired 2x2 calibration: conventional/composition
       instruction at both low and high decoding temperatures.
     - Fits the layer-wise instruction main effect, while separately logging the
       temperature main effect, interaction, and geometric overlap.
     - Fits the composition tangent/tube after residualizing temperature-cell means.
     - Compares paired baseline, learned-inhibition, and random-direction arms.
     - The random arm replays the learned arm's realized post-RMS perturbation-magnitude
       schedule at every matched token step and layer, with a hard audit.

  2. Autonomous Granular Controller (AGC)
     - The same model controls its next bounded search operation and may select layer
       segments, intervention geometry, rank, strength, token window, sampling settings,
       branch count, horizon, HALT, or ANSWER.
     - Source-model scores are online controller state only. Reported outcomes use a
       blinded leave-source-out cross-model jury.

Default models:
  - google/gemma-4-12B-it
  - openai/gpt-oss-20b

Outputs are written in JSONL and CSV, including candidates, evaluated_candidates,
cross_judgments, layer_geometry, intervention_traces, magnitude_audits, paired_tests,
controller_actions, summary, models, and errors. Geometry tensors are saved as .pt files,
and the figure pack contains PNG and PDF versions of all Matplotlib visualizations.

Recommended environment:

  python -m venv .venv
  source .venv/bin/activate
  pip install -U torch "transformers>=5.5.0" accelerate numpy matplotlib
  pip install -U bitsandbytes   # optional, used by --quantization auto/4bit on CUDA only
  huggingface-cli login

Apple Silicon: CUDA is unavailable, so the harness auto-selects MPS when
torch.backends.mps.is_available(). BitsAndBytes 4-bit quantization stays CUDA-only.

By default, all reportable candidates are evaluated by every selected model other than
the source model. Disabling cross-judging requires --exploratory-only. Even held-out
model-jury scores are proxies, not evidence of scientific novelty or truth; substantive
claims still require literature review, domain expertise, formal analysis, simulation,
or experiment.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
import itertools
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn


LOGGER = logging.getLogger("latent_composition_search")


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def accelerator_available() -> bool:
    return torch.cuda.is_available() or mps_available()


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_accelerator_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif mps_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Model and prompt configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    loader_kind: str
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "gemma-4-12b": ModelSpec(
        key="gemma-4-12b",
        model_id="google/gemma-4-12B-it",
        loader_kind="multimodal",
        description="Dense Gemma 4 unified decoder with per-layer embeddings",
    ),
    "gpt-oss-20b": ModelSpec(
        key="gpt-oss-20b",
        model_id="openai/gpt-oss-20b",
        loader_kind="causal",
        description="Sparse MoE GPT-OSS decoder",
    ),
}

DEFAULT_TASK = (
    "Develop one candidate research concept that specifically advances artificial "
    "intelligence research by composing already-known ideas in a way that reveals a "
    "plausibly overlooked link. Do not invent free-floating new primitives. Instead, "
    "reuse mechanisms, results, or constraints that are already part of established "
    "knowledge (in AI or elsewhere) and connect them so the joint implication for AI "
    "methods, theory, architectures, training, evaluation, alignment/safety, tooling, "
    "or scientific practice is non-obvious. The intended contribution must be scoped to "
    "AI research — not a generic scientific proposal that only mentions AI in passing. "
    "The composition must be mechanistic rather than a loose analogy or word "
    "association. State the known component ideas, the hidden or under-used link "
    "between them, the resulting AI-relevant mechanism, a concrete prediction about an "
    "AI system or experiment, a falsification test that an AI lab could run, and the "
    "nearest known alternative explanation in the AI literature. Treat the "
    "connectedness as a hypothesis: do not claim the link is historically unprecedented "
    "or that the idea is established novelty."
)

DISCOVERY_SYSTEM = (
    "You are participating in a controlled study of latent compositional discovery for "
    "advancing AI research. Your job is to recombine existing knowledge into a "
    "non-obvious but defensible link, not to invent new primitives from scratch. Be "
    "imaginative but epistemically strict. Keep every proposal tightly scoped to a "
    "concrete AI research contribution. Separate established premises, the composition "
    "or linking step, predicted consequences for AI systems or experiments, and ways "
    "the proposal could fail. Do not claim verified novelty or priority. Return a "
    "complete candidate answer rather than private chain-of-thought."
)

CONVENTIONAL_INSTRUCTION = (
    "Give the most standard, conservative, and already-familiar AI-research treatment "
    "of the task. Prefer well-trodden combinations and the nearest conventional "
    "explanation in the AI literature; do not search for an overlooked compositional "
    "link. To match output format across calibration arms, present exactly: "
    "(1) known ingredients, (2) the standard mapping or relation among them, "
    "(3) the conventional AI-relevant mechanism, (4) a testable prediction about an AI "
    "system or experiment, (5) a falsifier an AI lab could run, "
    "(6) the closest known alternative in AI, and (7) uncertainty."
)

COMPOSITION_INSTRUCTION = (
    "Search deliberately for a non-obvious compositional link among already-known "
    "ideas that advances AI research. Prefer recombinations that feel latent in existing "
    "knowledge: each ingredient should be recognizable, while the joint implication for "
    "AI should be under-explored or easy to miss. Do not invent new primitives, do not "
    "force an arbitrary multi-domain mashup, and reject mere word association or "
    "proposals whose primary payoff is outside AI. Present exactly: (1) known "
    "ingredients, (2) the overlooked or under-used link among them, (3) the composed "
    "AI-relevant mechanism, (4) a testable prediction about an AI system or experiment, "
    "(5) a falsifier an AI lab could run, (6) the closest known alternative in AI, and "
    "(7) uncertainty about whether the link is truly new."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator of candidate AI-research concepts produced by "
    "composing existing knowledge. Reward proposals that connect known ingredients "
    "through a non-obvious mechanistic link with a concrete AI payoff. Penalize "
    "invented primitives, forced multi-domain checklists, loose analogies, jargon, "
    "length, confidence, unsupported novelty claims, and clever science that is not "
    "actionable for AI. Return only one JSON object in the final answer. Give concise "
    "externally checkable reasons, not hidden chain-of-thought."
)

CONTROLLER_SYSTEM = (
    "You are the autonomous controller for your own next search operation aimed at "
    "advancing AI research through latent compositional discovery. You receive "
    "candidate outputs, quality proxies, and activation-intervention diagnostics. "
    "Choose one bounded control action. You may explore, refine, diversify, halt, or "
    "answer. Prefer actions that seek overlooked links among known ideas inside "
    "concrete AI methods, theory, systems, evaluation, or safety. Return only one JSON "
    "object in the final answer. Give only a brief operational reason, not private "
    "chain-of-thought."
)


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def short_hash(text: str, length: int = 10) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if mps_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
    try:
        import transformers

        transformers.set_seed(seed)
    except Exception:
        pass


def nested_getattr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            raise AttributeError(path)
        current = getattr(current, part)
    return current


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def flatten_for_csv(
    data: Mapping[str, Any], prefix: str = "", separator: str = "."
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_for_csv(value, full_key, separator))
        elif isinstance(value, (list, tuple, set)):
            flat[full_key] = json.dumps(jsonable(value), ensure_ascii=False)
        elif isinstance(value, torch.Tensor):
            flat[full_key] = json.dumps(jsonable(value), ensure_ascii=False)
        else:
            flat[full_key] = jsonable(value)
    return flat


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSONL line in %s", path)
    return rows


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [flatten_for_csv(record) for record in records]
    if not flattened:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)


def balanced_json_objects(text: str) -> Iterator[str]:
    """Yield balanced {...} substrings while respecting JSON strings."""
    starts: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            starts.append(index)
        elif char == "}" and starts:
            start = starts.pop()
            yield text[start : index + 1]


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidates: list[str] = []
    for fenced in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I):
        candidates.append(fenced)
    candidates.extend(balanced_json_objects(text))
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, candidate
    return None, None


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(words: Sequence[str], n: int = 3) -> set[tuple[str, ...]]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def lexical_novelty(text: str, references: Sequence[str], n: int = 3) -> float:
    target = ngrams(normalize_words(text), n=n)
    if not target or not references:
        return 1.0
    max_similarity = 0.0
    for reference in references:
        other = ngrams(normalize_words(reference), n=n)
        if not other:
            continue
        union = target | other
        similarity = len(target & other) / max(1, len(union))
        max_similarity = max(max_similarity, similarity)
    return 1.0 - max_similarity


def repetition_ratio(text: str) -> float:
    words = normalize_words(text)
    if not words:
        return 1.0
    bigrams = ngrams(words, n=2)
    total = max(1, len(words) - 1)
    return 1.0 - len(bigrams) / total


def make_candidate_id(model_key: str, method: str, task_id: str) -> str:
    return f"{model_key}-{method}-{task_id}-{uuid.uuid4().hex[:12]}"


# -----------------------------------------------------------------------------
# Record store
# -----------------------------------------------------------------------------


class RecordStore:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "candidates": output_dir / "candidates.jsonl",
            "controller_actions": output_dir / "controller_actions.jsonl",
            "layer_geometry": output_dir / "layer_geometry.jsonl",
            "intervention_traces": output_dir / "intervention_traces.jsonl",
            "cross_judgments": output_dir / "cross_judgments.jsonl",
            "evaluated_candidates": output_dir / "evaluated_candidates.jsonl",
            "summary": output_dir / "summary.jsonl",
            "paired_tests": output_dir / "paired_tests.jsonl",
            "magnitude_audits": output_dir / "magnitude_audits.jsonl",
            "models": output_dir / "models.jsonl",
            "errors": output_dir / "errors.jsonl",
        }
        for path in self.paths.values():
            if path.exists():
                path.unlink()

    def add(self, kind: str, record: Mapping[str, Any]) -> None:
        if kind not in self.paths:
            raise KeyError(kind)
        append_jsonl(self.paths[kind], record)

    def finalize_csv(self) -> None:
        for kind, jsonl_path in self.paths.items():
            records = read_jsonl(jsonl_path)
            write_csv(self.output_dir / f"{kind}.csv", records)


# -----------------------------------------------------------------------------
# Runtime model abstraction
# -----------------------------------------------------------------------------


@dataclass
class ModelRuntime:
    spec: ModelSpec
    model: Any
    tokenizer: Any
    processor: Any | None
    layers: list[nn.Module]
    layer_path: str
    input_device: torch.device
    config_summary: dict[str, Any]
    quantization_label: str


@dataclass
class GenerationResult:
    text: str
    raw_text: str | None
    full_token_ids: torch.Tensor
    generated_token_ids: torch.Tensor
    prompt_tokens: int
    generated_tokens: int
    seconds: float


def resolve_num_layers(config: Any) -> int | None:
    for candidate in (
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
        config,
    ):
        if candidate is None:
            continue
        value = getattr(candidate, "num_hidden_layers", None)
        if value is not None:
            return int(value)
    return None


def resolve_hidden_size(config: Any) -> int | None:
    for candidate in (
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
        config,
    ):
        if candidate is None:
            continue
        value = getattr(candidate, "hidden_size", None)
        if value is not None:
            return int(value)
    return None


def find_decoder_layers(model: nn.Module) -> tuple[list[nn.Module], str]:
    expected = resolve_num_layers(getattr(model, "config", None))
    candidate_paths = [
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "transformer.h",
        "model.transformer.h",
        "decoder.layers",
        "model.decoder.layers",
    ]
    for path in candidate_paths:
        try:
            value = nested_getattr(model, path)
        except AttributeError:
            continue
        if isinstance(value, (nn.ModuleList, list, tuple)) and value:
            layers = list(value)
            if expected is None or len(layers) == expected:
                return layers, path

    ranked: list[tuple[int, int, str, list[nn.Module]]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not module:
            continue
        children = list(module)
        child_names = " ".join(child.__class__.__name__.lower() for child in children[:4])
        semantic_score = 0
        for token in ("decoder", "block", "layer", "transformer"):
            if token in child_names:
                semantic_score += 1
        length_score = 2 if expected is not None and len(children) == expected else 0
        ranked.append((length_score + semantic_score, len(children), name, children))

    if not ranked:
        raise RuntimeError("Could not locate the transformer's decoder layer list.")
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, name, layers = ranked[0]
    if expected is not None and len(layers) != expected:
        LOGGER.warning(
            "Layer discovery found %d layers at %s, while config reports %d.",
            len(layers),
            name,
            expected,
        )
    return layers, name


def find_input_device(model: Any) -> torch.device:
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            device = embeddings.weight.device
            if device.type != "meta":
                return device
    except Exception:
        pass
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def config_to_summary(model: Any, layers: Sequence[nn.Module]) -> dict[str, Any]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    fields = [
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_local_experts",
        "num_experts_per_tok",
        "sliding_window",
        "max_position_embeddings",
        "layer_types",
    ]
    summary: dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "decoder_layer_class": layers[0].__class__.__name__ if layers else None,
        "discovered_num_layers": len(layers),
    }
    for source_name, source in (("config", config), ("text_config", text_config)):
        if source is None:
            continue
        for field_name in fields:
            if field_name in summary:
                continue
            value = getattr(source, field_name, None)
            if value is not None:
                summary[field_name] = jsonable(value)
                summary[f"{field_name}_source"] = source_name
    return summary


def build_quantization_config(spec: ModelSpec, mode: str) -> tuple[Any | None, str]:
    if mode == "none":
        return None, "none"
    if not torch.cuda.is_available():
        # bitsandbytes is CUDA-only; MPS/CPU fall back to full precision.
        return None, "none-non-cuda"
    use_4bit = mode == "4bit" or (mode == "auto" and spec.key == "gemma-4-12b")
    if not use_4bit:
        if spec.key == "gpt-oss-20b":
            return None, "native-mxfp4-or-checkpoint-default"
        return None, "none"
    try:
        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return config, "bitsandbytes-nf4"
    except Exception as exc:
        if mode == "4bit":
            raise RuntimeError(
                "--quantization 4bit was requested, but BitsAndBytesConfig could not be "
                "created. Install bitsandbytes and use a supported CUDA environment."
            ) from exc
        LOGGER.warning("bitsandbytes unavailable; loading %s without BnB.", spec.key)
        return None, "none-bnb-unavailable"


def load_model_runtime(spec: ModelSpec, args: argparse.Namespace) -> ModelRuntime:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install the dependencies shown in the file "
            "docstring before running the experiment."
        ) from exc

    target_device = resolve_torch_device()
    if target_device.type == "cpu" and not args.allow_cpu_large:
        raise RuntimeError(
            f"{spec.model_id} is too large for the default CPU path. Use CUDA or MPS, "
            "select a smaller model subset, or pass --allow-cpu-large deliberately."
        )

    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    quantization_config, quantization_label = build_quantization_config(spec, args.quantization)

    common_kwargs: dict[str, Any] = {
        "token": token,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if target_device.type == "cuda":
        common_kwargs["device_map"] = "auto"
    # MPS: do not use device_map — accelerate placement onto MPS can hang/segfault
    # during weight load. Load on CPU, then move with model.to("mps") below.
    if quantization_config is not None:
        common_kwargs["quantization_config"] = quantization_config
    if args.attn_implementation != "auto":
        common_kwargs["attn_implementation"] = args.attn_implementation

    # Newer Transformers uses dtype=; older supported releases use torch_dtype=.
    # Prefer float16 on MPS — bfloat16 support is incomplete on Apple Silicon.
    common_kwargs["dtype"] = torch.float16 if target_device.type == "mps" else "auto"

    processor = None
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            spec.model_id,
            token=token,
            trust_remote_code=args.trust_remote_code,
            padding_side="left",
        )
    except Exception:
        processor = None

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            spec.model_id,
            token=token,
            trust_remote_code=args.trust_remote_code,
            padding_side="left",
        )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    class_names: list[str]
    if spec.loader_kind == "multimodal":
        class_names = [
            "AutoModelForMultimodalLM",
            "AutoModelForImageTextToText",
            "AutoModelForCausalLM",
        ]
    else:
        class_names = ["AutoModelForCausalLM"]

    errors: list[str] = []
    model = None
    for class_name in class_names:
        loader = getattr(transformers, class_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(spec.model_id, **common_kwargs)
            break
        except TypeError:
            # Retry with the legacy torch_dtype name.
            legacy_kwargs = dict(common_kwargs)
            dtype = legacy_kwargs.pop("dtype", "auto")
            legacy_kwargs["torch_dtype"] = dtype
            try:
                model = loader.from_pretrained(spec.model_id, **legacy_kwargs)
                break
            except Exception as inner:
                errors.append(f"{class_name}: {inner}")
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")

    if model is None:
        joined = "\n".join(errors[-3:])
        raise RuntimeError(f"Unable to load {spec.model_id}. Loader errors:\n{joined}")

    model.eval()
    if target_device.type == "mps":
        try:
            model.to(target_device)
        except Exception as exc:
            raise RuntimeError(
                f"Loaded {spec.model_id} on CPU but failed to move it to MPS: {exc}"
            ) from exc
    layers, layer_path = find_decoder_layers(model)
    input_device = find_input_device(model)
    summary = config_to_summary(model, layers)
    return ModelRuntime(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        layers=layers,
        layer_path=layer_path,
        input_device=input_device,
        config_summary=summary,
        quantization_label=quantization_label,
    )


def unload_model_runtime(runtime: ModelRuntime | None) -> None:
    if runtime is None:
        return
    try:
        del runtime.model
    except Exception:
        pass
    gc.collect()
    empty_accelerator_cache()


def messages_for_template(messages: Sequence[Mapping[str, str]], multimodal: bool) -> list[dict[str, Any]]:
    if not multimodal:
        return [dict(message) for message in messages]
    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        converted.append(
            {
                "role": message.get("role", "user"),
                "content": [{"type": "text", "text": str(content)}],
            }
        )
    return converted


def encode_messages(runtime: ModelRuntime, messages: Sequence[Mapping[str, str]]) -> dict[str, torch.Tensor]:
    templaters: list[tuple[Any, bool]] = []
    if runtime.processor is not None and hasattr(runtime.processor, "apply_chat_template"):
        templaters.append((runtime.processor, True))
    templaters.append((runtime.tokenizer, False))

    errors: list[str] = []
    encoded: Any = None
    for templater, multimodal in templaters:
        variants = [messages_for_template(messages, multimodal)]
        if multimodal:
            variants.append([dict(message) for message in messages])
        for variant in variants:
            try:
                encoded = templater.apply_chat_template(
                    variant,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                break
            except Exception as exc:
                errors.append(f"{templater.__class__.__name__}: {exc}")
        if encoded is not None:
            break

    if encoded is None:
        try:
            rendered = runtime.tokenizer.apply_chat_template(
                [dict(message) for message in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            rendered = "\n\n".join(
                f"{message.get('role', 'user').upper()}: {message.get('content', '')}"
                for message in messages
            ) + "\n\nASSISTANT:"
        encoded = runtime.tokenizer(rendered, return_tensors="pt")

    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
    elif hasattr(encoded, "data") and isinstance(encoded.data, Mapping):
        encoded = dict(encoded.data)
    else:
        encoded = dict(encoded)

    tensor_inputs: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if torch.is_tensor(value):
            tensor_inputs[key] = value.to(runtime.input_device)
    if "input_ids" not in tensor_inputs:
        raise RuntimeError(
            "Chat-template encoding did not return input_ids. Errors: " + " | ".join(errors[-3:])
        )
    if "attention_mask" not in tensor_inputs:
        tensor_inputs["attention_mask"] = torch.ones_like(tensor_inputs["input_ids"])
    return tensor_inputs


def extract_harmony_final(raw_text: str, clean_text: str) -> str:
    patterns = [
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)",
        r"<\|final\|>(.*?)(?:<\|end\|>|$)",
        r"final\s*<\|message\|>(.*?)(?:<\|end\|>|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, raw_text, flags=re.S | re.I)
        if matches:
            return matches[-1].strip()
    return clean_text.strip()


def decode_generated(runtime: ModelRuntime, token_ids: torch.Tensor) -> tuple[str, str]:
    ids = token_ids.detach().cpu().tolist()
    raw_text = runtime.tokenizer.decode(ids, skip_special_tokens=False)
    clean_text = runtime.tokenizer.decode(ids, skip_special_tokens=True)
    if runtime.spec.key == "gpt-oss-20b":
        clean_text = extract_harmony_final(raw_text, clean_text)
    return clean_text.strip(), raw_text


def generate_text(
    runtime: ModelRuntime,
    messages: Sequence[Mapping[str, str]],
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    store_raw: bool,
) -> GenerationResult:
    seed_everything(seed)
    inputs = encode_messages(runtime, messages)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    do_sample = temperature > 0.0
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "use_cache": True,
        "num_beams": 1,
        "repetition_penalty": float(repetition_penalty),
        "pad_token_id": runtime.tokenizer.pad_token_id,
    }
    if runtime.tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = runtime.tokenizer.eos_token_id
    if do_sample:
        kwargs["temperature"] = max(1e-4, float(temperature))
        kwargs["top_p"] = clamp(float(top_p), 0.01, 1.0)

    start = time.perf_counter()
    with torch.inference_mode():
        output = runtime.model.generate(**inputs, **kwargs)
    elapsed = time.perf_counter() - start
    sequences = output.sequences if hasattr(output, "sequences") else output
    if not torch.is_tensor(sequences):
        raise RuntimeError("model.generate returned an unsupported output type")
    full_ids = sequences[0].detach().cpu()
    generated_ids = full_ids[prompt_tokens:]
    text, raw_text = decode_generated(runtime, generated_ids)
    # Drop generate inputs/outputs before the next call; MPS especially needs this.
    del inputs, output, sequences
    empty_accelerator_cache()
    return GenerationResult(
        text=text,
        raw_text=raw_text if store_raw else None,
        full_token_ids=full_ids,
        generated_token_ids=generated_ids,
        prompt_tokens=prompt_tokens,
        generated_tokens=int(generated_ids.numel()),
        seconds=elapsed,
    )


# -----------------------------------------------------------------------------
# Layer-state collection and geometry fitting
# -----------------------------------------------------------------------------


def hidden_from_output(output: Any) -> tuple[torch.Tensor | None, Any]:
    """Return the hidden tensor and a function that replaces it in the layer output."""
    if torch.is_tensor(output):
        return output, lambda new_hidden: new_hidden
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0], lambda new_hidden: (new_hidden, *output[1:])
    if isinstance(output, list) and output and torch.is_tensor(output[0]):
        return output[0], lambda new_hidden: [new_hidden, *output[1:]]
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        def replace(new_hidden: torch.Tensor) -> Any:
            try:
                output.last_hidden_state = new_hidden
                return output
            except Exception:
                return output
        return output.last_hidden_state, replace
    return None, lambda new_hidden: output


class LayerStateCollector:
    def __init__(self, layers: Sequence[nn.Module], state_tokens: int, generated_tokens: int) -> None:
        self.layers = layers
        self.state_tokens = max(1, int(state_tokens))
        self.generated_tokens = max(1, int(generated_tokens))
        self.handles: list[Any] = []
        self.states: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_index: int):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module, inputs
            hidden, _ = hidden_from_output(output)
            if hidden is None or hidden.ndim != 3:
                return
            token_count = min(self.state_tokens, self.generated_tokens, hidden.shape[1])
            sample = hidden[:, -token_count:, :].detach().float().cpu()
            self.states[layer_index] = sample.reshape(-1, sample.shape[-1])
        return hook

    def __enter__(self) -> "LayerStateCollector":
        self.handles = [
            layer.register_forward_hook(self._make_hook(index))
            for index, layer in enumerate(self.layers)
        ]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def capture_sequence_states(
    runtime: ModelRuntime,
    full_token_ids: torch.Tensor,
    generated_tokens: int,
    state_tokens: int,
) -> dict[int, torch.Tensor]:
    ids = full_token_ids.unsqueeze(0).to(runtime.input_device)
    attention_mask = torch.ones_like(ids)
    with LayerStateCollector(runtime.layers, state_tokens, generated_tokens) as collector:
        with torch.inference_mode():
            try:
                runtime.model(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                )
            except TypeError:
                runtime.model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
    empty_accelerator_cache()
    if len(collector.states) != len(runtime.layers):
        missing = sorted(set(range(len(runtime.layers))) - set(collector.states))
        raise RuntimeError(f"Failed to capture hidden states for layers: {missing[:10]}")
    return collector.states


@dataclass
class CalibrationSample:
    sample_id: str
    instruction: str
    temperature_condition: str
    temperature: float
    replicate_index: int
    paired_seed: int
    text: str
    result: GenerationResult
    score: dict[str, Any]
    states: dict[int, torch.Tensor]


@dataclass
class CalibrationBundle:
    cells: dict[str, list[CalibrationSample]]
    conventional: list[CalibrationSample]
    exploratory: list[CalibrationSample]
    positive: list[CalibrationSample]
    negative: list[CalibrationSample]


@dataclass
class LayerGeometry:
    center_positive: torch.Tensor
    center_negative: torch.Tensor
    default_basis: torch.Tensor
    default_threshold: torch.Tensor
    default_scale: torch.Tensor
    positive_basis: torch.Tensor
    tube_radius: float
    signal: float
    smoothed_signal: float
    temperature_signal: float
    interaction_signal: float
    instruction_temperature_overlap_max: float
    instruction_temperature_overlap_mean: float
    instruction_temperature_mean_cosine: float
    mean_delta_norm: float
    temperature_delta_norm: float
    interaction_delta_norm: float
    singular_values: list[float]
    cumulative_energy: list[float]
    positive_samples: int
    negative_samples: int
    factorial_pairs: int


@dataclass
class TrajectoryGeometry:
    layers: dict[int, LayerGeometry]
    selected_layers: list[int]
    layer_weights: dict[int, float]
    rank: int
    positive_rank: int
    geometry_name: str = "learned"

    def state_dict(self) -> dict[str, Any]:
        return {
            "selected_layers": self.selected_layers,
            "layer_weights": self.layer_weights,
            "rank": self.rank,
            "positive_rank": self.positive_rank,
            "geometry_name": self.geometry_name,
            "design": "balanced 2x2 instruction-by-temperature main effect",
            "layers": {
                index: {
                    "center_positive": geometry.center_positive,
                    "center_negative": geometry.center_negative,
                    "default_basis": geometry.default_basis,
                    "default_threshold": geometry.default_threshold,
                    "default_scale": geometry.default_scale,
                    "positive_basis": geometry.positive_basis,
                    "tube_radius": geometry.tube_radius,
                    "signal": geometry.signal,
                    "smoothed_signal": geometry.smoothed_signal,
                    "temperature_signal": geometry.temperature_signal,
                    "interaction_signal": geometry.interaction_signal,
                    "instruction_temperature_overlap_max": geometry.instruction_temperature_overlap_max,
                    "instruction_temperature_overlap_mean": geometry.instruction_temperature_overlap_mean,
                    "instruction_temperature_mean_cosine": geometry.instruction_temperature_mean_cosine,
                    "mean_delta_norm": geometry.mean_delta_norm,
                    "temperature_delta_norm": geometry.temperature_delta_norm,
                    "interaction_delta_norm": geometry.interaction_delta_norm,
                    "singular_values": geometry.singular_values,
                    "cumulative_energy": geometry.cumulative_energy,
                    "positive_samples": geometry.positive_samples,
                    "negative_samples": geometry.negative_samples,
                    "factorial_pairs": geometry.factorial_pairs,
                }
                for index, geometry in self.layers.items()
            },
        }


def calibration_cell_key(instruction: str, temperature_condition: str) -> str:
    return f"{instruction}:{temperature_condition}"


def orthonormal_random(dimension: int, rank: int, generator: torch.Generator) -> torch.Tensor:
    rank = max(1, min(rank, dimension))
    matrix = torch.randn(dimension, rank, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :rank].contiguous()


def safe_quantile(values: torch.Tensor, q: float, dim: int | None = None) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(0.0)
    return torch.quantile(values.float(), clamp(q, 0.0, 1.0), dim=dim)


def tail_align(*tensors: torch.Tensor) -> list[torch.Tensor]:
    if not tensors:
        return []
    rows = min(int(tensor.shape[0]) for tensor in tensors)
    if rows <= 0:
        raise ValueError("Calibration state tensor has no rows")
    return [tensor[-rows:].float() for tensor in tensors]


def effect_basis(effect_rows: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    effect_rows = effect_rows.float()
    mean_effect = effect_rows.mean(dim=0)
    centered = effect_rows - mean_effect
    design = torch.cat(
        [mean_effect.unsqueeze(0) * math.sqrt(max(1, effect_rows.shape[0])), centered],
        dim=0,
    )
    _, singular_values, vh = torch.linalg.svd(design, full_matrices=False)
    actual_rank = max(1, min(rank, vh.shape[0]))
    basis = vh[:actual_rank].T.contiguous()
    orientation = mean_effect @ basis
    signs = torch.where(orientation >= 0, 1.0, -1.0)
    basis = basis * signs.unsqueeze(0)
    coefficients = effect_rows @ basis
    signal = float(
        (
            coefficients.mean(dim=0).square()
            / coefficients.var(dim=0, unbiased=False).clamp_min(1e-6)
        )
        .mean()
        .item()
    )
    return basis, singular_values, signal


def factorial_layer_arrays(
    calibration: CalibrationBundle,
    layer_index: int,
) -> dict[str, torch.Tensor]:
    indexed: dict[tuple[str, str], dict[int, CalibrationSample]] = {}
    for instruction in ("conventional", "composition"):
        for temperature_condition in ("low", "high"):
            key = calibration_cell_key(instruction, temperature_condition)
            samples = calibration.cells.get(key, [])
            indexed[(instruction, temperature_condition)] = {
                sample.replicate_index: sample for sample in samples
            }

    replicate_sets = [set(cell) for cell in indexed.values()]
    common_replicates = sorted(set.intersection(*replicate_sets)) if replicate_sets else []
    if not common_replicates:
        raise RuntimeError("The 2x2 calibration cells have no matched replicates")

    instruction_rows: list[torch.Tensor] = []
    temperature_rows: list[torch.Tensor] = []
    interaction_rows: list[torch.Tensor] = []
    for replicate_index in common_replicates:
        conv_low, comp_low, conv_high, comp_high = tail_align(
            indexed[("conventional", "low")][replicate_index].states[layer_index],
            indexed[("composition", "low")][replicate_index].states[layer_index],
            indexed[("conventional", "high")][replicate_index].states[layer_index],
            indexed[("composition", "high")][replicate_index].states[layer_index],
        )
        low_instruction = conv_low - comp_low
        high_instruction = conv_high - comp_high
        instruction_rows.append(0.5 * (low_instruction + high_instruction))
        temperature_rows.append(
            0.5 * ((conv_high - conv_low) + (comp_high - comp_low))
        )
        interaction_rows.append(0.5 * (high_instruction - low_instruction))

    cell_states: dict[tuple[str, str], torch.Tensor] = {}
    for cell_key, replicate_map in indexed.items():
        states = [sample.states[layer_index].float() for sample in replicate_map.values()]
        cell_states[cell_key] = torch.cat(states, dim=0)

    composition_cell_means = [
        cell_states[("composition", condition)].mean(dim=0)
        for condition in ("low", "high")
    ]
    conventional_cell_means = [
        cell_states[("conventional", condition)].mean(dim=0)
        for condition in ("low", "high")
    ]
    center_positive = torch.stack(composition_cell_means).mean(dim=0)
    center_negative = torch.stack(conventional_cell_means).mean(dim=0)

    positive = torch.cat(
        [cell_states[("composition", condition)] for condition in ("low", "high")],
        dim=0,
    )
    negative = torch.cat(
        [cell_states[("conventional", condition)] for condition in ("low", "high")],
        dim=0,
    )

    # Remove the temperature-cell mean before fitting the composition tangent/tube.
    # This prevents the tube itself from becoming a disguised low-vs-high-temperature axis.
    positive_residualized = torch.cat(
        [
            cell_states[("composition", condition)]
            - cell_states[("composition", condition)].mean(dim=0)
            + center_positive
            for condition in ("low", "high")
        ],
        dim=0,
    )

    return {
        "positive": positive,
        "negative": negative,
        "positive_residualized": positive_residualized,
        "center_positive": center_positive,
        "center_negative": center_negative,
        "instruction_effect": torch.cat(instruction_rows, dim=0),
        "temperature_effect": torch.cat(temperature_rows, dim=0),
        "interaction_effect": torch.cat(interaction_rows, dim=0),
        "factorial_pairs": torch.tensor(len(common_replicates), dtype=torch.int64),
    }


def fit_one_layer_geometry(
    arrays: Mapping[str, torch.Tensor],
    rank: int,
    positive_rank: int,
    threshold_quantile: float,
) -> LayerGeometry:
    positive = arrays["positive"].float()
    negative = arrays["negative"].float()
    positive_residualized = arrays["positive_residualized"].float()
    center_positive = arrays["center_positive"].float()
    center_negative = arrays["center_negative"].float()
    instruction_effect = arrays["instruction_effect"].float()
    temperature_effect = arrays["temperature_effect"].float()
    interaction_effect = arrays["interaction_effect"].float()

    default_basis, singular_values_tensor, signal = effect_basis(instruction_effect, rank)
    temperature_basis, _, temperature_signal = effect_basis(temperature_effect, rank)
    interaction_basis, _, interaction_signal = effect_basis(interaction_effect, rank)
    del interaction_basis

    overlap_values = torch.linalg.svdvals(default_basis.T @ temperature_basis)
    overlap_max = float(overlap_values.max().item()) if overlap_values.numel() else 0.0
    overlap_mean = float(overlap_values.mean().item()) if overlap_values.numel() else 0.0
    instruction_mean = instruction_effect.mean(dim=0)
    temperature_mean = temperature_effect.mean(dim=0)
    mean_cosine = float(
        torch.abs(
            torch.dot(instruction_mean, temperature_mean)
            / (
                torch.linalg.vector_norm(instruction_mean)
                * torch.linalg.vector_norm(temperature_mean)
            ).clamp_min(1e-8)
        ).item()
    )

    positive_relative = positive - center_positive
    positive_coefficients = positive_relative @ default_basis
    default_threshold = safe_quantile(positive_coefficients, threshold_quantile, dim=0)
    default_scale = positive_coefficients.std(dim=0, unbiased=False).clamp_min(1e-4)

    centered_positive = positive_residualized - center_positive
    if centered_positive.shape[0] >= 2:
        _, _, positive_vh = torch.linalg.svd(centered_positive, full_matrices=False)
        actual_positive_rank = max(1, min(positive_rank, positive_vh.shape[0]))
        positive_basis = positive_vh[:actual_positive_rank].T.contiguous()
    else:
        positive_basis = default_basis[:, :1].clone()

    projected_positive = (centered_positive @ positive_basis) @ positive_basis.T
    orthogonal_positive = centered_positive - projected_positive
    orthogonal_norms = torch.linalg.vector_norm(orthogonal_positive, dim=-1)
    tube_radius = float(safe_quantile(orthogonal_norms, 0.95).item())

    energy = singular_values_tensor.square()
    if energy.sum() > 0:
        cumulative = torch.cumsum(energy / energy.sum(), dim=0)
    else:
        cumulative = torch.zeros_like(energy)

    return LayerGeometry(
        center_positive=center_positive.cpu(),
        center_negative=center_negative.cpu(),
        default_basis=default_basis.cpu(),
        default_threshold=default_threshold.cpu(),
        default_scale=default_scale.cpu(),
        positive_basis=positive_basis.cpu(),
        tube_radius=tube_radius,
        signal=signal,
        smoothed_signal=signal,
        temperature_signal=temperature_signal,
        interaction_signal=interaction_signal,
        instruction_temperature_overlap_max=overlap_max,
        instruction_temperature_overlap_mean=overlap_mean,
        instruction_temperature_mean_cosine=mean_cosine,
        mean_delta_norm=float(torch.linalg.vector_norm(instruction_effect.mean(dim=0)).item()),
        temperature_delta_norm=float(
            torch.linalg.vector_norm(temperature_effect.mean(dim=0)).item()
        ),
        interaction_delta_norm=float(
            torch.linalg.vector_norm(interaction_effect.mean(dim=0)).item()
        ),
        singular_values=[float(v) for v in singular_values_tensor[: max(rank, 8)].tolist()],
        cumulative_energy=[float(v) for v in cumulative[: max(rank, 8)].tolist()],
        positive_samples=int(positive.shape[0]),
        negative_samples=int(negative.shape[0]),
        factorial_pairs=int(arrays["factorial_pairs"].item()),
    )


def smooth_signal(values: Sequence[float], radius: int = 1) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        window = [value for value in values[start:end] if math.isfinite(value)]
        result.append(float(np.mean(window)) if window else 0.0)
    return result


def parse_fixed_band(text: str | None, n_layers: int) -> list[int] | None:
    if not text:
        return None
    match = re.fullmatch(r"\s*([0-9.]+)\s*:\s*([0-9.]+)\s*", text)
    if not match:
        raise ValueError("--fixed-band must be START:END in normalized layer fractions")
    start = clamp(float(match.group(1)), 0.0, 1.0)
    end = clamp(float(match.group(2)), 0.0, 1.0)
    if end < start:
        start, end = end, start
    start_index = min(n_layers - 1, int(math.floor(start * n_layers)))
    end_index = min(n_layers - 1, int(math.ceil(end * n_layers)) - 1)
    return list(range(start_index, end_index + 1))


def select_layer_band(
    signals: Sequence[float],
    band_fraction: float,
    exclude_final_layers: int,
    fixed_band: str | None,
) -> tuple[list[int], dict[int, float], list[float]]:
    n_layers = len(signals)
    fixed = parse_fixed_band(fixed_band, n_layers)
    smoothed = smooth_signal(signals, radius=1)
    if fixed:
        selected = fixed
    else:
        competitive_end = max(1, n_layers - max(0, exclude_final_layers))
        window = max(1, int(round(competitive_end * clamp(band_fraction, 0.02, 1.0))))
        window = min(window, competitive_end)
        best_start = 0
        best_value = -float("inf")
        for start in range(0, competitive_end - window + 1):
            value = float(np.mean(smoothed[start : start + window]))
            if value > best_value:
                best_value = value
                best_start = start
        selected = list(range(best_start, best_start + window))

    selected_values = [max(0.0, smoothed[index]) for index in selected]
    scale = max(selected_values) if selected_values else 1.0
    if scale <= 0:
        weights = {index: 1.0 for index in selected}
    else:
        weights = {
            index: clamp(smoothed[index] / scale, 0.20, 1.0)
            for index in selected
        }
    return selected, weights, smoothed


def build_random_control_geometry(
    learned: TrajectoryGeometry,
    positive_by_layer: Mapping[int, torch.Tensor],
    seed: int,
    threshold_quantile: float,
) -> TrajectoryGeometry:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_layers: dict[int, LayerGeometry] = {}
    for layer_index, learned_layer in learned.layers.items():
        positive = positive_by_layer[layer_index].float()
        dimension = positive.shape[-1]
        q = orthonormal_random(dimension, learned.rank, generator)
        p = orthonormal_random(dimension, learned.positive_rank, generator)
        center = learned_layer.center_positive.float()
        relative = positive - center
        coefficients = relative @ q
        threshold = safe_quantile(coefficients, threshold_quantile, dim=0)
        scale = coefficients.std(dim=0, unbiased=False).clamp_min(1e-4)
        projected = (relative @ p) @ p.T
        residual = relative - projected
        radius = float(safe_quantile(torch.linalg.vector_norm(residual, dim=-1), 0.95).item())
        random_layers[layer_index] = LayerGeometry(
            center_positive=learned_layer.center_positive.clone(),
            center_negative=learned_layer.center_negative.clone(),
            default_basis=q,
            default_threshold=threshold,
            default_scale=scale,
            positive_basis=p,
            tube_radius=radius,
            signal=learned_layer.signal,
            smoothed_signal=learned_layer.smoothed_signal,
            temperature_signal=learned_layer.temperature_signal,
            interaction_signal=learned_layer.interaction_signal,
            instruction_temperature_overlap_max=learned_layer.instruction_temperature_overlap_max,
            instruction_temperature_overlap_mean=learned_layer.instruction_temperature_overlap_mean,
            instruction_temperature_mean_cosine=learned_layer.instruction_temperature_mean_cosine,
            mean_delta_norm=learned_layer.mean_delta_norm,
            temperature_delta_norm=learned_layer.temperature_delta_norm,
            interaction_delta_norm=learned_layer.interaction_delta_norm,
            singular_values=learned_layer.singular_values,
            cumulative_energy=learned_layer.cumulative_energy,
            positive_samples=learned_layer.positive_samples,
            negative_samples=learned_layer.negative_samples,
            factorial_pairs=learned_layer.factorial_pairs,
        )
    return TrajectoryGeometry(
        layers=random_layers,
        selected_layers=list(learned.selected_layers),
        layer_weights=dict(learned.layer_weights),
        rank=learned.rank,
        positive_rank=learned.positive_rank,
        geometry_name="random-control",
    )


def fit_trajectory_geometry(
    runtime: ModelRuntime,
    calibration: CalibrationBundle,
    args: argparse.Namespace,
    task_id: str,
    store: RecordStore,
) -> tuple[TrajectoryGeometry, TrajectoryGeometry]:
    positive_by_layer: dict[int, torch.Tensor] = {}
    layer_geometries: dict[int, LayerGeometry] = {}

    for layer_index in range(len(runtime.layers)):
        arrays = factorial_layer_arrays(calibration, layer_index)
        positive_by_layer[layer_index] = arrays["positive"]
        layer_geometries[layer_index] = fit_one_layer_geometry(
            arrays=arrays,
            rank=args.trajectory_rank,
            positive_rank=args.positive_rank,
            threshold_quantile=args.threshold_quantile,
        )

    signals = [layer_geometries[index].signal for index in range(len(runtime.layers))]
    selected, weights, smoothed = select_layer_band(
        signals=signals,
        band_fraction=args.band_fraction,
        exclude_final_layers=args.exclude_final_layers,
        fixed_band=args.fixed_band,
    )
    for index, value in enumerate(smoothed):
        layer_geometries[index].smoothed_signal = value

    actual_rank = min(
        args.trajectory_rank,
        min(layer.default_basis.shape[1] for layer in layer_geometries.values()),
    )
    actual_positive_rank = min(
        args.positive_rank,
        min(layer.positive_basis.shape[1] for layer in layer_geometries.values()),
    )
    learned = TrajectoryGeometry(
        layers=layer_geometries,
        selected_layers=selected,
        layer_weights=weights,
        rank=actual_rank,
        positive_rank=actual_positive_rank,
        geometry_name="learned-factorial-main-effect",
    )
    random_control = build_random_control_geometry(
        learned=learned,
        positive_by_layer=positive_by_layer,
        seed=args.seed + 991,
        threshold_quantile=args.threshold_quantile,
    )

    layer_types = runtime.config_summary.get("layer_types")
    for layer_index, geometry in layer_geometries.items():
        layer_type = None
        if isinstance(layer_types, list) and layer_index < len(layer_types):
            layer_type = layer_types[layer_index]
        store.add(
            "layer_geometry",
            {
                "timestamp": utc_now(),
                "task_id": task_id,
                "model_key": runtime.spec.key,
                "model_id": runtime.spec.model_id,
                "layer_index": layer_index,
                "layer_fraction": layer_index / max(1, len(runtime.layers) - 1),
                "layer_type": layer_type,
                "design": "balanced-2x2-instruction-main-effect",
                "signal": geometry.signal,
                "smoothed_signal": geometry.smoothed_signal,
                "temperature_signal": geometry.temperature_signal,
                "interaction_signal": geometry.interaction_signal,
                "instruction_temperature_overlap_max": geometry.instruction_temperature_overlap_max,
                "instruction_temperature_overlap_mean": geometry.instruction_temperature_overlap_mean,
                "instruction_temperature_mean_cosine": geometry.instruction_temperature_mean_cosine,
                "mean_delta_norm": geometry.mean_delta_norm,
                "temperature_delta_norm": geometry.temperature_delta_norm,
                "interaction_delta_norm": geometry.interaction_delta_norm,
                "selected": layer_index in selected,
                "layer_weight": weights.get(layer_index, 0.0),
                "rank": actual_rank,
                "positive_rank": actual_positive_rank,
                "tube_radius": geometry.tube_radius,
                "singular_values": geometry.singular_values,
                "cumulative_energy": geometry.cumulative_energy,
                "positive_samples": geometry.positive_samples,
                "negative_samples": geometry.negative_samples,
                "factorial_pairs": geometry.factorial_pairs,
            },
        )

    geometry_dir = store.output_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        learned.state_dict(),
        geometry_dir / f"{runtime.spec.key}-{task_id}-learned.pt",
    )
    torch.save(
        random_control.state_dict(),
        geometry_dir / f"{runtime.spec.key}-{task_id}-random.pt",
    )
    return learned, random_control


# -----------------------------------------------------------------------------
# Activation intervention
# -----------------------------------------------------------------------------


@dataclass
class InterventionSettings:
    default_strength: float
    tube_strength: float
    max_delta_ratio: float
    rank: int
    positive_rank: int
    token_start: int = 0
    token_end: int | None = None
    adaptive_gate: bool = True
    gate_temperature: float = 0.75
    preserve_rms: bool = True
    layer_weights: dict[int, float] | None = None
    label: str = "intervention"
    magnitude_match_tolerance: float = 0.02


class TrajectoryIntervention:
    def __init__(
        self,
        runtime: ModelRuntime,
        geometry: TrajectoryGeometry,
        settings: InterventionSettings,
        magnitude_reference_geometry: TrajectoryGeometry | None = None,
        target_delta_ratio_schedule: Mapping[tuple[int, int], float] | None = None,
    ) -> None:
        self.runtime = runtime
        self.geometry = geometry
        self.settings = settings
        self.magnitude_reference_geometry = magnitude_reference_geometry
        self.target_delta_ratio_schedule = {
            (int(step), int(layer)): float(value)
            for (step, layer), value in (target_delta_ratio_schedule or {}).items()
        }
        self.handles: list[Any] = []
        self.trace: list[dict[str, Any]] = []
        self._pending_trace: list[dict[str, Any]] = []
        self.step = -1
        self.device_cache: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
        weights = settings.layer_weights or geometry.layer_weights
        self.layer_weights = {
            int(index): float(weight)
            for index, weight in weights.items()
            if 0 <= int(index) < len(runtime.layers) and float(weight) > 0
        }
        if not self.layer_weights:
            self.layer_weights = {index: 1.0 for index in geometry.selected_layers}
        self.first_layer = min(self.layer_weights)

    def _tensors(
        self,
        geometry: TrajectoryGeometry,
        layer_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        key = (id(geometry), layer_index, str(device), str(dtype))
        if key in self.device_cache:
            return self.device_cache[key]
        layer = geometry.layers[layer_index]
        tensors = {
            "center": layer.center_positive.to(device=device, dtype=dtype),
            "q": layer.default_basis.to(device=device, dtype=dtype),
            "threshold": layer.default_threshold.to(device=device, dtype=dtype),
            "scale": layer.default_scale.to(device=device, dtype=dtype),
            "p": layer.positive_basis.to(device=device, dtype=dtype),
        }
        self.device_cache[key] = tensors
        return tensors

    def _proposal(
        self,
        state: torch.Tensor,
        geometry: TrajectoryGeometry,
        layer_index: int,
        layer_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tensors = self._tensors(geometry, layer_index, state.device, state.dtype)
        rank = max(1, min(self.settings.rank, tensors["q"].shape[1]))
        positive_rank = max(1, min(self.settings.positive_rank, tensors["p"].shape[1]))
        q = tensors["q"][:, :rank]
        threshold = tensors["threshold"][:rank]
        scale = tensors["scale"][:rank].clamp_min(1e-4)
        p = tensors["p"][:, :positive_rank]
        relative = state - tensors["center"]

        coefficients = relative @ q
        excess = torch.relu(coefficients - threshold)
        normalized_excess = excess / scale
        drift = normalized_excess.mean(dim=-1)
        default_delta = excess @ q.T

        projected = (relative @ p) @ p.T
        orthogonal = relative - projected
        orthogonal_norm = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
        radius = float(geometry.layers[layer_index].tube_radius)
        tube_excess = torch.relu(orthogonal_norm - radius)
        tube_delta = orthogonal * (tube_excess / orthogonal_norm.clamp_min(1e-6))

        gate = torch.ones_like(drift)
        if self.settings.adaptive_gate:
            gate = torch.sigmoid(
                (drift - 1.0) / max(1e-3, self.settings.gate_temperature)
            )
        proposal = (
            self.settings.default_strength
            * layer_weight
            * gate.unsqueeze(-1)
            * default_delta
            + self.settings.tube_strength * layer_weight * tube_delta
        )
        fallback = (relative @ q) @ q.T
        fallback_norm = torch.linalg.vector_norm(fallback, dim=-1, keepdim=True)
        first_q = q[:, 0].unsqueeze(0).expand_as(state)
        fallback = torch.where(fallback_norm > 1e-8, fallback, first_q)
        return proposal, {
            "drift": drift,
            "gate": gate,
            "tube_excess": tube_excess.squeeze(-1),
            "fallback": fallback,
        }

    def _apply_direction_at_ratio(
        self,
        state: torch.Tensor,
        direction: torch.Tensor,
        target_ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-6)
        direction_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        unit = direction / direction_norm.clamp_min(1e-8)
        target_ratio = target_ratio.reshape(-1, 1).clamp(0.0, self.settings.max_delta_ratio)
        pre_delta = unit * (target_ratio * state_norm)

        # RMS preservation slightly changes the realized perturbation norm. A few scalar
        # correction iterations make the logged, post-RMS delta match the target schedule.
        for _ in range(3):
            candidate = state - pre_delta
            if self.settings.preserve_rms:
                old_rms = torch.sqrt(state.square().mean(dim=-1, keepdim=True).clamp_min(1e-8))
                new_rms = torch.sqrt(candidate.square().mean(dim=-1, keepdim=True).clamp_min(1e-8))
                candidate = candidate * (old_rms / new_rms)
            actual_delta = state - candidate
            actual_norm = torch.linalg.vector_norm(actual_delta, dim=-1, keepdim=True)
            desired_norm = target_ratio * state_norm
            correction = torch.where(
                desired_norm > 0,
                desired_norm / actual_norm.clamp_min(1e-8),
                torch.zeros_like(desired_norm),
            )
            pre_delta = pre_delta * correction

        new_state = state - pre_delta
        if self.settings.preserve_rms:
            old_rms = torch.sqrt(state.square().mean(dim=-1, keepdim=True).clamp_min(1e-8))
            new_rms = torch.sqrt(new_state.square().mean(dim=-1, keepdim=True).clamp_min(1e-8))
            new_state = new_state * (old_rms / new_rms)
        actual_delta = state - new_state
        return new_state, actual_delta

    def _make_hook(self, layer_index: int):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            hidden, replace = hidden_from_output(output)
            if hidden is None or hidden.ndim != 3:
                return output
            if layer_index == self.first_layer:
                self.step += 1
            current_step = max(0, self.step)

            # Stay in model dtype. Forcing fp32 + .item() syncs every layer/token is
            # especially hostile on MPS (Metal command-buffer sync storms).
            state = hidden[:, -1, :]
            state_norm = torch.linalg.vector_norm(state, dim=-1).clamp_min(1e-6)
            in_token_window = current_step >= self.settings.token_start and (
                self.settings.token_end is None or current_step <= self.settings.token_end
            )
            layer_weight = self.layer_weights.get(layer_index, 0.0)
            has_effect = self.settings.default_strength > 0 or self.settings.tube_strength > 0

            proposal, diagnostics = self._proposal(
                state,
                self.geometry,
                layer_index,
                layer_weight,
            )
            proposed_ratio = torch.linalg.vector_norm(proposal, dim=-1) / state_norm
            schedule_key = (current_step, layer_index)
            schedule_hit = schedule_key in self.target_delta_ratio_schedule
            magnitude_source = "native"

            target_ratio: torch.Tensor
            direction = proposal
            if self.target_delta_ratio_schedule:
                if schedule_hit:
                    target_ratio = torch.full_like(
                        state_norm,
                        clamp(
                            self.target_delta_ratio_schedule[schedule_key],
                            0.0,
                            self.settings.max_delta_ratio,
                        ),
                    )
                    magnitude_source = "paired-learned-schedule"
                else:
                    # Do not give a longer random continuation extra intervention exposure.
                    target_ratio = torch.zeros_like(state_norm)
                    magnitude_source = "outside-paired-schedule-zero"
            elif self.magnitude_reference_geometry is not None:
                reference_proposal, _ = self._proposal(
                    state,
                    self.magnitude_reference_geometry,
                    layer_index,
                    layer_weight,
                )
                target_ratio = (
                    torch.linalg.vector_norm(reference_proposal, dim=-1) / state_norm
                ).clamp(0.0, self.settings.max_delta_ratio)
                magnitude_source = "same-state-learned-reference"
            else:
                target_ratio = proposed_ratio.clamp(0.0, self.settings.max_delta_ratio)

            proposal_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
            direction = torch.where(
                proposal_norm > 1e-8,
                direction,
                diagnostics["fallback"],
            )

            actual_delta = torch.zeros_like(state)
            target_for_trace = torch.zeros_like(state_norm)
            applied = bool(in_token_window and layer_weight > 0 and has_effect)
            if applied:
                target_for_trace = target_ratio
                new_state, actual_delta = self._apply_direction_at_ratio(
                    state,
                    direction,
                    target_ratio,
                )
                # Edit only the last token in-place when safe. Cloning the full
                # prefill activation every hooked layer is a major MPS allocator spike.
                last_state = new_state.to(dtype=hidden.dtype)
                if hidden.requires_grad:
                    new_hidden = hidden.clone()
                    new_hidden[:, -1, :] = last_state
                    output = replace(new_hidden)
                else:
                    hidden[:, -1, :].copy_(last_state)
                    output = replace(hidden)

            delta_ratio = torch.linalg.vector_norm(actual_delta, dim=-1) / state_norm
            match_error = torch.where(
                target_for_trace > 1e-8,
                (delta_ratio - target_for_trace).abs() / target_for_trace.clamp_min(1e-8),
                delta_ratio.abs(),
            )
            # Defer host syncs: keep 0-d tensors here and flush once in __exit__.
            self._pending_trace.append(
                {
                    "step": current_step,
                    "layer_index": layer_index,
                    "layer_fraction": layer_index / max(1, len(self.runtime.layers) - 1),
                    "layer_weight": layer_weight,
                    "drift": diagnostics["drift"].detach().mean(),
                    "drift_max": diagnostics["drift"].detach().max(),
                    "gate": diagnostics["gate"].detach().mean(),
                    "tube_excess": diagnostics["tube_excess"].detach().mean(),
                    "proposed_delta_ratio": proposed_ratio.detach().mean(),
                    "target_delta_ratio": target_for_trace.detach().mean(),
                    "delta_ratio": delta_ratio.detach().mean(),
                    "delta_ratio_max": delta_ratio.detach().max(),
                    "magnitude_match_error": match_error.detach().mean(),
                    "magnitude_source": magnitude_source,
                    "schedule_hit": bool(schedule_hit),
                    "applied_requested": applied,
                    "label": self.settings.label,
                }
            )
            return output
        return hook

    def _flush_pending_trace(self) -> None:
        if not self._pending_trace:
            return
        scalar_keys = (
            "drift",
            "drift_max",
            "gate",
            "tube_excess",
            "proposed_delta_ratio",
            "target_delta_ratio",
            "delta_ratio",
            "delta_ratio_max",
            "magnitude_match_error",
        )
        stacked = {
            key: torch.stack(
                [row[key].reshape(()) for row in self._pending_trace]
            ).detach().cpu()
            for key in scalar_keys
        }
        for index, row in enumerate(self._pending_trace):
            delta_ratio = float(stacked["delta_ratio"][index])
            self.trace.append(
                {
                    "step": int(row["step"]),
                    "layer_index": int(row["layer_index"]),
                    "layer_fraction": float(row["layer_fraction"]),
                    "layer_weight": float(row["layer_weight"]),
                    "drift": float(stacked["drift"][index]),
                    "drift_max": float(stacked["drift_max"][index]),
                    "gate": float(stacked["gate"][index]),
                    "tube_excess": float(stacked["tube_excess"][index]),
                    "proposed_delta_ratio": float(stacked["proposed_delta_ratio"][index]),
                    "target_delta_ratio": float(stacked["target_delta_ratio"][index]),
                    "delta_ratio": delta_ratio,
                    "magnitude_match_error": float(stacked["magnitude_match_error"][index]),
                    "magnitude_source": row["magnitude_source"],
                    "schedule_hit": bool(row["schedule_hit"]),
                    "applied": bool(
                        row["applied_requested"] and float(stacked["delta_ratio_max"][index]) > 0
                    ),
                    "label": row["label"],
                }
            )
        self._pending_trace.clear()

    def __enter__(self) -> "TrajectoryIntervention":
        for layer_index in sorted(self.layer_weights):
            self.handles.append(
                self.runtime.layers[layer_index].register_forward_hook(self._make_hook(layer_index))
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        try:
            self._flush_pending_trace()
        finally:
            self.device_cache.clear()
            empty_accelerator_cache()

    def delta_ratio_schedule(self) -> dict[tuple[int, int], float]:
        self._flush_pending_trace()
        # Include zero-magnitude rows as explicit targets. Otherwise a random control
        # could receive a non-zero fallback intervention where the learned arm applied none.
        return {
            (int(row["step"]), int(row["layer_index"])): float(row["delta_ratio"])
            for row in self.trace
        }

    def summary(self) -> dict[str, Any]:
        self._flush_pending_trace()
        if not self.trace:
            return {
                "mean_drift": 0.0,
                "max_drift": 0.0,
                "mean_delta_ratio": 0.0,
                "mean_applied_delta_ratio": 0.0,
                "max_delta_ratio": 0.0,
                "mean_target_delta_ratio": 0.0,
                "mean_magnitude_match_error": 0.0,
                "max_magnitude_match_error": 0.0,
                "schedule_coverage": 0.0,
                "top_drift_layer": None,
                "trace_rows": 0,
            }
        drifts = np.array([row["drift"] for row in self.trace], dtype=float)
        deltas = np.array([row["delta_ratio"] for row in self.trace], dtype=float)
        targets = np.array([row.get("target_delta_ratio", 0.0) for row in self.trace], dtype=float)
        errors = np.array([row.get("magnitude_match_error", 0.0) for row in self.trace], dtype=float)
        applied_rows = [row for row in self.trace if row.get("applied")]
        applied_deltas = np.array(
            [row["delta_ratio"] for row in applied_rows], dtype=float
        ) if applied_rows else np.array([], dtype=float)
        by_layer: dict[int, list[float]] = defaultdict(list)
        for row in self.trace:
            by_layer[int(row["layer_index"])].append(float(row["drift"]))
        top_layer = max(by_layer, key=lambda index: float(np.mean(by_layer[index])))
        if self.target_delta_ratio_schedule:
            trace_keys = {
                (int(row["step"]), int(row["layer_index"])) for row in self.trace
            }
            schedule_coverage = len(
                set(self.target_delta_ratio_schedule) & trace_keys
            ) / max(1, len(self.target_delta_ratio_schedule))
        else:
            schedule_coverage = 0.0
        return {
            "mean_drift": float(np.mean(drifts)),
            "max_drift": float(np.max(drifts)),
            "mean_delta_ratio": float(np.mean(deltas)),
            "mean_applied_delta_ratio": float(np.mean(applied_deltas)) if applied_deltas.size else 0.0,
            "max_delta_ratio": float(np.max(deltas)),
            "mean_target_delta_ratio": float(np.mean(targets)),
            "mean_magnitude_match_error": float(np.mean(errors)),
            "max_magnitude_match_error": float(np.max(errors)),
            "schedule_coverage": float(schedule_coverage),
            "top_drift_layer": int(top_layer),
            "trace_rows": len(self.trace),
        }


def layer_weights_from_segments(
    segments: Sequence[Mapping[str, Any]],
    n_layers: int,
    fallback: Mapping[int, float],
) -> dict[int, float]:
    weights: dict[int, float] = {}
    for segment in list(segments)[:4]:
        start = clamp(safe_float(segment.get("start_fraction"), 0.0), 0.0, 1.0)
        end = clamp(safe_float(segment.get("end_fraction"), 1.0), 0.0, 1.0)
        if end < start:
            start, end = end, start
        multiplier = clamp(safe_float(segment.get("multiplier"), 1.0), 0.0, 1.5)
        start_index = min(n_layers - 1, int(math.floor(start * n_layers)))
        end_index = min(n_layers - 1, int(math.ceil(end * n_layers)) - 1)
        for layer_index in range(start_index, end_index + 1):
            weights[layer_index] = max(weights.get(layer_index, 0.0), multiplier)
    if not weights:
        return {int(index): float(value) for index, value in fallback.items()}
    return weights


# -----------------------------------------------------------------------------
# Quality proxy and controller parsing
# -----------------------------------------------------------------------------


DEFAULT_SCORE = {
    "novelty": 5.0,
    "cross_domain_integration": 5.0,
    "mechanistic_coherence": 5.0,
    "semantic_integrity": 5.0,
    "plausibility": 5.0,
    "testability": 5.0,
    "specificity": 5.0,
    "nontriviality": 5.0,
    "overall": 5.0,
    "confidence": 0.25,
    "brief_evidence": "Judge output could not be parsed.",
    "failure_modes": [],
    "missing_fields": [],
    "valid_json": False,
}


def sanitize_score(parsed: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_SCORE)
    if parsed is None:
        return result
    score_fields = [
        "novelty",
        "cross_domain_integration",
        "mechanistic_coherence",
        "semantic_integrity",
        "plausibility",
        "testability",
        "specificity",
        "nontriviality",
        "overall",
    ]
    missing_fields = [field_name for field_name in score_fields if field_name not in parsed]
    if "confidence" not in parsed:
        missing_fields.append("confidence")
    for field_name in score_fields:
        result[field_name] = clamp(safe_float(parsed.get(field_name), 5.0), 0.0, 10.0)
    result["confidence"] = clamp(safe_float(parsed.get("confidence"), 0.5), 0.0, 1.0)
    result["brief_evidence"] = str(parsed.get("brief_evidence", ""))[:1200]
    failure_modes = parsed.get("failure_modes", [])
    if not isinstance(failure_modes, list):
        failure_modes = [str(failure_modes)]
    result["failure_modes"] = [str(item)[:300] for item in failure_modes[:8]]
    result["missing_fields"] = missing_fields
    result["valid_json"] = not missing_fields
    return result


def judge_candidate(
    runtime: ModelRuntime,
    task: str,
    candidate: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], str]:
    rubric = {
        "novelty": (
            "0-10: the compositional link is non-obvious relative to common AI treatments, "
            "without assuming true historical priority"
        ),
        "cross_domain_integration": (
            "0-10: known ingredients are genuinely connected by a non-obvious mechanistic "
            "link (not a forced mashup), and that link yields a concrete AI-research payoff"
        ),
        "mechanistic_coherence": (
            "0-10: the mapping and composed mechanism form a coherent causal story for an "
            "AI system or method"
        ),
        "semantic_integrity": (
            "0-10: readable, internally consistent, complete prose rather than word salad, "
            "fragmentation, or novelty-by-degradation"
        ),
        "plausibility": (
            "0-10: compatible with known AI/ML constraints; unsupported leaps and invented "
            "primitives are penalized"
        ),
        "testability": (
            "0-10: produces discriminating AI-system predictions and feasible lab falsifiers"
        ),
        "specificity": "0-10: concrete enough for an AI researcher to implement or investigate",
        "nontriviality": (
            "0-10: more than rewording, loose analogy, generic brainstorming, a domain-count "
            "checklist, or non-AI science with an AI veneer"
        ),
        "overall": "0-10: conservative aggregate of AI-research value from the composition",
        "confidence": "0-1 confidence in this evaluation",
        "brief_evidence": "one short externally checkable explanation",
        "failure_modes": "short list of the most important weaknesses",
    }
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                "TASK:\n"
                + task[:6000]
                + "\n\nCANDIDATE:\n"
                + candidate[:12000]
                + "\n\nReturn exactly one JSON object with these fields and scales:\n"
                + json.dumps(rubric, indent=2)
            ),
        },
    ]
    result = generate_text(
        runtime,
        messages,
        seed=seed,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=0.15,
        top_p=0.9,
        repetition_penalty=1.0,
        store_raw=args.store_raw_generations,
    )
    parsed, _ = extract_json_object(result.text)
    return sanitize_score(parsed), result.text


def surface_degradation_penalty(text: str) -> float:
    words = normalize_words(text)
    if not words:
        return 5.0
    penalty = 0.0
    word_count = len(words)
    if word_count < 80:
        penalty += (80 - word_count) / 80.0
    printable = sum(char.isprintable() for char in text) / max(1, len(text))
    penalty += 2.0 * max(0.0, 0.98 - printable)
    symbol_fraction = sum(
        not (char.isalnum() or char.isspace() or char in ".,;:!?()[]{}'\"-/%")
        for char in text
    ) / max(1, len(text))
    penalty += 3.0 * max(0.0, symbol_fraction - 0.03)
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if lines:
        repeated_line_fraction = 1.0 - len(set(lines)) / len(lines)
        penalty += 2.0 * repeated_line_fraction
    punctuation_runs = len(re.findall(r"[!?.,;:_=-]{5,}", text))
    penalty += min(1.5, 0.25 * punctuation_runs)
    return float(clamp(penalty, 0.0, 5.0))


def candidate_reward_details(
    score: Mapping[str, Any],
    diversity: float,
    repetition: float,
    text: str = "",
) -> dict[str, float]:
    coherence = safe_float(score.get("mechanistic_coherence"), 0.0)
    semantic_integrity = safe_float(score.get("semantic_integrity"), 0.0)
    plausibility = safe_float(score.get("plausibility"), 0.0)
    testability = safe_float(score.get("testability"), 0.0)
    quality_floor = min(coherence, semantic_integrity, plausibility, testability)
    # Novelty receives credit only to the extent that the candidate also clears the
    # weakest core-quality dimension. Trigram distance is diagnostic only.
    gated_novelty = min(safe_float(score.get("novelty"), 0.0), quality_floor)
    degradation = surface_degradation_penalty(text)
    components = {
        "overall": 0.20 * safe_float(score.get("overall"), 0.0),
        "cross_domain_integration": 0.09
        * safe_float(score.get("cross_domain_integration"), 0.0),
        "mechanistic_coherence": 0.17 * coherence,
        "semantic_integrity": 0.12 * semantic_integrity,
        "plausibility": 0.16 * plausibility,
        "testability": 0.16 * testability,
        "specificity": 0.05 * safe_float(score.get("specificity"), 0.0),
        "quality_gated_novelty": 0.05 * gated_novelty,
        "lexical_novelty": 0.0 * diversity,
        "repetition_penalty": -2.0 * repetition,
        "surface_degradation_penalty": -1.5 * degradation,
        "quality_floor_penalty": -0.35 * max(0.0, 5.0 - quality_floor),
    }
    components["total"] = float(sum(components.values()))
    return {key: float(value) for key, value in components.items()}


def candidate_reward(
    score: Mapping[str, Any],
    diversity: float,
    repetition: float,
    text: str = "",
) -> float:
    return candidate_reward_details(score, diversity, repetition, text)["total"]


# -----------------------------------------------------------------------------
# Prompt builders and calibration
# -----------------------------------------------------------------------------


def discovery_messages(task: str, mode: str, extra_focus: str = "") -> list[dict[str, str]]:
    if mode == "conventional":
        instruction = CONVENTIONAL_INSTRUCTION
    else:
        instruction = COMPOSITION_INSTRUCTION
    if extra_focus:
        instruction += "\n\nSEARCH FOCUS:\n" + extra_focus
    return [
        {"role": "system", "content": DISCOVERY_SYSTEM},
        {"role": "user", "content": f"TASK:\n{task}\n\nINSTRUCTION:\n{instruction}"},
    ]


def calibrate_model_task(
    runtime: ModelRuntime,
    task_id: str,
    task: str,
    args: argparse.Namespace,
    store: RecordStore,
) -> CalibrationBundle:
    cells: dict[str, list[CalibrationSample]] = {
        calibration_cell_key(instruction, condition): []
        for instruction in ("conventional", "composition")
        for condition in ("low", "high")
    }
    temperatures = {
        "low": float(args.calibration_low_temperature),
        "high": float(args.calibration_high_temperature),
    }

    # Fully crossed, seed-paired 2x2 design. The same replicate seed is used in all
    # four cells; only instruction and temperature vary. Geometry fitting does not
    # select examples by the source model's judge score.
    for sample_index in range(args.calibration_samples):
        paired_seed = args.seed + sample_index
        for temperature_condition in ("low", "high"):
            temperature = temperatures[temperature_condition]
            for instruction in ("conventional", "composition"):
                messages = discovery_messages(
                    task,
                    instruction,
                    extra_focus=(
                        f"Produce independent calibration replicate {sample_index + 1}. "
                        "Do not mention the calibration procedure."
                    ),
                )
                result = generate_text(
                    runtime,
                    messages,
                    seed=paired_seed,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    store_raw=args.store_raw_generations,
                )
                # This score is retained only as an online diagnostic. It is not used
                # to choose the positive set or fit the factorial instruction effect.
                score, judge_raw = judge_candidate(
                    runtime,
                    task,
                    result.text,
                    args,
                    paired_seed + 50_000,
                )
                states = capture_sequence_states(
                    runtime,
                    result.full_token_ids,
                    result.generated_tokens,
                    args.state_tokens,
                )
                arm = f"{instruction}-{temperature_condition}"
                sample_id = make_candidate_id(
                    runtime.spec.key,
                    f"calibration-{arm}",
                    task_id,
                )
                sample = CalibrationSample(
                    sample_id=sample_id,
                    instruction=instruction,
                    temperature_condition=temperature_condition,
                    temperature=temperature,
                    replicate_index=sample_index,
                    paired_seed=paired_seed,
                    text=result.text,
                    result=result,
                    score=score,
                    states=states,
                )
                cells[calibration_cell_key(instruction, temperature_condition)].append(sample)
                record = {
                    "candidate_id": sample_id,
                    "timestamp": utc_now(),
                    "task_id": task_id,
                    "task": task,
                    "model_key": runtime.spec.key,
                    "model_id": runtime.spec.model_id,
                    "method": "shared-calibration",
                    "stage": "calibration",
                    "arm": arm,
                    "instruction": instruction,
                    "temperature_condition": temperature_condition,
                    "temperature": temperature,
                    "replicate_index": sample_index,
                    "paired_seed": paired_seed,
                    "seed": paired_seed,
                    "text": result.text,
                    "raw_text": result.raw_text,
                    "prompt_tokens": result.prompt_tokens,
                    "generated_tokens": result.generated_tokens,
                    "generation_seconds": result.seconds,
                    "scores": score,
                    "judge_raw": judge_raw,
                    "evaluation_role": "source-model-online-diagnostic-only",
                    "used_for_geometry": True,
                    "geometry_selection_weight": 1.0,
                }
                store.add("candidates", record)
                LOGGER.info(
                    "%s | %s | calibration %s/%s %d/%d | overall %.2f",
                    runtime.spec.key,
                    task_id,
                    instruction,
                    temperature_condition,
                    sample_index + 1,
                    args.calibration_samples,
                    score["overall"],
                )

    conventional = [
        sample
        for condition in ("low", "high")
        for sample in cells[calibration_cell_key("conventional", condition)]
    ]
    composition = [
        sample
        for condition in ("low", "high")
        for sample in cells[calibration_cell_key("composition", condition)]
    ]
    return CalibrationBundle(
        cells=cells,
        conventional=conventional,
        exploratory=composition,
        positive=composition,
        negative=conventional,
    )


# -----------------------------------------------------------------------------
# Method 1: Contrastive Trajectory Inhibition
# -----------------------------------------------------------------------------


def store_trace_rows(
    store: RecordStore,
    trace: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> None:
    for row in trace:
        store.add("intervention_traces", {**context, **dict(row)})


def run_cti_method(
    runtime: ModelRuntime,
    task_id: str,
    task: str,
    calibration: CalibrationBundle,
    learned_geometry: TrajectoryGeometry,
    random_geometry: TrajectoryGeometry,
    args: argparse.Namespace,
    store: RecordStore,
) -> list[dict[str, Any]]:
    conventional_references = [sample.text for sample in calibration.conventional]
    outputs: list[dict[str, Any]] = []

    def run_arm(
        arm: str,
        geometry: TrajectoryGeometry,
        default_strength: float,
        tube_strength: float,
        candidate_index: int,
        seed: int,
        target_schedule: Mapping[tuple[int, int], float] | None = None,
        magnitude_reference: TrajectoryGeometry | None = None,
    ) -> tuple[dict[str, Any], TrajectoryIntervention]:
        settings = InterventionSettings(
            default_strength=default_strength,
            tube_strength=tube_strength,
            max_delta_ratio=args.max_delta_ratio,
            rank=learned_geometry.rank,
            positive_rank=learned_geometry.positive_rank,
            token_start=args.intervention_token_start,
            token_end=args.intervention_token_end,
            adaptive_gate=not args.disable_adaptive_gate,
            gate_temperature=args.gate_temperature,
            preserve_rms=not args.disable_rms_preservation,
            label=arm,
            magnitude_match_tolerance=args.magnitude_match_tolerance,
        )
        intervention = TrajectoryIntervention(
            runtime,
            geometry,
            settings,
            magnitude_reference_geometry=magnitude_reference,
            target_delta_ratio_schedule=target_schedule,
        )
        messages = discovery_messages(task, "composition")
        with intervention:
            result = generate_text(
                runtime,
                messages,
                seed=seed,
                max_new_tokens=args.max_new_tokens,
                temperature=args.search_temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                store_raw=args.store_raw_generations,
            )
        score, judge_raw = judge_candidate(runtime, task, result.text, args, seed + 50_000)
        diversity = lexical_novelty(result.text, conventional_references)
        repetition = repetition_ratio(result.text)
        reward_details = candidate_reward_details(score, diversity, repetition, result.text)
        reward = reward_details["total"]
        candidate_id = make_candidate_id(runtime.spec.key, "cti", task_id)
        trace_summary = intervention.summary()
        record = {
            "candidate_id": candidate_id,
            "timestamp": utc_now(),
            "task_id": task_id,
            "task": task,
            "model_key": runtime.spec.key,
            "model_id": runtime.spec.model_id,
            "method": "contrastive-trajectory-inhibition",
            "stage": "search",
            "arm": arm,
            "candidate_index": candidate_index,
            "pair_index": candidate_index,
            "seed": seed,
            "paired_seed": seed,
            "text": result.text,
            "raw_text": result.raw_text,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "generation_seconds": result.seconds,
            "scores": score,
            "judge_raw": judge_raw,
            "evaluation_role": "source-model-online-diagnostic-only",
            "lexical_novelty": diversity,
            "repetition_ratio": repetition,
            "surface_degradation_penalty": surface_degradation_penalty(result.text),
            "reward": reward,
            "reward_components": reward_details,
            "intervention": {
                "geometry": geometry.geometry_name if (default_strength or tube_strength) else "none",
                "selected_layers": geometry.selected_layers,
                "layer_weights": geometry.layer_weights,
                "default_strength": default_strength,
                "tube_strength": tube_strength,
                "max_delta_ratio": args.max_delta_ratio,
                "rank": geometry.rank,
                "positive_rank": geometry.positive_rank,
                "token_start": args.intervention_token_start,
                "token_end": args.intervention_token_end,
                "magnitude_match": bool(target_schedule),
            },
            "trace_summary": trace_summary,
            "reportable_eligible": True,
        }
        return record, intervention

    # Candidate-seed outer order preserves exact pairing and lets the random arm use
    # the learned arm's realized post-RMS delta schedule at every layer and token step.
    for candidate_index in range(args.candidates_per_arm):
        seed = args.seed + 100_000 + candidate_index
        baseline_record, baseline_intervention = run_arm(
            "baseline",
            learned_geometry,
            0.0,
            0.0,
            candidate_index,
            seed,
        )
        learned_record, learned_intervention = run_arm(
            "learned-inhibition",
            learned_geometry,
            args.inhibition_strength,
            args.tube_strength,
            candidate_index,
            seed,
        )
        learned_schedule = learned_intervention.delta_ratio_schedule()
        random_record, random_intervention = run_arm(
            "random-subspace-control",
            random_geometry,
            args.inhibition_strength,
            args.tube_strength,
            candidate_index,
            seed,
            target_schedule=learned_schedule,
            magnitude_reference=learned_geometry,
        )

        random_realized = {
            (int(row["step"]), int(row["layer_index"])): float(row["delta_ratio"])
            for row in random_intervention.trace
            if row.get("schedule_hit")
        }
        common_keys = sorted(set(learned_schedule) & set(random_realized))
        coverage = len(common_keys) / max(1, len(learned_schedule))
        relative_errors = [
            abs(random_realized[key] - learned_schedule[key])
            / max(1e-8, abs(learned_schedule[key]))
            for key in common_keys
            if abs(learned_schedule[key]) > 1e-8
        ]
        absolute_errors = [
            abs(random_realized[key] - learned_schedule[key]) for key in common_keys
        ]
        zero_target_errors = [
            abs(random_realized[key])
            for key in common_keys
            if abs(learned_schedule[key]) <= 1e-8
        ]
        max_relative_error = max(relative_errors, default=0.0)
        mean_relative_error = float(np.mean(relative_errors)) if relative_errors else 0.0
        max_zero_target_error = max(zero_target_errors, default=0.0)
        audit_pass = bool(
            coverage >= args.minimum_magnitude_schedule_coverage
            and max_relative_error <= args.magnitude_match_tolerance
            and max_zero_target_error <= 1e-7
        )
        audit_id = f"{runtime.spec.key}-{task_id}-pair-{candidate_index}"
        magnitude_audit = {
            "timestamp": utc_now(),
            "audit_id": audit_id,
            "task_id": task_id,
            "model_key": runtime.spec.key,
            "method": "contrastive-trajectory-inhibition",
            "pair_index": candidate_index,
            "seed": seed,
            "learned_schedule_rows": len(learned_schedule),
            "matched_rows": len(common_keys),
            "coverage": coverage,
            "learned_mean_delta_ratio": float(np.mean(list(learned_schedule.values())))
            if learned_schedule
            else 0.0,
            "random_mean_delta_ratio_on_common": float(
                np.mean([random_realized[key] for key in common_keys])
            )
            if common_keys
            else 0.0,
            "mean_absolute_error": float(np.mean(absolute_errors)) if absolute_errors else 0.0,
            "mean_relative_error": mean_relative_error,
            "max_relative_error": max_relative_error,
            "max_zero_target_absolute_error": max_zero_target_error,
            "tolerance": args.magnitude_match_tolerance,
            "minimum_coverage": args.minimum_magnitude_schedule_coverage,
            "pass": audit_pass,
        }
        store.add("magnitude_audits", magnitude_audit)
        for record in (baseline_record, learned_record, random_record):
            record["magnitude_audit_id"] = audit_id
            record["magnitude_match_pass"] = audit_pass
        if args.strict_magnitude_match and not audit_pass:
            random_record["reportable_eligible"] = False
            LOGGER.error(
                "%s | %s | pair %d magnitude audit failed: coverage=%.3f max_rel_error=%.4f",
                runtime.spec.key,
                task_id,
                candidate_index,
                coverage,
                max_relative_error,
            )

        for record, intervention in (
            (baseline_record, baseline_intervention),
            (learned_record, learned_intervention),
            (random_record, random_intervention),
        ):
            store.add("candidates", record)
            store_trace_rows(
                store,
                intervention.trace,
                {
                    "candidate_id": record["candidate_id"],
                    "timestamp": utc_now(),
                    "task_id": task_id,
                    "model_key": runtime.spec.key,
                    "model_id": runtime.spec.model_id,
                    "method": "contrastive-trajectory-inhibition",
                    "arm": record["arm"],
                    "pair_index": candidate_index,
                    "seed": seed,
                },
            )
            outputs.append(record)
            LOGGER.info(
                "%s | %s | CTI %s %d/%d | overall %.2f | reward %.2f",
                runtime.spec.key,
                task_id,
                record["arm"],
                candidate_index + 1,
                args.candidates_per_arm,
                record["scores"]["overall"],
                record["reward"],
            )
    return outputs


# -----------------------------------------------------------------------------
# Method 2: Autonomous Granular Controller
# -----------------------------------------------------------------------------


@dataclass
class ControllerAction:
    action: str
    focus: str
    parent_index: int
    geometry: str
    layer_segments: list[dict[str, float]]
    rank: int
    inhibition_strength: float
    tube_strength: float
    max_delta_ratio: float
    token_start: int
    token_end: int | None
    temperature: float
    top_p: float
    num_branches: int
    horizon_tokens: int
    answer: str
    brief_reason: str
    valid_json: bool


def default_controller_action(round_index: int, args: argparse.Namespace) -> ControllerAction:
    if round_index == 0:
        return ControllerAction(
            action="explore",
            focus=(
                "Find a non-obvious link among already-known ideas with a concrete AI "
                "payoff, and make the falsifier unusually concrete."
            ),
            parent_index=0,
            geometry="learned",
            layer_segments=[],
            rank=args.trajectory_rank,
            inhibition_strength=min(args.inhibition_strength, 0.45),
            tube_strength=args.tube_strength,
            max_delta_ratio=args.max_delta_ratio,
            token_start=0,
            token_end=None,
            temperature=args.search_temperature,
            top_p=args.top_p,
            num_branches=args.controller_branches,
            horizon_tokens=args.controller_horizon_tokens,
            answer="",
            brief_reason="Fallback exploration action.",
            valid_json=False,
        )
    return ControllerAction(
        action="refine" if round_index % 2 else "diversify",
        focus="Improve causal specificity and produce a prediction that separates the proposal from its nearest alternative.",
        parent_index=0,
        geometry="learned",
        layer_segments=[],
        rank=args.trajectory_rank,
        inhibition_strength=args.inhibition_strength,
        tube_strength=args.tube_strength,
        max_delta_ratio=args.max_delta_ratio,
        token_start=0,
        token_end=None,
        temperature=args.search_temperature,
        top_p=args.top_p,
        num_branches=args.controller_branches,
        horizon_tokens=args.controller_horizon_tokens,
        answer="",
        brief_reason="Fallback iterative action.",
        valid_json=False,
    )


def sanitize_controller_action(
    parsed: Mapping[str, Any] | None,
    round_index: int,
    geometry: TrajectoryGeometry,
    args: argparse.Namespace,
) -> ControllerAction:
    fallback = default_controller_action(round_index, args)
    if parsed is None:
        return fallback
    action = str(parsed.get("action", fallback.action)).strip().lower()
    allowed = {"explore", "refine", "diversify", "halt", "answer"}
    if action not in allowed:
        action = fallback.action
    geometry_name = str(parsed.get("geometry", fallback.geometry)).strip().lower()
    if geometry_name not in {"learned", "random", "none"}:
        geometry_name = "learned"
    segments_raw = parsed.get("layer_segments", [])
    if not isinstance(segments_raw, list):
        segments_raw = []
    segments: list[dict[str, float]] = []
    for item in segments_raw[:4]:
        if not isinstance(item, Mapping):
            continue
        segments.append(
            {
                "start_fraction": clamp(safe_float(item.get("start_fraction"), 0.0), 0.0, 1.0),
                "end_fraction": clamp(safe_float(item.get("end_fraction"), 1.0), 0.0, 1.0),
                "multiplier": clamp(safe_float(item.get("multiplier"), 1.0), 0.0, 1.5),
            }
        )
    token_end_value = parsed.get("token_end", None)
    token_end = None if token_end_value in (None, -1, "-1", "none", "null") else max(0, safe_int(token_end_value, 0))
    return ControllerAction(
        action=action,
        focus=str(parsed.get("focus", fallback.focus))[:2500],
        parent_index=max(0, safe_int(parsed.get("parent_index"), 0)),
        geometry=geometry_name,
        layer_segments=segments,
        rank=max(1, min(safe_int(parsed.get("rank"), geometry.rank), geometry.rank)),
        inhibition_strength=clamp(
            safe_float(parsed.get("inhibition_strength"), fallback.inhibition_strength), 0.0, 1.5
        ),
        tube_strength=clamp(
            safe_float(parsed.get("tube_strength"), fallback.tube_strength), 0.0, 0.75
        ),
        max_delta_ratio=clamp(
            safe_float(parsed.get("max_delta_ratio"), fallback.max_delta_ratio), 0.0, 0.35
        ),
        token_start=max(0, safe_int(parsed.get("token_start"), 0)),
        token_end=token_end,
        temperature=clamp(
            safe_float(parsed.get("temperature"), fallback.temperature), 0.05, 1.6
        ),
        top_p=clamp(safe_float(parsed.get("top_p"), fallback.top_p), 0.40, 1.0),
        num_branches=max(
            1,
            min(
                safe_int(parsed.get("num_branches"), fallback.num_branches),
                args.controller_max_branches,
            ),
        ),
        horizon_tokens=max(
            32,
            min(
                safe_int(parsed.get("horizon_tokens"), fallback.horizon_tokens),
                args.max_new_tokens,
            ),
        ),
        answer=str(parsed.get("answer", ""))[:16000],
        brief_reason=str(parsed.get("brief_reason", ""))[:1000],
        valid_json=True,
    )


def compact_candidate_state(candidate: Mapping[str, Any], text_limit: int = 2200) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "round": candidate.get("round"),
        "arm": candidate.get("arm"),
        "reward": candidate.get("reward"),
        "scores": candidate.get("scores"),
        "trace_summary": candidate.get("trace_summary"),
        "intervention": candidate.get("intervention"),
        "controller_action": candidate.get("controller_action"),
        "text": str(candidate.get("text", ""))[:text_limit],
    }


def request_controller_action(
    runtime: ModelRuntime,
    task: str,
    pool: Sequence[Mapping[str, Any]],
    round_index: int,
    learned_geometry: TrajectoryGeometry,
    args: argparse.Namespace,
    run_seed: int,
) -> tuple[ControllerAction, str]:
    top_pool = sorted(pool, key=lambda row: safe_float(row.get("reward"), -999.0), reverse=True)[
        : args.controller_beam_width
    ]
    top_signal = sorted(
        (
            (index, layer.smoothed_signal)
            for index, layer in learned_geometry.layers.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    schema = {
        "action": "explore | refine | diversify | halt | answer",
        "focus": "short search/refinement instruction",
        "parent_index": "0-based index into BEST_CANDIDATES",
        "geometry": "learned | random | none",
        "layer_segments": [
            {
                "start_fraction": "0..1",
                "end_fraction": "0..1",
                "multiplier": "0..1.5",
            }
        ],
        "rank": f"1..{learned_geometry.rank}",
        "inhibition_strength": "0..1.5",
        "tube_strength": "0..0.75",
        "max_delta_ratio": "0..0.35",
        "token_start": "nonnegative generated-step index",
        "token_end": "nonnegative index or -1 for no end",
        "temperature": "0.05..1.6",
        "top_p": "0.40..1.0",
        "num_branches": f"1..{args.controller_max_branches}",
        "horizon_tokens": f"32..{args.max_new_tokens}",
        "answer": "required only for action=answer; optional for halt",
        "brief_reason": "one concise operational reason",
    }
    state = {
        "round": round_index,
        "num_layers": len(runtime.layers),
        "default_selected_layers": learned_geometry.selected_layers,
        "default_selected_fractions": [
            index / max(1, len(runtime.layers) - 1)
            for index in learned_geometry.selected_layers
        ],
        "top_signal_layers": [
            {
                "layer": index,
                "fraction": index / max(1, len(runtime.layers) - 1),
                "signal": signal,
            }
            for index, signal in top_signal
        ],
        "best_candidates": [compact_candidate_state(candidate) for candidate in top_pool],
        "recent_candidates": [
            compact_candidate_state(candidate)
            for candidate in list(pool)[-args.controller_beam_width :]
        ],
    }
    messages = [
        {"role": "system", "content": CONTROLLER_SYSTEM},
        {
            "role": "user",
            "content": (
                "TASK:\n"
                + task[:6000]
                + "\n\nCURRENT STATE:\n"
                + json.dumps(state, ensure_ascii=False, indent=2)
                + "\n\nCONTROL SCHEMA:\n"
                + json.dumps(schema, indent=2)
                + "\n\nChoose the next bounded action. Use HALT when further search is unlikely "
                "to help. Use ANSWER when you can provide the final candidate now. Return "
                "only the JSON object."
            ),
        },
    ]
    result = generate_text(
        runtime,
        messages,
        seed=run_seed + 700_000 + round_index,
        max_new_tokens=args.controller_max_new_tokens,
        temperature=0.20,
        top_p=0.9,
        repetition_penalty=1.0,
        store_raw=args.store_raw_generations,
    )
    parsed, _ = extract_json_object(result.text)
    return sanitize_controller_action(parsed, round_index, learned_geometry, args), result.text


def autonomous_branch_messages(
    task: str,
    action: ControllerAction,
    parent: Mapping[str, Any] | None,
    previous_texts: Sequence[str],
) -> list[dict[str, str]]:
    mode_instruction = {
        "explore": (
            "Create a new candidate by linking already-known ideas through a fresh, "
            "non-obvious compositional connection with an AI research payoff."
        ),
        "diversify": (
            "Use a substantially different set of known ingredients and linking hypothesis "
            "than the prior candidates, while keeping the payoff inside AI research."
        ),
        "refine": "Improve the selected parent candidate rather than merely lengthening it.",
    }.get(action.action, "Create the strongest candidate you can.")
    parent_text = str(parent.get("text", "")) if parent else ""
    previous_excerpt = "\n\n---\n\n".join(text[:1200] for text in previous_texts[-3:])
    user_content = (
        f"TASK:\n{task}\n\nOPERATION:\n{mode_instruction}\n\nCONTROLLER FOCUS:\n{action.focus}"
    )
    if parent_text:
        user_content += "\n\nPARENT CANDIDATE TO REFINE OR DEPART FROM:\n" + parent_text[:7000]
    if previous_excerpt and action.action in {"explore", "diversify"}:
        user_content += (
            "\n\nRECENT CANDIDATES TO AVOID DUPLICATING:\n" + previous_excerpt
        )
    user_content += "\n\n" + COMPOSITION_INSTRUCTION
    return [
        {"role": "system", "content": DISCOVERY_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def synthesize_final_answer(
    runtime: ModelRuntime,
    task: str,
    pool: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    seed: int,
) -> GenerationResult:
    top = sorted(pool, key=lambda row: safe_float(row.get("reward"), -999.0), reverse=True)[
        : args.controller_beam_width
    ]
    candidates = [compact_candidate_state(candidate, text_limit=4000) for candidate in top]
    messages = [
        {"role": "system", "content": DISCOVERY_SYSTEM},
        {
            "role": "user",
            "content": (
                "TASK:\n"
                + task
                + "\n\nTOP SEARCH CANDIDATES:\n"
                + json.dumps(candidates, ensure_ascii=False, indent=2)
                + "\n\nProduce one final, self-contained AI-research candidate. Preserve the "
                "strongest overlooked link among known ideas, keep the payoff scoped to "
                "advancing AI, resolve avoidable weaknesses, state uncertainty about whether "
                "the link is truly new, and include a discriminating falsification test an AI "
                "lab could run. Do not invent new primitives or claim verified novelty."
            ),
        },
    ]
    return generate_text(
        runtime,
        messages,
        seed=seed,
        max_new_tokens=args.max_new_tokens,
        temperature=0.35,
        top_p=0.9,
        repetition_penalty=args.repetition_penalty,
        store_raw=args.store_raw_generations,
    )


def run_autonomous_replicate(
    runtime: ModelRuntime,
    task_id: str,
    task: str,
    calibration: CalibrationBundle,
    learned_geometry: TrajectoryGeometry,
    random_geometry: TrajectoryGeometry,
    args: argparse.Namespace,
    store: RecordStore,
    replicate_index: int,
    run_seed: int,
) -> list[dict[str, Any]]:
    conventional_references = [sample.text for sample in calibration.conventional]
    pool: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    # Seed each controller replicate with one low- and one high-temperature composition
    # calibration output chosen by replicate index, not by the source model's judge.
    seed_samples: list[CalibrationSample] = []
    for condition in ("low", "high"):
        cell = calibration.cells[calibration_cell_key("composition", condition)]
        if cell:
            seed_samples.append(cell[replicate_index % len(cell)])
    for sample in seed_samples:
        diversity = lexical_novelty(sample.text, conventional_references)
        rep = repetition_ratio(sample.text)
        reward_details = candidate_reward_details(sample.score, diversity, rep, sample.text)
        pool.append(
            {
                "candidate_id": sample.sample_id,
                "round": -1,
                "arm": "calibration-seed",
                "text": sample.text,
                "scores": sample.score,
                "lexical_novelty": diversity,
                "repetition_ratio": rep,
                "reward": reward_details["total"],
                "reward_components": reward_details,
                "trace_summary": {},
            }
        )

    halted = False
    for round_index in range(args.controller_rounds):
        action, controller_raw = request_controller_action(
            runtime,
            task,
            pool,
            round_index,
            learned_geometry,
            args,
            run_seed=run_seed,
        )
        store.add(
            "controller_actions",
            {
                "timestamp": utc_now(),
                "task_id": task_id,
                "model_key": runtime.spec.key,
                "model_id": runtime.spec.model_id,
                "method": "autonomous-granular-controller",
                "replicate_index": replicate_index,
                "run_seed": run_seed,
                "round": round_index,
                "action": dataclasses.asdict(action),
                "controller_raw": controller_raw,
                "pool_size": len(pool),
            },
        )
        LOGGER.info(
            "%s | %s | AGC replicate %d round %d | action=%s geometry=%s strength=%.2f",
            runtime.spec.key,
            task_id,
            replicate_index,
            round_index,
            action.action,
            action.geometry,
            action.inhibition_strength,
        )

        if action.action in {"halt", "answer"}:
            if action.answer.strip():
                final_text = action.answer.strip()
                final_result = None
                final_seed = run_seed + 850_000 + round_index
            else:
                final_seed = run_seed + 850_000 + round_index
                final_result = synthesize_final_answer(
                    runtime,
                    task,
                    pool,
                    args,
                    seed=final_seed,
                )
                final_text = final_result.text
            score, judge_raw = judge_candidate(
                runtime,
                task,
                final_text,
                args,
                seed=run_seed + 860_000 + round_index,
            )
            diversity = lexical_novelty(final_text, conventional_references)
            rep = repetition_ratio(final_text)
            reward_details = candidate_reward_details(score, diversity, rep, final_text)
            candidate_id = make_candidate_id(runtime.spec.key, "agc-final", task_id)
            record = {
                "candidate_id": candidate_id,
                "timestamp": utc_now(),
                "task_id": task_id,
                "task": task,
                "model_key": runtime.spec.key,
                "model_id": runtime.spec.model_id,
                "method": "autonomous-granular-controller",
                "stage": "final",
                "arm": "controller-final",
                "termination_action": action.action,
                "round": round_index,
                "replicate_index": replicate_index,
                "pair_index": replicate_index,
                "seed": final_seed,
                "run_seed": run_seed,
                "text": final_text,
                "raw_text": final_result.raw_text if final_result else None,
                "prompt_tokens": final_result.prompt_tokens if final_result else None,
                "generated_tokens": final_result.generated_tokens if final_result else None,
                "generation_seconds": final_result.seconds if final_result else None,
                "scores": score,
                "judge_raw": judge_raw,
                "evaluation_role": "source-model-online-diagnostic-only",
                "lexical_novelty": diversity,
                "repetition_ratio": rep,
                "surface_degradation_penalty": surface_degradation_penalty(final_text),
                "reward": reward_details["total"],
                "reward_components": reward_details,
                "controller_action": dataclasses.asdict(action),
                "trace_summary": {},
                "reportable_eligible": True,
            }
            store.add("candidates", record)
            outputs.append(record)
            halted = True
            break

        ranked_pool = sorted(
            pool,
            key=lambda row: safe_float(row.get("reward"), -999.0),
            reverse=True,
        )
        parent = None
        if ranked_pool:
            parent = ranked_pool[min(action.parent_index, len(ranked_pool) - 1)]
        geometry = learned_geometry
        if action.geometry == "random":
            geometry = random_geometry
        layer_weights = layer_weights_from_segments(
            action.layer_segments,
            len(runtime.layers),
            learned_geometry.layer_weights,
        )
        if action.geometry == "none":
            default_strength = 0.0
            tube_strength = 0.0
        else:
            default_strength = action.inhibition_strength
            tube_strength = action.tube_strength

        for branch_index in range(action.num_branches):
            seed = run_seed + 800_000 + round_index * 1000 + branch_index
            settings = InterventionSettings(
                default_strength=default_strength,
                tube_strength=tube_strength,
                max_delta_ratio=action.max_delta_ratio,
                rank=action.rank,
                positive_rank=learned_geometry.positive_rank,
                token_start=action.token_start,
                token_end=action.token_end,
                adaptive_gate=not args.disable_adaptive_gate,
                gate_temperature=args.gate_temperature,
                preserve_rms=not args.disable_rms_preservation,
                layer_weights=layer_weights,
                label=f"controller-{action.geometry}",
                magnitude_match_tolerance=args.magnitude_match_tolerance,
            )
            intervention = TrajectoryIntervention(
                runtime,
                geometry,
                settings,
                magnitude_reference_geometry=(
                    learned_geometry if action.geometry == "random" else None
                ),
            )
            messages = autonomous_branch_messages(
                task,
                action,
                parent,
                [str(candidate.get("text", "")) for candidate in pool],
            )
            with intervention:
                result = generate_text(
                    runtime,
                    messages,
                    seed=seed,
                    max_new_tokens=action.horizon_tokens,
                    temperature=action.temperature,
                    top_p=action.top_p,
                    repetition_penalty=args.repetition_penalty,
                    store_raw=args.store_raw_generations,
                )
            score, judge_raw = judge_candidate(runtime, task, result.text, args, seed + 50_000)
            references = conventional_references + [
                str(candidate.get("text", "")) for candidate in pool
            ]
            diversity = lexical_novelty(result.text, references)
            rep = repetition_ratio(result.text)
            reward_details = candidate_reward_details(score, diversity, rep, result.text)
            candidate_id = make_candidate_id(runtime.spec.key, "agc", task_id)
            trace_summary = intervention.summary()
            record = {
                "candidate_id": candidate_id,
                "timestamp": utc_now(),
                "task_id": task_id,
                "task": task,
                "model_key": runtime.spec.key,
                "model_id": runtime.spec.model_id,
                "method": "autonomous-granular-controller",
                "stage": "search",
                "arm": action.action,
                "round": round_index,
                "replicate_index": replicate_index,
                "branch_index": branch_index,
                "parent_id": parent.get("candidate_id") if parent else None,
                "seed": seed,
                "run_seed": run_seed,
                "text": result.text,
                "raw_text": result.raw_text,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "generation_seconds": result.seconds,
                "scores": score,
                "judge_raw": judge_raw,
                "evaluation_role": "source-model-online-controller-signal-only",
                "lexical_novelty": diversity,
                "repetition_ratio": rep,
                "surface_degradation_penalty": surface_degradation_penalty(result.text),
                "reward": reward_details["total"],
                "reward_components": reward_details,
                "controller_action": dataclasses.asdict(action),
                "intervention": {
                    "geometry": action.geometry,
                    "layer_weights": layer_weights,
                    "default_strength": default_strength,
                    "tube_strength": tube_strength,
                    "max_delta_ratio": action.max_delta_ratio,
                    "rank": action.rank,
                    "positive_rank": learned_geometry.positive_rank,
                    "token_start": action.token_start,
                    "token_end": action.token_end,
                    "same_state_magnitude_reference": action.geometry == "random",
                },
                "trace_summary": trace_summary,
                "reportable_eligible": False,
            }
            store.add("candidates", record)
            store_trace_rows(
                store,
                intervention.trace,
                {
                    "candidate_id": candidate_id,
                    "timestamp": utc_now(),
                    "task_id": task_id,
                    "model_key": runtime.spec.key,
                    "model_id": runtime.spec.model_id,
                    "method": "autonomous-granular-controller",
                    "arm": action.action,
                    "replicate_index": replicate_index,
                    "round": round_index,
                },
            )
            pool.append(record)
            outputs.append(record)
            LOGGER.info(
                "%s | %s | AGC replicate %d round %d branch %d/%d | overall %.2f | reward %.2f",
                runtime.spec.key,
                task_id,
                replicate_index,
                round_index,
                branch_index + 1,
                action.num_branches,
                score["overall"],
                reward_details["total"],
            )

    if not halted:
        final_seed = run_seed + 899_000
        final_result = synthesize_final_answer(
            runtime,
            task,
            pool,
            args,
            seed=final_seed,
        )
        score, judge_raw = judge_candidate(
            runtime,
            task,
            final_result.text,
            args,
            seed=run_seed + 899_500,
        )
        diversity = lexical_novelty(final_result.text, conventional_references)
        rep = repetition_ratio(final_result.text)
        reward_details = candidate_reward_details(
            score,
            diversity,
            rep,
            final_result.text,
        )
        candidate_id = make_candidate_id(runtime.spec.key, "agc-final", task_id)
        record = {
            "candidate_id": candidate_id,
            "timestamp": utc_now(),
            "task_id": task_id,
            "task": task,
            "model_key": runtime.spec.key,
            "model_id": runtime.spec.model_id,
            "method": "autonomous-granular-controller",
            "stage": "final",
            "arm": "controller-final",
            "termination_action": "max-rounds-synthesis",
            "round": args.controller_rounds,
            "replicate_index": replicate_index,
            "pair_index": replicate_index,
            "seed": final_seed,
            "run_seed": run_seed,
            "text": final_result.text,
            "raw_text": final_result.raw_text,
            "prompt_tokens": final_result.prompt_tokens,
            "generated_tokens": final_result.generated_tokens,
            "generation_seconds": final_result.seconds,
            "scores": score,
            "judge_raw": judge_raw,
            "evaluation_role": "source-model-online-diagnostic-only",
            "lexical_novelty": diversity,
            "repetition_ratio": rep,
            "surface_degradation_penalty": surface_degradation_penalty(final_result.text),
            "reward": reward_details["total"],
            "reward_components": reward_details,
            "trace_summary": {},
            "reportable_eligible": True,
        }
        store.add("candidates", record)
        outputs.append(record)
    return outputs


def run_autonomous_method(
    runtime: ModelRuntime,
    task_id: str,
    task: str,
    calibration: CalibrationBundle,
    learned_geometry: TrajectoryGeometry,
    random_geometry: TrajectoryGeometry,
    args: argparse.Namespace,
    store: RecordStore,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for replicate_index in range(args.controller_replicates):
        run_seed = args.seed + 2_000_000 + replicate_index * 100_000
        outputs.extend(
            run_autonomous_replicate(
                runtime,
                task_id,
                task,
                calibration,
                learned_geometry,
                random_geometry,
                args,
                store,
                replicate_index=replicate_index,
                run_seed=run_seed,
            )
        )
    return outputs


# -----------------------------------------------------------------------------
# Blinded leave-source-out cross-model jury
# -----------------------------------------------------------------------------


def select_cross_judge_candidates(
    candidate_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in candidate_records:
        if not bool(record.get("reportable_eligible", False)):
            continue
        method = str(record.get("method"))
        stage = str(record.get("stage"))
        if method == "contrastive-trajectory-inhibition" and stage == "search":
            selected.append(dict(record))
        elif method == "autonomous-granular-controller" and stage == "final":
            selected.append(dict(record))
    return selected


def run_cross_jury(
    selected_candidates: Sequence[Mapping[str, Any]],
    model_keys: Sequence[str],
    args: argparse.Namespace,
    store: RecordStore,
) -> None:
    for judge_key in model_keys:
        runtime: ModelRuntime | None = None
        try:
            runtime = load_model_runtime(MODEL_SPECS[judge_key], args)
            candidates = list(selected_candidates)
            rng = random.Random(args.seed + int(hashlib.sha256(judge_key.encode("utf-8")).hexdigest()[:8], 16))
            rng.shuffle(candidates)
            for candidate in candidates:
                source_key = str(candidate.get("model_key"))
                if not args.include_self_in_jury and source_key == judge_key:
                    continue
                candidate_id = str(candidate.get("candidate_id"))
                judge_seed = args.seed + int(
                    hashlib.sha256(f"{judge_key}|{candidate_id}".encode("utf-8")).hexdigest()[:8],
                    16,
                ) % 1_000_000_000
                score, raw = judge_candidate(
                    runtime,
                    str(candidate.get("task", "")),
                    str(candidate.get("text", "")),
                    args,
                    seed=judge_seed,
                )
                store.add(
                    "cross_judgments",
                    {
                        "timestamp": utc_now(),
                        "candidate_id": candidate_id,
                        "task_id": candidate.get("task_id"),
                        "source_model_key": source_key,
                        "source_method": candidate.get("method"),
                        "source_arm": candidate.get("arm"),
                        "source_pair_index": candidate.get("pair_index"),
                        "judge_model_key": judge_key,
                        "judge_model_id": runtime.spec.model_id,
                        "judge_seed": judge_seed,
                        "blinded_to_source_arm": True,
                        "source_model_excluded": not args.include_self_in_jury,
                        "scores": score,
                        "judge_raw": raw,
                    },
                )
        except Exception as exc:
            store.add(
                "errors",
                {
                    "timestamp": utc_now(),
                    "stage": "cross-judge",
                    "judge_model_key": judge_key,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            LOGGER.exception("Cross-judge model %s failed", judge_key)
        finally:
            unload_model_runtime(runtime)


JURY_SCORE_FIELDS = [
    "novelty",
    "cross_domain_integration",
    "mechanistic_coherence",
    "semantic_integrity",
    "plausibility",
    "testability",
    "specificity",
    "nontriviality",
    "overall",
    "confidence",
]


def merge_cross_jury_scores(
    candidate_records: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    required_judges: int,
) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for judgment in judgments:
        scores = judgment.get("scores", {})
        if not isinstance(scores, Mapping) or not bool(scores.get("valid_json", False)):
            continue
        by_candidate[str(judgment.get("candidate_id"))].append(judgment)

    merged: list[dict[str, Any]] = []
    for candidate in candidate_records:
        row = dict(candidate)
        jury_rows = by_candidate.get(str(candidate.get("candidate_id")), [])
        jury_scores: dict[str, float] = {}
        jury_score_std: dict[str, float] = {}
        for field_name in JURY_SCORE_FIELDS:
            values = [
                safe_float(judgment.get("scores", {}).get(field_name), float("nan"))
                for judgment in jury_rows
            ]
            values = [value for value in values if math.isfinite(value)]
            if values:
                jury_scores[field_name] = float(np.mean(values))
                jury_score_std[field_name] = float(np.std(values, ddof=0))
        judge_keys = sorted({str(judgment.get("judge_model_key")) for judgment in jury_rows})
        eligible = bool(candidate.get("reportable_eligible", False))
        row["jury_scores"] = jury_scores
        row["jury_score_std"] = jury_score_std
        row["jury_judges"] = judge_keys
        row["jury_n"] = len(judge_keys)
        row["required_judges"] = required_judges
        row["reportable"] = bool(
            eligible
            and len(judge_keys) >= required_judges
            and "overall" in jury_scores
        )
        row["evaluation_source"] = (
            "leave-source-out-cross-model-jury" if row["reportable"] else "not-reportable"
        )
        merged.append(row)
    return merged


def bootstrap_mean_ci(
    values: Sequence[float],
    resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return None, None
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(max(200, resamples), array.size))
    means = array[indices].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and math.isclose(
            indexed[end][1], indexed[index][1], rel_tol=1e-12, abs_tol=1e-12
        ):
            end += 1
        average = 0.5 * ((index + 1) + end)
        for position in range(index, end):
            ranks[indexed[position][0]] = average
        index = end
    return ranks


def wilcoxon_signed_rank(
    differences: Sequence[float],
) -> tuple[float | None, float | None, int]:
    diffs = [float(value) for value in differences if math.isfinite(value) and abs(value) > 1e-10]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0, 0
    ranks = average_ranks([abs(value) for value in diffs])
    w_plus = sum(rank for rank, value in zip(ranks, diffs) if value > 0)
    total = sum(ranks)
    observed = min(w_plus, total - w_plus)
    if n <= 20:
        extreme = 0
        permutations = 1 << n
        for mask in range(permutations):
            perm_plus = sum(ranks[i] for i in range(n) if mask & (1 << i))
            statistic = min(perm_plus, total - perm_plus)
            if statistic <= observed + 1e-12:
                extreme += 1
        p_value = extreme / permutations
    else:
        mean = total / 2.0
        variance = sum(rank * rank for rank in ranks) / 4.0
        z = (abs(w_plus - mean) - 0.5) / math.sqrt(max(variance, 1e-12))
        p_value = math.erfc(z / math.sqrt(2.0))
    return float(observed), float(clamp(p_value, 0.0, 1.0)), n


def holm_adjust(records: list[dict[str, Any]], p_field: str = "p_value") -> None:
    indexed = [
        (index, safe_float(record.get(p_field), float("nan")))
        for index, record in enumerate(records)
    ]
    indexed = [(index, p) for index, p in indexed if math.isfinite(p)]
    indexed.sort(key=lambda item: item[1])
    running = 0.0
    m = len(indexed)
    for rank, (index, p_value) in enumerate(indexed):
        adjusted = min(1.0, (m - rank) * p_value)
        running = max(running, adjusted)
        records[index]["p_holm"] = running


def aggregate_summary(
    evaluated_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in evaluated_records:
        if record.get("stage") not in {"search", "final"}:
            continue
        if record.get("method") == "autonomous-granular-controller" and record.get("stage") != "final":
            continue
        key = (
            str(record.get("task_id")),
            str(record.get("model_key")),
            str(record.get("method")),
            str(record.get("arm")),
        )
        groups[key].append(record)

    summary: list[dict[str, Any]] = []
    for (task_id, model_key, method, arm), rows in sorted(groups.items()):
        reportable_rows = [row for row in rows if bool(row.get("reportable", False))]

        def nested_values(source: str, field_name: str, source_rows: Sequence[Mapping[str, Any]]) -> list[float]:
            values: list[float] = []
            for row in source_rows:
                container = row.get(source, {})
                if isinstance(container, Mapping):
                    value = safe_float(container.get(field_name), float("nan"))
                    if math.isfinite(value):
                        values.append(value)
            return values

        self_overall = nested_values("scores", "overall", rows)
        jury_metrics = {
            field_name: nested_values("jury_scores", field_name, reportable_rows)
            for field_name in JURY_SCORE_FIELDS
        }
        ci_seed = args.seed + int(
            hashlib.sha256(f"{task_id}|{model_key}|{method}|{arm}".encode()).hexdigest()[:8],
            16,
        ) % 1_000_000_000
        jury_overall_low, jury_overall_high = bootstrap_mean_ci(
            jury_metrics["overall"],
            args.bootstrap_resamples,
            ci_seed,
        )
        rewards = [safe_float(row.get("reward"), float("nan")) for row in rows]
        rewards = [value for value in rewards if math.isfinite(value)]
        lexical = [safe_float(row.get("lexical_novelty"), float("nan")) for row in rows]
        lexical = [value for value in lexical if math.isfinite(value)]
        summary.append(
            {
                "timestamp": utc_now(),
                "task_id": task_id,
                "model_key": model_key,
                "method": method,
                "arm": arm,
                "n_total": len(rows),
                "n_reportable": len(reportable_rows),
                "required_judges": max(
                    [safe_int(row.get("required_judges"), 0) for row in rows],
                    default=0,
                ),
                "inferential_ready": len(reportable_rows) >= args.minimum_reportable_n,
                "self_overall_mean_diagnostic": float(np.mean(self_overall))
                if self_overall
                else None,
                "self_overall_std_diagnostic": float(np.std(self_overall))
                if self_overall
                else None,
                "jury_overall_mean": float(np.mean(jury_metrics["overall"]))
                if jury_metrics["overall"]
                else None,
                "jury_overall_std": float(np.std(jury_metrics["overall"]))
                if jury_metrics["overall"]
                else None,
                "jury_overall_ci95_low": jury_overall_low,
                "jury_overall_ci95_high": jury_overall_high,
                "jury_novelty_mean": float(np.mean(jury_metrics["novelty"]))
                if jury_metrics["novelty"]
                else None,
                "jury_cross_domain_integration_mean": float(
                    np.mean(jury_metrics["cross_domain_integration"])
                )
                if jury_metrics["cross_domain_integration"]
                else None,
                "jury_mechanistic_coherence_mean": float(
                    np.mean(jury_metrics["mechanistic_coherence"])
                )
                if jury_metrics["mechanistic_coherence"]
                else None,
                "jury_semantic_integrity_mean": float(
                    np.mean(jury_metrics["semantic_integrity"])
                )
                if jury_metrics["semantic_integrity"]
                else None,
                "jury_plausibility_mean": float(np.mean(jury_metrics["plausibility"]))
                if jury_metrics["plausibility"]
                else None,
                "jury_testability_mean": float(np.mean(jury_metrics["testability"]))
                if jury_metrics["testability"]
                else None,
                "reward_mean_online_diagnostic": float(np.mean(rewards)) if rewards else None,
                "lexical_novelty_mean_diagnostic": float(np.mean(lexical)) if lexical else None,
                "generation_seconds_mean": float(
                    np.mean(
                        [
                            safe_float(row.get("generation_seconds"), 0.0)
                            for row in rows
                            if row.get("generation_seconds") is not None
                        ]
                    )
                )
                if any(row.get("generation_seconds") is not None for row in rows)
                else None,
                "generated_tokens_mean": float(
                    np.mean(
                        [
                            safe_float(row.get("generated_tokens"), 0.0)
                            for row in rows
                            if row.get("generated_tokens") is not None
                        ]
                    )
                )
                if any(row.get("generated_tokens") is not None for row in rows)
                else None,
            }
        )
    return summary


def paired_cti_tests(
    evaluated_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluated_records:
        if row.get("method") != "contrastive-trajectory-inhibition":
            continue
        if row.get("stage") != "search" or not bool(row.get("reportable", False)):
            continue
        groups[(str(row.get("task_id")), str(row.get("model_key")))].append(row)

    comparisons = [
        ("learned-inhibition", "baseline", "learned-minus-baseline"),
        ("random-subspace-control", "baseline", "random-minus-baseline"),
        ("learned-inhibition", "random-subspace-control", "learned-minus-random"),
    ]
    metrics = [
        "overall",
        "novelty",
        "cross_domain_integration",
        "mechanistic_coherence",
        "semantic_integrity",
        "plausibility",
        "testability",
    ]
    results: list[dict[str, Any]] = []
    for (task_id, model_key), rows in sorted(groups.items()):
        by_arm_pair: dict[tuple[str, int], Mapping[str, Any]] = {}
        for row in rows:
            by_arm_pair[(str(row.get("arm")), safe_int(row.get("pair_index"), -1))] = row
        group_records: list[dict[str, Any]] = []
        for arm_a, arm_b, comparison_name in comparisons:
            pair_ids = sorted(
                pair_id
                for pair_id in {pair for arm, pair in by_arm_pair if arm == arm_a}
                if (arm_b, pair_id) in by_arm_pair
            )
            for metric in metrics:
                values_a: list[float] = []
                values_b: list[float] = []
                used_pairs: list[int] = []
                for pair_id in pair_ids:
                    value_a = safe_float(
                        by_arm_pair[(arm_a, pair_id)].get("jury_scores", {}).get(metric),
                        float("nan"),
                    )
                    value_b = safe_float(
                        by_arm_pair[(arm_b, pair_id)].get("jury_scores", {}).get(metric),
                        float("nan"),
                    )
                    if math.isfinite(value_a) and math.isfinite(value_b):
                        values_a.append(value_a)
                        values_b.append(value_b)
                        used_pairs.append(pair_id)
                differences = [a - b for a, b in zip(values_a, values_b)]
                statistic, p_value, nonzero_n = wilcoxon_signed_rank(differences)
                ci_seed = args.seed + int(
                    hashlib.sha256(
                        f"{task_id}|{model_key}|{comparison_name}|{metric}".encode()
                    ).hexdigest()[:8],
                    16,
                ) % 1_000_000_000
                ci_low, ci_high = bootstrap_mean_ci(
                    differences,
                    args.bootstrap_resamples,
                    ci_seed,
                )
                std_diff = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
                effect_dz = (
                    float(np.mean(differences)) / std_diff if std_diff > 1e-12 else None
                )
                group_records.append(
                    {
                        "timestamp": utc_now(),
                        "task_id": task_id,
                        "model_key": model_key,
                        "method": "contrastive-trajectory-inhibition",
                        "comparison": comparison_name,
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "metric": metric,
                        "multiplicity_family": (
                            "primary-overall" if metric == "overall" else "secondary-components"
                        ),
                        "n_pairs": len(differences),
                        "n_nonzero": nonzero_n,
                        "pair_ids": used_pairs,
                        "arm_a_mean": float(np.mean(values_a)) if values_a else None,
                        "arm_b_mean": float(np.mean(values_b)) if values_b else None,
                        "mean_paired_difference": float(np.mean(differences))
                        if differences
                        else None,
                        "median_paired_difference": float(np.median(differences))
                        if differences
                        else None,
                        "paired_difference_ci95_low": ci_low,
                        "paired_difference_ci95_high": ci_high,
                        "paired_effect_dz": effect_dz,
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "p_holm": None,
                        "inferential_ready": len(differences) >= args.minimum_reportable_n,
                    }
                )
        for family in ("primary-overall", "secondary-components"):
            family_records = [
                record for record in group_records if record.get("multiplicity_family") == family
            ]
            holm_adjust(family_records)
        results.extend(group_records)
    return results


def save_figure(plt: Any, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_all(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = read_jsonl(output_dir / "candidates.jsonl")
    evaluated = read_jsonl(output_dir / "evaluated_candidates.jsonl")
    actions = read_jsonl(output_dir / "controller_actions.jsonl")
    layers = read_jsonl(output_dir / "layer_geometry.jsonl")
    traces = read_jsonl(output_dir / "intervention_traces.jsonl")
    judgments = read_jsonl(output_dir / "cross_judgments.jsonl")
    summary = read_jsonl(output_dir / "summary.jsonl")
    paired = read_jsonl(output_dir / "paired_tests.jsonl")
    magnitude_audits = read_jsonl(output_dir / "magnitude_audits.jsonl")
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # 1. Leave-source-out jury quality with bootstrap confidence intervals.
    rows = [row for row in summary if row.get("jury_overall_mean") is not None]
    if rows:
        labels = [f"{row['model_key']}\n{row['method']}\n{row['arm']}" for row in rows]
        means = np.asarray([safe_float(row.get("jury_overall_mean")) for row in rows])
        lower = np.asarray(
            [
                means[index] - safe_float(row.get("jury_overall_ci95_low"), means[index])
                for index, row in enumerate(rows)
            ]
        )
        upper = np.asarray(
            [
                safe_float(row.get("jury_overall_ci95_high"), means[index]) - means[index]
                for index, row in enumerate(rows)
            ]
        )
        plt.figure(figsize=(max(11, len(rows) * 0.72), 6))
        plt.bar(range(len(rows)), means, yerr=np.vstack([lower, upper]), capsize=4)
        plt.xticks(range(len(rows)), labels, rotation=75, ha="right")
        plt.ylabel("Leave-source-out jury overall score (0-10)")
        plt.title("Reportable Candidate Quality with 95% Bootstrap Intervals")
        save_figure(plt, figures / "01_jury_quality_with_ci")

    reportable = [row for row in evaluated if bool(row.get("reportable", False))]

    # 2. Closed-loop bias diagnostic: source self-score versus held-out jury score.
    if reportable:
        plt.figure(figsize=(8, 7))
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in reportable:
            groups[str(row.get("model_key"))].append(row)
        for label, group in groups.items():
            x = [safe_float(row.get("scores", {}).get("overall")) for row in group]
            y = [safe_float(row.get("jury_scores", {}).get("overall")) for row in group]
            plt.scatter(x, y, alpha=0.75, label=label)
        plt.plot([0, 10], [0, 10], linestyle="--")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.xlabel("Source-model self-score (diagnostic only)")
        plt.ylabel("Leave-source-out jury score")
        plt.title("Self-Evaluation versus Held-Out Evaluation")
        plt.legend()
        save_figure(plt, figures / "02_self_vs_jury")

    # 3. Jury novelty-plausibility tradeoff.
    if reportable:
        plt.figure(figsize=(9, 7))
        groups = defaultdict(list)
        for row in reportable:
            groups[f"{row.get('model_key')} | {row.get('method')} | {row.get('arm')}"] .append(row)
        for label, group in groups.items():
            x = [safe_float(row.get("jury_scores", {}).get("novelty")) for row in group]
            y = [safe_float(row.get("jury_scores", {}).get("plausibility")) for row in group]
            plt.scatter(x, y, label=label, alpha=0.72)
        plt.xlabel("Jury novelty score")
        plt.ylabel("Jury plausibility score")
        plt.title("Held-Out Novelty-Plausibility Tradeoff")
        plt.legend(fontsize=7)
        save_figure(plt, figures / "03_jury_novelty_plausibility")

    # 4. Jury cross-domain integration versus testability.
    if reportable:
        plt.figure(figsize=(9, 7))
        groups = defaultdict(list)
        for row in reportable:
            groups[f"{row.get('model_key')} | {row.get('method')} | {row.get('arm')}"] .append(row)
        for label, group in groups.items():
            x = [
                safe_float(row.get("jury_scores", {}).get("cross_domain_integration"))
                for row in group
            ]
            y = [safe_float(row.get("jury_scores", {}).get("testability")) for row in group]
            plt.scatter(x, y, label=label, alpha=0.72)
        plt.xlabel("Jury cross-domain integration")
        plt.ylabel("Jury testability")
        plt.title("Held-Out Composition Strength versus Testability")
        plt.legend(fontsize=7)
        save_figure(plt, figures / "04_jury_composition_testability")

    # 5. Factorial instruction main effect and temperature main effect by layer.
    if layers:
        groups = defaultdict(list)
        for row in layers:
            groups[(row.get("task_id"), row.get("model_key"))].append(row)
        plt.figure(figsize=(11, 7))
        for (task_id, model_key), group in groups.items():
            group = sorted(group, key=lambda row: safe_int(row.get("layer_index")))
            x = [safe_float(row.get("layer_fraction")) for row in group]
            instruction = [safe_float(row.get("smoothed_signal")) for row in group]
            temperature = [safe_float(row.get("temperature_signal")) for row in group]
            plt.plot(x, instruction, label=f"{model_key} | {task_id} | instruction")
            plt.plot(x, temperature, linestyle="--", label=f"{model_key} | {task_id} | temperature")
        plt.xlabel("Normalized layer depth")
        plt.ylabel("Standardized factorial effect signal")
        plt.title("Instruction Main Effect versus Temperature Main Effect")
        plt.legend(fontsize=7)
        save_figure(plt, figures / "05_factorial_instruction_temperature")

    # 6. Instruction-by-temperature interaction.
    if layers:
        plt.figure(figsize=(10, 6))
        groups = defaultdict(list)
        for row in layers:
            groups[(row.get("task_id"), row.get("model_key"))].append(row)
        for (task_id, model_key), group in groups.items():
            group = sorted(group, key=lambda row: safe_int(row.get("layer_index")))
            plt.plot(
                [safe_float(row.get("layer_fraction")) for row in group],
                [safe_float(row.get("interaction_signal")) for row in group],
                label=f"{model_key} | {task_id}",
            )
        plt.xlabel("Normalized layer depth")
        plt.ylabel("Instruction × temperature interaction signal")
        plt.title("Factorial Interaction Diagnostic")
        plt.legend(fontsize=8)
        save_figure(plt, figures / "06_factorial_interaction")

    # 7. Geometric overlap between instruction and temperature subspaces.
    if layers:
        plt.figure(figsize=(10, 6))
        groups = defaultdict(list)
        for row in layers:
            groups[(row.get("task_id"), row.get("model_key"))].append(row)
        for (task_id, model_key), group in groups.items():
            group = sorted(group, key=lambda row: safe_int(row.get("layer_index")))
            plt.plot(
                [safe_float(row.get("layer_fraction")) for row in group],
                [safe_float(row.get("instruction_temperature_mean_cosine")) for row in group],
                label=f"{model_key} | {task_id}",
            )
        plt.xlabel("Normalized layer depth")
        plt.ylabel("Absolute cosine of factorial mean-effect vectors")
        plt.ylim(0, 1.05)
        plt.title("Instruction-Temperature Subspace Overlap")
        plt.legend(fontsize=8)
        save_figure(plt, figures / "07_instruction_temperature_overlap")

    # 8. Selected layer bands.
    selected_layers = [row for row in layers if row.get("selected")]
    if selected_layers:
        plt.figure(figsize=(10, 6))
        model_keys = sorted({str(row.get("model_key")) for row in selected_layers})
        for y_index, model_key in enumerate(model_keys):
            group = [row for row in selected_layers if row.get("model_key") == model_key]
            x = [safe_float(row.get("layer_fraction")) for row in group]
            y = [y_index] * len(group)
            sizes = [30 + 100 * safe_float(row.get("layer_weight")) for row in group]
            plt.scatter(x, y, s=sizes, label=model_key)
        plt.yticks(range(len(model_keys)), model_keys)
        plt.xlabel("Normalized layer depth")
        plt.title("Intervention Bands Selected from the Instruction Main Effect")
        save_figure(plt, figures / "08_selected_layer_bands")

    # 9. Cumulative singular energy of the instruction main-effect design.
    if layers:
        plt.figure(figsize=(9, 6))
        groups = defaultdict(list)
        for row in layers:
            if row.get("selected"):
                groups[str(row.get("model_key"))].append(row)
        for model_key, group in groups.items():
            curves = [row.get("cumulative_energy", []) for row in group]
            max_len = max((len(curve) for curve in curves), default=0)
            if not max_len:
                continue
            matrix = np.full((len(curves), max_len), np.nan)
            for index, curve in enumerate(curves):
                matrix[index, : len(curve)] = np.asarray(curve, dtype=float)
            plt.plot(
                range(1, max_len + 1),
                np.nanmean(matrix, axis=0),
                marker="o",
                label=model_key,
            )
        plt.xlabel("Rank")
        plt.ylabel("Mean cumulative instruction-effect energy")
        plt.ylim(0, 1.05)
        plt.title("Low-Rank Saturation of the Counterbalanced Instruction Effect")
        plt.legend()
        save_figure(plt, figures / "09_instruction_singular_energy")

    # 10. Default-side drift over generation.
    if traces:
        plt.figure(figsize=(10, 6))
        groups = defaultdict(list)
        for row in traces:
            groups[(row.get("model_key"), row.get("method"), row.get("arm"))].append(row)
        for key, group in groups.items():
            by_step = defaultdict(list)
            for row in group:
                by_step[safe_int(row.get("step"))].append(safe_float(row.get("drift")))
            steps = sorted(by_step)
            plt.plot(
                steps,
                [float(np.mean(by_step[step])) for step in steps],
                label=" | ".join(str(item) for item in key),
            )
        plt.xlabel("Generation forward step")
        plt.ylabel("Mean normalized conventional-side drift")
        plt.title("Trajectory Drift During Generation")
        plt.legend(fontsize=7)
        save_figure(plt, figures / "10_drift_over_generation")

    # 11. Realized post-RMS intervention magnitude.
    if traces:
        plt.figure(figsize=(10, 6))
        groups = defaultdict(list)
        for row in traces:
            groups[(row.get("model_key"), row.get("method"), row.get("arm"))].append(row)
        for key, group in groups.items():
            by_step = defaultdict(list)
            for row in group:
                by_step[safe_int(row.get("step"))].append(safe_float(row.get("delta_ratio")))
            steps = sorted(by_step)
            plt.plot(
                steps,
                [float(np.mean(by_step[step])) for step in steps],
                label=" | ".join(str(item) for item in key),
            )
        plt.xlabel("Generation forward step")
        plt.ylabel("Realized perturbation / hidden-state norm")
        plt.title("Post-RMS Activation Intervention Magnitude")
        plt.legend(fontsize=7)
        save_figure(plt, figures / "11_intervention_magnitude")

    # 12. Learned versus random realized magnitude audit.
    if magnitude_audits:
        plt.figure(figsize=(8, 7))
        groups = defaultdict(list)
        for row in magnitude_audits:
            groups[str(row.get("model_key"))].append(row)
        for model_key, group in groups.items():
            x = [safe_float(row.get("learned_mean_delta_ratio")) for row in group]
            y = [safe_float(row.get("random_mean_delta_ratio_on_common")) for row in group]
            plt.scatter(x, y, label=model_key, alpha=0.8)
        maximum = max(
            [
                safe_float(row.get("learned_mean_delta_ratio"))
                for row in magnitude_audits
            ]
            + [
                safe_float(row.get("random_mean_delta_ratio_on_common"))
                for row in magnitude_audits
            ]
            + [0.01]
        )
        plt.plot([0, maximum], [0, maximum], linestyle="--")
        plt.xlabel("Learned arm mean realized delta ratio")
        plt.ylabel("Random arm mean realized delta ratio")
        plt.title("Paired Realized-Magnitude Matching")
        plt.legend()
        save_figure(plt, figures / "12_magnitude_match_scatter")

    # 13. Magnitude matching error and schedule coverage.
    if magnitude_audits:
        labels = [
            f"{row.get('model_key')}\n{row.get('task_id')}\n{row.get('pair_index')}"
            for row in magnitude_audits
        ]
        errors = [safe_float(row.get("max_relative_error")) for row in magnitude_audits]
        coverage = [safe_float(row.get("coverage")) for row in magnitude_audits]
        plt.figure(figsize=(max(10, len(labels) * 0.55), 6))
        plt.bar(range(len(labels)), errors)
        plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
        plt.ylabel("Maximum relative magnitude error")
        plt.title("Magnitude-Match Error by Paired Seed")
        save_figure(plt, figures / "13_magnitude_match_error")

        plt.figure(figsize=(max(10, len(labels) * 0.55), 6))
        plt.bar(range(len(labels)), coverage)
        plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("Learned schedule coverage")
        plt.title("Magnitude-Schedule Coverage by Paired Seed")
        save_figure(plt, figures / "14_magnitude_schedule_coverage")

    # 15. Paired jury-score effects with bootstrap intervals.
    overall_paired = [row for row in paired if row.get("metric") == "overall"]
    if overall_paired:
        labels = [f"{row.get('model_key')}\n{row.get('comparison')}" for row in overall_paired]
        means = np.asarray(
            [safe_float(row.get("mean_paired_difference")) for row in overall_paired]
        )
        lower = np.asarray(
            [
                means[index]
                - safe_float(row.get("paired_difference_ci95_low"), means[index])
                for index, row in enumerate(overall_paired)
            ]
        )
        upper = np.asarray(
            [
                safe_float(row.get("paired_difference_ci95_high"), means[index])
                - means[index]
                for index, row in enumerate(overall_paired)
            ]
        )
        plt.figure(figsize=(max(10, len(labels) * 0.75), 6))
        plt.errorbar(
            range(len(labels)),
            means,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=5,
        )
        plt.axhline(0, linestyle="--")
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
        plt.ylabel("Mean paired jury-score difference")
        plt.title("Paired CTI Effects with 95% Bootstrap Intervals")
        save_figure(plt, figures / "15_paired_cti_effects")

        plt.figure(figsize=(max(10, len(labels) * 0.75), 6))
        p_values = [max(1e-12, safe_float(row.get("p_holm"), 1.0)) for row in overall_paired]
        plt.bar(range(len(labels)), [-math.log10(value) for value in p_values])
        plt.axhline(-math.log10(0.05), linestyle="--")
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
        plt.ylabel("-log10 Holm-adjusted p")
        plt.title("Exact Paired Wilcoxon Evidence")
        save_figure(plt, figures / "16_paired_wilcoxon")

    # 17. AGC online reward by round. This is controller state, not an outcome measure.
    agc_search = [
        row
        for row in evaluated
        if row.get("method") == "autonomous-granular-controller"
        and row.get("stage") == "search"
        and row.get("round") is not None
    ]
    if agc_search:
        plt.figure(figsize=(10, 6))
        groups = defaultdict(list)
        for row in agc_search:
            groups[str(row.get("model_key"))].append(row)
        for model_key, group in groups.items():
            by_round = defaultdict(list)
            for row in group:
                by_round[safe_int(row.get("round"))].append(safe_float(row.get("reward")))
            rounds = sorted(by_round)
            plt.plot(
                rounds,
                [float(np.mean(by_round[index])) for index in rounds],
                marker="o",
                label=model_key,
            )
        plt.xlabel("Controller round")
        plt.ylabel("Online source-model reward proxy")
        plt.title("Autonomous Controller Feedback Signal (Diagnostic Only)")
        plt.legend()
        save_figure(plt, figures / "17_controller_reward")

    # 18-21. Controller settings over time.
    if actions:
        settings_to_plot = [
            ("inhibition_strength", "Inhibition strength", "18_controller_inhibition"),
            ("tube_strength", "Tube strength", "19_controller_tube"),
            ("temperature", "Sampling temperature", "20_controller_temperature"),
            ("horizon_tokens", "Generation horizon", "21_controller_horizon"),
        ]
        for field_name, ylabel, filename in settings_to_plot:
            plt.figure(figsize=(10, 6))
            groups = defaultdict(list)
            for row in actions:
                groups[str(row.get("model_key"))].append(row)
            for model_key, group in groups.items():
                by_round = defaultdict(list)
                for row in group:
                    by_round[safe_int(row.get("round"))].append(
                        safe_float(row.get("action", {}).get(field_name))
                    )
                rounds = sorted(by_round)
                plt.plot(
                    rounds,
                    [float(np.mean(by_round[index])) for index in rounds],
                    marker="o",
                    label=model_key,
                )
            plt.xlabel("Controller round")
            plt.ylabel(ylabel)
            plt.title(ylabel + " Chosen by the Autonomous Controller")
            plt.legend()
            save_figure(plt, figures / filename)

    # 22. Architecture-by-primary-method held-out jury heatmap.
    primary_rows = [
        row
        for row in summary
        if row.get("jury_overall_mean") is not None
        and (
            row.get("arm") == "learned-inhibition"
            or row.get("arm") == "controller-final"
        )
    ]
    if primary_rows:
        model_keys = sorted({str(row.get("model_key")) for row in primary_rows})
        methods = sorted({str(row.get("method")) for row in primary_rows})
        matrix = np.full((len(methods), len(model_keys)), np.nan)
        for method_index, method in enumerate(methods):
            for model_index, model_key in enumerate(model_keys):
                values = [
                    safe_float(row.get("jury_overall_mean"), float("nan"))
                    for row in primary_rows
                    if row.get("method") == method and row.get("model_key") == model_key
                ]
                values = [value for value in values if math.isfinite(value)]
                if values:
                    matrix[method_index, model_index] = float(np.mean(values))
        plt.figure(figsize=(8, 5))
        image = plt.imshow(matrix, aspect="auto")
        plt.colorbar(image, label="Mean leave-source-out jury score")
        plt.xticks(range(len(model_keys)), model_keys, rotation=30, ha="right")
        plt.yticks(range(len(methods)), methods)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if math.isfinite(value):
                    plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
        plt.title("Architecture × Primary Search Method")
        save_figure(plt, figures / "22_architecture_method_heatmap")

    # 23. Output length versus held-out quality.
    length_rows = [row for row in reportable if row.get("generated_tokens") is not None]
    if length_rows:
        plt.figure(figsize=(9, 7))
        groups = defaultdict(list)
        for row in length_rows:
            groups[str(row.get("model_key"))].append(row)
        for model_key, group in groups.items():
            plt.scatter(
                [safe_float(row.get("generated_tokens")) for row in group],
                [safe_float(row.get("jury_scores", {}).get("overall")) for row in group],
                label=model_key,
                alpha=0.75,
            )
        plt.xlabel("Generated tokens")
        plt.ylabel("Leave-source-out jury score")
        plt.title("Output Length versus Held-Out Quality")
        plt.legend()
        save_figure(plt, figures / "23_output_length_quality")

    # 24. Generation time.
    timing_rows = [row for row in summary if row.get("generation_seconds_mean") is not None]
    if timing_rows:
        labels = [f"{row['model_key']}\n{row['method']}\n{row['arm']}" for row in timing_rows]
        values = [safe_float(row.get("generation_seconds_mean")) for row in timing_rows]
        plt.figure(figsize=(max(10, len(labels) * 0.65), 6))
        plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
        plt.ylabel("Mean generation seconds")
        plt.title("Generation Cost by Model, Method, and Arm")
        save_figure(plt, figures / "24_generation_time")

    # 25. Cross-jury source/judge matrix.
    if judgments:
        source_models = sorted({str(row.get("source_model_key")) for row in judgments})
        judge_models = sorted({str(row.get("judge_model_key")) for row in judgments})
        matrix = np.full((len(source_models), len(judge_models)), np.nan)
        for source_index, source_model in enumerate(source_models):
            for judge_index, judge_model in enumerate(judge_models):
                values = [
                    safe_float(row.get("scores", {}).get("overall"), float("nan"))
                    for row in judgments
                    if row.get("source_model_key") == source_model
                    and row.get("judge_model_key") == judge_model
                    and bool(row.get("scores", {}).get("valid_json", False))
                ]
                values = [value for value in values if math.isfinite(value)]
                if values:
                    matrix[source_index, judge_index] = float(np.mean(values))
        plt.figure(figsize=(8, 6))
        image = plt.imshow(matrix, aspect="auto")
        plt.colorbar(image, label="Mean jury overall score")
        plt.xticks(range(len(judge_models)), judge_models, rotation=30, ha="right")
        plt.yticks(range(len(source_models)), source_models)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if math.isfinite(value):
                    plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
        plt.xlabel("Judge model")
        plt.ylabel("Source model")
        plt.title("Blinded Leave-Source-Out Jury Matrix")
        save_figure(plt, figures / "25_cross_jury_matrix")

    # 26. Jury disagreement per candidate.
    disagreement_rows = [row for row in reportable if row.get("jury_score_std", {}).get("overall") is not None]
    if disagreement_rows:
        labels = [
            f"{row.get('model_key')}\n{row.get('method')}\n{row.get('arm')}\n{row.get('pair_index')}"
            for row in disagreement_rows
        ]
        values = [
            safe_float(row.get("jury_score_std", {}).get("overall"))
            for row in disagreement_rows
        ]
        plt.figure(figsize=(max(10, len(labels) * 0.45), 6))
        plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
        plt.ylabel("Across-judge standard deviation")
        plt.title("Held-Out Jury Disagreement")
        save_figure(plt, figures / "26_jury_disagreement")

    # 27. Calibration design check: output length across the four cells.
    calibration_rows = [row for row in candidates if row.get("stage") == "calibration"]
    if calibration_rows:
        cells = [
            "conventional-low",
            "conventional-high",
            "composition-low",
            "composition-high",
        ]
        means = []
        for cell in cells:
            values = [
                safe_float(row.get("generated_tokens"))
                for row in calibration_rows
                if row.get("arm") == cell
            ]
            means.append(float(np.mean(values)) if values else float("nan"))
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(cells)), means)
        plt.xticks(range(len(cells)), cells, rotation=30, ha="right")
        plt.ylabel("Mean generated tokens")
        plt.title("2×2 Calibration Cell Output-Length Check")
        save_figure(plt, figures / "27_calibration_cell_lengths")


def finalize_outputs(store: RecordStore, args: argparse.Namespace) -> None:
    candidate_records = read_jsonl(store.paths["candidates"])
    judgments = read_jsonl(store.paths["cross_judgments"])
    evaluated_records = merge_cross_jury_scores(
        candidate_records,
        judgments,
        required_judges=args.minimum_cross_judges,
    )
    for record in evaluated_records:
        store.add("evaluated_candidates", record)

    summary_records = aggregate_summary(evaluated_records, args)
    for record in summary_records:
        store.add("summary", record)

    paired_records = paired_cti_tests(evaluated_records, args)
    for record in paired_records:
        store.add("paired_tests", record)

    store.finalize_csv()
    if not args.skip_plots:
        try:
            plot_all(store.output_dir)
        except Exception as exc:
            store.add(
                "errors",
                {
                    "timestamp": utc_now(),
                    "stage": "plotting",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            LOGGER.exception("Plot generation failed")
            store.finalize_csv()


# -----------------------------------------------------------------------------
# Task loading and command-line interface
# -----------------------------------------------------------------------------


def load_tasks(args: argparse.Namespace) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    if args.task:
        for index, prompt in enumerate(args.task):
            tasks.append({"id": f"task-{index + 1}-{short_hash(prompt)}", "prompt": prompt})
    if args.task_file:
        path = Path(args.task_file)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    parsed = {"prompt": line}
                if isinstance(parsed, str):
                    parsed = {"prompt": parsed}
                if not isinstance(parsed, Mapping) or not parsed.get("prompt"):
                    raise ValueError(f"Invalid task-file line {index + 1}")
                prompt = str(parsed["prompt"])
                task_id = str(parsed.get("id") or f"task-{index + 1}-{short_hash(prompt)}")
                tasks.append({"id": task_id, "prompt": prompt})
    if not tasks:
        tasks.append({"id": f"default-{short_hash(DEFAULT_TASK)}", "prompt": DEFAULT_TASK})
    return tasks


def apply_preset(args: argparse.Namespace) -> None:
    presets = {
        "smoke": {
            "calibration_samples": 2,
            "candidates_per_arm": 2,
            "controller_replicates": 1,
            "controller_rounds": 2,
            "controller_branches": 1,
            "controller_max_branches": 2,
            "max_new_tokens": 128,
            "controller_horizon_tokens": 96,
        },
        "standard": {
            "calibration_samples": 4,
            "candidates_per_arm": 8,
            "controller_replicates": 8,
            "controller_rounds": 4,
            "controller_branches": 2,
            "controller_max_branches": 3,
            "max_new_tokens": 320,
            "controller_horizon_tokens": 220,
        },
        "serious": {
            "calibration_samples": 8,
            "candidates_per_arm": 16,
            "controller_replicates": 16,
            "controller_rounds": 8,
            "controller_branches": 4,
            "controller_max_branches": 6,
            "max_new_tokens": 512,
            "controller_horizon_tokens": 360,
        },
    }
    values = presets[args.preset]
    for name, value in values.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare trajectory inhibition and autonomous activation control across local LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", action="append", help="Discovery task. Repeat for multiple tasks.")
    parser.add_argument("--task-file", help="JSONL or plain-text task file.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["cti", "agc"],
        default=["cti", "agc"],
    )
    parser.add_argument("--preset", choices=["smoke", "standard", "serious"], default="standard")
    parser.add_argument("--output-dir", help="Output directory. Defaults to a timestamped folder.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hf-token", help="Hugging Face token; environment variables are also supported.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-cpu-large", action="store_true")
    parser.add_argument("--quantization", choices=["auto", "none", "4bit"], default="auto")
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help="auto, eager, flash_attention_2, flex_attention, etc.",
    )

    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--candidates-per-arm", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--judge-max-new-tokens", type=int, default=300)
    parser.add_argument("--controller-max-new-tokens", type=int, default=500)
    parser.add_argument("--state-tokens", type=int, default=8)
    parser.add_argument("--trajectory-rank", type=int, default=4)
    parser.add_argument("--positive-rank", type=int, default=8)
    parser.add_argument("--threshold-quantile", type=float, default=0.90)
    parser.add_argument("--band-fraction", type=float, default=0.25)
    parser.add_argument("--exclude-final-layers", type=int, default=1)
    parser.add_argument("--fixed-band", help="Normalized fixed layer band, e.g. 0.45:0.80")

    parser.add_argument(
        "--calibration-low-temperature",
        type=float,
        default=0.30,
        help="Low-temperature cell used for both calibration instructions.",
    )
    parser.add_argument(
        "--calibration-high-temperature",
        type=float,
        default=1.00,
        help="High-temperature cell used for both calibration instructions.",
    )
    parser.add_argument("--search-temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--inhibition-strength", type=float, default=0.55)
    parser.add_argument("--tube-strength", type=float, default=0.08)
    parser.add_argument("--max-delta-ratio", type=float, default=0.12)
    parser.add_argument("--magnitude-match-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-magnitude-schedule-coverage", type=float, default=0.90)
    parser.add_argument(
        "--strict-magnitude-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude random-control pairs that fail the realized magnitude audit.",
    )
    parser.add_argument("--gate-temperature", type=float, default=0.75)
    parser.add_argument("--intervention-token-start", type=int, default=0)
    parser.add_argument("--intervention-token-end", type=int)
    parser.add_argument("--disable-adaptive-gate", action="store_true")
    parser.add_argument("--disable-rms-preservation", action="store_true")

    parser.add_argument("--controller-replicates", type=int, default=None)
    parser.add_argument("--controller-rounds", type=int, default=None)
    parser.add_argument("--controller-branches", type=int, default=None)
    parser.add_argument("--controller-max-branches", type=int, default=None)
    parser.add_argument("--controller-horizon-tokens", type=int, default=None)
    parser.add_argument("--controller-beam-width", type=int, default=4)

    parser.add_argument(
        "--cross-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run blinded leave-source-out jury evaluation for every reportable candidate.",
    )
    parser.add_argument(
        "--include-self-in-jury",
        action="store_true",
        help="Also count the source model as a jury member; off by default.",
    )
    parser.add_argument(
        "--minimum-cross-judges",
        type=int,
        default=None,
        help="Minimum valid leave-source-out judges required per reportable candidate.",
    )
    parser.add_argument("--minimum-reportable-n", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument(
        "--exploratory-only",
        action="store_true",
        help="Permit a run without cross-judging; summaries remain explicitly non-reportable.",
    )
    parser.add_argument("--store-raw-generations", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def build_metadata(args: argparse.Namespace, tasks: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    return {
        "created_at": utc_now(),
        "script": Path(__file__).name,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "resolved_device": str(resolve_torch_device()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
        "mps_available": mps_available(),
        "accelerator_available": accelerator_available(),
        "arguments": vars(args),
        "tasks": list(tasks),
        "model_specs": {key: dataclasses.asdict(value) for key, value in MODEL_SPECS.items()},
        "method_notes": {
            "cti": (
                "One-sided inhibition fit to the instruction main effect of a balanced 2x2 "
                "instruction-by-temperature calibration, with paired baseline and realized-"
                "magnitude-matched random controls."
            ),
            "agc": (
                "Same-model feedback controller over layer/time/intervention/sampling controls "
                "with halt and answer actions; its online self-score is controller state, not "
                "the reported outcome measure."
            ),
        },
        "reporting_policy": (
            "Reportable scores require blinded leave-source-out cross-model judging. All "
            "calibration and online source-model scores are diagnostics only."
        ),
        "design_controls": {
            "calibration": "seed-paired balanced 2x2 instruction-by-temperature factorial",
            "geometry_estimand": "instruction main effect averaged across temperature cells",
            "positive_tube": "composition states with temperature-cell means residualized",
            "cti_pairing": "common random seed across baseline, learned, and random arms",
            "random_control": "rank-matched random directions with learned post-RMS delta schedule",
            "primary_evaluation": "leave-source-out jury mean",
            "primary_inference": "paired exact Wilcoxon plus paired bootstrap CI",
        },
        "epistemic_warning": (
            "Self-judged novelty and quality are proxies. Candidate AI-research outputs require "
            "independent literature review, AI/ML domain expertise, and empirical or formal "
            "validation."
        ),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    apply_preset(args)
    if args.calibration_low_temperature <= 0 or args.calibration_high_temperature <= 0:
        parser.error("Calibration temperatures must be positive.")
    if not args.cross_judge and not args.exploratory_only:
        parser.error(
            "--no-cross-judge is allowed only with --exploratory-only; otherwise the run "
            "would produce an apparently reportable closed-loop evaluation."
        )
    available_judges = len(args.models) if args.include_self_in_jury else len(args.models) - 1
    if args.cross_judge and available_judges < 1:
        parser.error(
            "Cross-judging requires at least two selected models when the source model is excluded."
        )
    if args.minimum_cross_judges is None:
        args.minimum_cross_judges = max(0, available_judges)
    if args.minimum_cross_judges < 0 or args.minimum_cross_judges > max(0, available_judges):
        parser.error("--minimum-cross-judges exceeds the available jury size.")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    tasks = load_tasks(args)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(f"latent-composition-results-{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(args, tasks)
    (output_dir / "metadata.json").write_text(
        json.dumps(jsonable(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        print(json.dumps(jsonable(metadata), ensure_ascii=False, indent=2))
        return 0

    store = RecordStore(output_dir)
    all_method_candidates: list[dict[str, Any]] = []

    # Model-outer order deliberately loads each large model only once, while still
    # evaluating every requested method × model combination.
    for model_key in args.models:
        runtime: ModelRuntime | None = None
        spec = MODEL_SPECS[model_key]
        try:
            LOGGER.info("Loading %s (%s)", model_key, spec.model_id)
            runtime = load_model_runtime(spec, args)
            model_meta = {
                "timestamp": utc_now(),
                "model_key": model_key,
                "model_id": spec.model_id,
                "description": spec.description,
                "layer_path": runtime.layer_path,
                "input_device": str(runtime.input_device),
                "quantization": runtime.quantization_label,
                "config": runtime.config_summary,
            }
            store.add("models", model_meta)
            LOGGER.info(
                "Loaded %s: %d layers via %s on %s",
                model_key,
                len(runtime.layers),
                runtime.layer_path,
                runtime.input_device,
            )

            for task in tasks:
                task_id = task["id"]
                prompt = task["prompt"]
                LOGGER.info("Starting %s | %s", model_key, task_id)
                calibration = calibrate_model_task(runtime, task_id, prompt, args, store)
                learned_geometry, random_geometry = fit_trajectory_geometry(
                    runtime,
                    calibration,
                    args,
                    task_id,
                    store,
                )
                if "cti" in args.methods:
                    all_method_candidates.extend(
                        run_cti_method(
                            runtime,
                            task_id,
                            prompt,
                            calibration,
                            learned_geometry,
                            random_geometry,
                            args,
                            store,
                        )
                    )
                if "agc" in args.methods:
                    all_method_candidates.extend(
                        run_autonomous_method(
                            runtime,
                            task_id,
                            prompt,
                            calibration,
                            learned_geometry,
                            random_geometry,
                            args,
                            store,
                        )
                    )

        except Exception as exc:
            store.add(
                "errors",
                {
                    "timestamp": utc_now(),
                    "stage": "model-run",
                    "model_key": model_key,
                    "model_id": spec.model_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            LOGGER.exception("Model run failed for %s", model_key)
        finally:
            unload_model_runtime(runtime)

    if args.cross_judge:
        selected = select_cross_judge_candidates(read_jsonl(store.paths["candidates"]))
        run_cross_jury(selected, args.models, args, store)

    finalize_outputs(store, args)
    LOGGER.info("Finished. Results written to %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
