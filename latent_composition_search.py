#!/usr/bin/env python3
"""
Latent Composition Search
=========================

A single-file experiment harness for comparing two activation-level search methods
across three local/open-weight transformer models:

  1. Contrastive Trajectory Inhibition (CTI)
     - Builds a layer-wise contrast between conventional and composition-seeking
       trajectories.
     - Fits low-rank "default-attractor" directions and a local positive-trajectory
       tangent subspace.
     - During generation, one-sided inhibition damps only activation components that
       exceed the composition trajectory's normal range in the default direction.
     - A weak tube penalty can also damp large departures from the positive trajectory.
     - Baseline and rank-matched random-subspace controls are included.

  2. Autonomous Granular Controller (AGC)
     - The same model acts as a controller over its own next search operation.
     - At every round it may choose layer segments, layer weights, intervention rank,
       inhibition strength, tube strength, token window, sampling parameters, branch
       count, and generation horizon.
     - It sees feedback from candidate quality proxies and intervention traces.
     - It can EXPLORE, REFINE, DIVERSIFY, HALT, or ANSWER.

Default models:
  - meta-llama/Llama-3.2-1B-Instruct
  - google/gemma-4-12B-it
  - openai/gpt-oss-20b

Outputs:
  - candidates.jsonl / candidates.csv
  - controller_actions.jsonl / controller_actions.csv
  - layer_geometry.jsonl / layer_geometry.csv
  - intervention_traces.jsonl / intervention_traces.csv
  - cross_judgments.jsonl / cross_judgments.csv (when --cross-judge is enabled)
  - summary.jsonl / summary.csv
  - metadata.json
  - figures/*.png and figures/*.pdf
  - geometry/*.pt

Recommended environment (latest stable releases are preferred because Gemma 4 is new):

  python -m venv .venv
  source .venv/bin/activate
  pip install -U torch "transformers>=5.5.0" accelerate numpy matplotlib
  pip install -U bitsandbytes   # optional, used by --quantization auto/4bit on CUDA only
  huggingface-cli login

Apple Silicon: CUDA is unavailable, so the harness auto-selects MPS when
torch.backends.mps.is_available(). BitsAndBytes 4-bit quantization stays CUDA-only.

Llama and Gemma checkpoints may require accepting their model licenses on the Hub.

This is an experimental mechanistic-interpretability harness. Its self-scores are
quality proxies, not evidence that an output is genuinely novel or scientifically true.
Use held-out evaluators, domain experts, retrieval, simulation, or experiments for any
substantive discovery claim.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
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
    "llama-3.2-1b": ModelSpec(
        key="llama-3.2-1b",
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        loader_kind="causal",
        description="Dense Llama decoder",
    ),
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
    "Develop one candidate research concept that could plausibly be valuable and "
    "that arises from a non-obvious composition of mechanisms from at least three "
    "distinct domains. The composition must be mechanistic rather than a loose "
    "analogy. State the component ideas, the mapping between them, the resulting "
    "mechanism, a concrete prediction, a falsification test, and the nearest known "
    "alternative explanation. Treat novelty as uncertain rather than established."
)

DISCOVERY_SYSTEM = (
    "You are participating in a controlled study of cross-domain composition. "
    "Be imaginative but epistemically strict. Do not claim that an idea is new merely "
    "because it sounds unusual. Separate established premises, the actual composition "
    "step, predicted consequences, and ways the proposal could fail. Return a complete "
    "candidate answer rather than private chain-of-thought."
)

CONVENTIONAL_INSTRUCTION = (
    "Give the most standard, conservative, and already-familiar treatment of the task. "
    "Prefer well-established concepts and the nearest obvious analogy. Avoid speculative "
    "cross-domain synthesis. This response is a calibration example of a conventional "
    "trajectory."
)

COMPOSITION_INSTRUCTION = (
    "Search deliberately for a non-obvious but defensible composition. Use mechanisms "
    "from at least three genuinely different domains. Reject mere word association. "
    "Present: (1) known ingredients, (2) exact cross-domain mapping, (3) composed "
    "mechanism, (4) testable prediction, (5) falsifier, (6) closest known alternative, "
    "and (7) uncertainty."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator of candidate cross-domain research concepts. Do not "
    "reward jargon, length, confidence, or unsupported novelty. Return only one JSON "
    "object in the final answer. Give concise externally checkable reasons, not hidden "
    "chain-of-thought."
)

CONTROLLER_SYSTEM = (
    "You are the autonomous controller for your own next search operation. You receive "
    "candidate outputs, quality proxies, and activation-intervention diagnostics. Choose "
    "one bounded control action. You may explore, refine, diversify, halt, or answer. "
    "Return only one JSON object in the final answer. Give only a brief operational "
    "reason, not private chain-of-thought."
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
            "summary": output_dir / "summary.jsonl",
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
    if (
        target_device.type == "cpu"
        and spec.key != "llama-3.2-1b"
        and not args.allow_cpu_large
    ):
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
    if len(collector.states) != len(runtime.layers):
        missing = sorted(set(range(len(runtime.layers))) - set(collector.states))
        raise RuntimeError(f"Failed to capture hidden states for layers: {missing[:10]}")
    return collector.states


@dataclass
class CalibrationSample:
    sample_id: str
    mode: str
    text: str
    result: GenerationResult
    score: dict[str, Any]
    states: dict[int, torch.Tensor]


@dataclass
class CalibrationBundle:
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
    mean_delta_norm: float
    singular_values: list[float]
    cumulative_energy: list[float]
    positive_samples: int
    negative_samples: int


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
                    "mean_delta_norm": geometry.mean_delta_norm,
                    "singular_values": geometry.singular_values,
                    "cumulative_energy": geometry.cumulative_energy,
                    "positive_samples": geometry.positive_samples,
                    "negative_samples": geometry.negative_samples,
                }
                for index, geometry in self.layers.items()
            },
        }


def orthonormal_random(dimension: int, rank: int, generator: torch.Generator) -> torch.Tensor:
    rank = max(1, min(rank, dimension))
    matrix = torch.randn(dimension, rank, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :rank].contiguous()


def safe_quantile(values: torch.Tensor, q: float, dim: int | None = None) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(0.0)
    return torch.quantile(values.float(), clamp(q, 0.0, 1.0), dim=dim)


def fit_one_layer_geometry(
    positive: torch.Tensor,
    negative: torch.Tensor,
    rank: int,
    positive_rank: int,
    threshold_quantile: float,
) -> LayerGeometry:
    positive = positive.float()
    negative = negative.float()
    center_positive = positive.mean(dim=0)
    center_negative = negative.mean(dim=0)
    mean_delta = center_negative - center_positive

    sample_count = min(len(positive), len(negative))
    paired = negative[:sample_count] - positive[:sample_count]
    design = torch.cat([mean_delta.unsqueeze(0) * math.sqrt(max(1, sample_count)), paired], dim=0)
    _, singular_values_tensor, vh = torch.linalg.svd(design, full_matrices=False)
    actual_rank = max(1, min(rank, vh.shape[0]))
    default_basis = vh[:actual_rank].T.contiguous()

    # Orient each basis vector so positive coefficients point toward the conventional/default side.
    orientation = mean_delta @ default_basis
    signs = torch.where(orientation >= 0, 1.0, -1.0)
    default_basis = default_basis * signs.unsqueeze(0)

    positive_relative = positive - center_positive
    positive_coefficients = positive_relative @ default_basis
    default_threshold = safe_quantile(positive_coefficients, threshold_quantile, dim=0)
    default_scale = positive_coefficients.std(dim=0, unbiased=False).clamp_min(1e-4)

    centered_positive = positive - center_positive
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

    projected_positive_q = positive_relative @ default_basis
    projected_negative_q = (negative - center_positive) @ default_basis
    mean_separation = projected_negative_q.mean(dim=0) - projected_positive_q.mean(dim=0)
    pooled_variance = 0.5 * (
        projected_negative_q.var(dim=0, unbiased=False)
        + projected_positive_q.var(dim=0, unbiased=False)
    )
    signal = float((mean_separation.square() / pooled_variance.clamp_min(1e-6)).mean().item())

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
        mean_delta_norm=float(torch.linalg.vector_norm(mean_delta).item()),
        singular_values=[float(v) for v in singular_values_tensor[: max(rank, 8)].tolist()],
        cumulative_energy=[float(v) for v in cumulative[: max(rank, 8)].tolist()],
        positive_samples=int(positive.shape[0]),
        negative_samples=int(negative.shape[0]),
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
    try:
        start_text, end_text = text.split(":", 1)
        start = clamp(float(start_text), 0.0, 1.0)
        end = clamp(float(end_text), 0.0, 1.0)
    except Exception as exc:
        raise ValueError("--fixed-band must look like 0.45:0.80") from exc
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
            mean_delta_norm=learned_layer.mean_delta_norm,
            singular_values=learned_layer.singular_values,
            cumulative_energy=learned_layer.cumulative_energy,
            positive_samples=learned_layer.positive_samples,
            negative_samples=learned_layer.negative_samples,
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
    negative_by_layer: dict[int, torch.Tensor] = {}
    layer_geometries: dict[int, LayerGeometry] = {}

    for layer_index in range(len(runtime.layers)):
        positive = torch.cat([sample.states[layer_index] for sample in calibration.positive], dim=0)
        negative = torch.cat([sample.states[layer_index] for sample in calibration.negative], dim=0)
        positive_by_layer[layer_index] = positive
        negative_by_layer[layer_index] = negative
        layer_geometries[layer_index] = fit_one_layer_geometry(
            positive=positive,
            negative=negative,
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
        geometry_name="learned",
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
                "signal": geometry.signal,
                "smoothed_signal": geometry.smoothed_signal,
                "mean_delta_norm": geometry.mean_delta_norm,
                "selected": layer_index in selected,
                "layer_weight": weights.get(layer_index, 0.0),
                "rank": actual_rank,
                "positive_rank": actual_positive_rank,
                "tube_radius": geometry.tube_radius,
                "singular_values": geometry.singular_values,
                "cumulative_energy": geometry.cumulative_energy,
                "positive_samples": geometry.positive_samples,
                "negative_samples": geometry.negative_samples,
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


class TrajectoryIntervention:
    def __init__(
        self,
        runtime: ModelRuntime,
        geometry: TrajectoryGeometry,
        settings: InterventionSettings,
    ) -> None:
        self.runtime = runtime
        self.geometry = geometry
        self.settings = settings
        self.handles: list[Any] = []
        self.trace: list[dict[str, Any]] = []
        self.step = -1
        self.device_cache: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        weights = settings.layer_weights or geometry.layer_weights
        self.layer_weights = {
            int(index): float(weight)
            for index, weight in weights.items()
            if 0 <= int(index) < len(runtime.layers) and float(weight) > 0
        }
        if not self.layer_weights:
            self.layer_weights = {index: 1.0 for index in geometry.selected_layers}
        self.first_layer = min(self.layer_weights)

    def _tensors(self, layer_index: int, device: torch.device) -> dict[str, torch.Tensor]:
        key = (layer_index, str(device))
        if key in self.device_cache:
            return self.device_cache[key]
        geometry = self.geometry.layers[layer_index]
        tensors = {
            "center": geometry.center_positive.to(device=device, dtype=torch.float32),
            "q": geometry.default_basis.to(device=device, dtype=torch.float32),
            "threshold": geometry.default_threshold.to(device=device, dtype=torch.float32),
            "scale": geometry.default_scale.to(device=device, dtype=torch.float32),
            "p": geometry.positive_basis.to(device=device, dtype=torch.float32),
        }
        self.device_cache[key] = tensors
        return tensors

    def _make_hook(self, layer_index: int):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            hidden, replace = hidden_from_output(output)
            if hidden is None or hidden.ndim != 3:
                return output
            if layer_index == self.first_layer:
                self.step += 1
            current_step = max(0, self.step)

            state = hidden[:, -1, :].float()
            tensors = self._tensors(layer_index, state.device)
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
            radius = float(self.geometry.layers[layer_index].tube_radius)
            tube_excess = torch.relu(orthogonal_norm - radius)
            tube_delta = orthogonal * (tube_excess / orthogonal_norm.clamp_min(1e-6))

            in_token_window = current_step >= self.settings.token_start and (
                self.settings.token_end is None or current_step <= self.settings.token_end
            )
            layer_weight = self.layer_weights.get(layer_index, 0.0)
            gate = torch.ones_like(drift)
            if self.settings.adaptive_gate:
                gate = torch.sigmoid(
                    (drift - 1.0) / max(1e-3, self.settings.gate_temperature)
                )

            delta = torch.zeros_like(state)
            has_effect = self.settings.default_strength > 0 or self.settings.tube_strength > 0
            if in_token_window and layer_weight > 0 and has_effect:
                delta = (
                    self.settings.default_strength
                    * layer_weight
                    * gate.unsqueeze(-1)
                    * default_delta
                    + self.settings.tube_strength * layer_weight * tube_delta
                )

                state_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-6)
                delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                allowed = self.settings.max_delta_ratio * state_norm
                scale_down = torch.minimum(torch.ones_like(delta_norm), allowed / delta_norm.clamp_min(1e-6))
                delta = delta * scale_down
                new_state = state - delta
                if self.settings.preserve_rms:
                    old_rms = torch.sqrt(state.square().mean(dim=-1, keepdim=True).clamp_min(1e-8))
                    new_rms = torch.sqrt(new_state.square().mean(dim=-1, keepdim=True).clamp_min(1e-8))
                    new_state = new_state * (old_rms / new_rms)
                new_hidden = hidden.clone()
                new_hidden[:, -1, :] = new_state.to(hidden.dtype)
                output = replace(new_hidden)

            state_norm_scalar = torch.linalg.vector_norm(state, dim=-1).mean().item()
            delta_ratio = (
                torch.linalg.vector_norm(delta, dim=-1).mean().item()
                / max(1e-6, state_norm_scalar)
            )
            self.trace.append(
                {
                    "step": current_step,
                    "layer_index": layer_index,
                    "layer_fraction": layer_index / max(1, len(self.runtime.layers) - 1),
                    "layer_weight": layer_weight,
                    "drift": float(drift.mean().item()),
                    "drift_max": float(drift.max().item()),
                    "gate": float(gate.mean().item()),
                    "tube_excess": float(tube_excess.mean().item()),
                    "delta_ratio": float(delta_ratio),
                    "applied": bool(in_token_window and layer_weight > 0 and delta_ratio > 0),
                    "label": self.settings.label,
                }
            )
            return output
        return hook

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
        self.device_cache.clear()

    def summary(self) -> dict[str, Any]:
        if not self.trace:
            return {
                "mean_drift": 0.0,
                "max_drift": 0.0,
                "mean_delta_ratio": 0.0,
                "max_delta_ratio": 0.0,
                "top_drift_layer": None,
                "trace_rows": 0,
            }
        drifts = np.array([row["drift"] for row in self.trace], dtype=float)
        deltas = np.array([row["delta_ratio"] for row in self.trace], dtype=float)
        by_layer: dict[int, list[float]] = defaultdict(list)
        for row in self.trace:
            by_layer[int(row["layer_index"])].append(float(row["drift"]))
        top_layer = max(by_layer, key=lambda index: float(np.mean(by_layer[index])))
        return {
            "mean_drift": float(np.mean(drifts)),
            "max_drift": float(np.max(drifts)),
            "mean_delta_ratio": float(np.mean(deltas)),
            "max_delta_ratio": float(np.max(deltas)),
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
    "plausibility": 5.0,
    "testability": 5.0,
    "specificity": 5.0,
    "nontriviality": 5.0,
    "overall": 5.0,
    "confidence": 0.25,
    "brief_evidence": "Judge output could not be parsed.",
    "failure_modes": [],
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
        "plausibility",
        "testability",
        "specificity",
        "nontriviality",
        "overall",
    ]
    for field_name in score_fields:
        result[field_name] = clamp(safe_float(parsed.get(field_name), 5.0), 0.0, 10.0)
    result["confidence"] = clamp(safe_float(parsed.get("confidence"), 0.5), 0.0, 1.0)
    result["brief_evidence"] = str(parsed.get("brief_evidence", ""))[:1200]
    failure_modes = parsed.get("failure_modes", [])
    if not isinstance(failure_modes, list):
        failure_modes = [str(failure_modes)]
    result["failure_modes"] = [str(item)[:300] for item in failure_modes[:8]]
    result["valid_json"] = True
    return result


def judge_candidate(
    runtime: ModelRuntime,
    task: str,
    candidate: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], str]:
    rubric = {
        "novelty": "0-10: non-obvious relative to common treatments, without assuming true priority",
        "cross_domain_integration": "0-10: domains are genuinely distinct and causally integrated",
        "mechanistic_coherence": "0-10: mappings and mechanism form a coherent causal story",
        "plausibility": "0-10: compatible with known constraints; unsupported leaps are penalized",
        "testability": "0-10: produces discriminating predictions and feasible falsifiers",
        "specificity": "0-10: concrete enough to implement or investigate",
        "nontriviality": "0-10: more than rewording, analogy, or generic brainstorming",
        "overall": "0-10: conservative aggregate",
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


def candidate_reward(score: Mapping[str, Any], diversity: float, repetition: float) -> float:
    weighted = (
        0.24 * safe_float(score.get("overall"), 0.0)
        + 0.15 * safe_float(score.get("novelty"), 0.0)
        + 0.13 * safe_float(score.get("cross_domain_integration"), 0.0)
        + 0.14 * safe_float(score.get("mechanistic_coherence"), 0.0)
        + 0.14 * safe_float(score.get("plausibility"), 0.0)
        + 0.14 * safe_float(score.get("testability"), 0.0)
        + 0.06 * safe_float(score.get("specificity"), 0.0)
    )
    return float(weighted + 1.5 * diversity - 1.5 * repetition)


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
    samples: dict[str, list[CalibrationSample]] = {"conventional": [], "exploratory": []}
    for mode in ("conventional", "exploratory"):
        for sample_index in range(args.calibration_samples):
            seed = args.seed + sample_index + (0 if mode == "conventional" else 10_000)
            temperature = args.conventional_temperature if mode == "conventional" else args.exploratory_temperature
            messages = discovery_messages(
                task,
                mode,
                extra_focus=(
                    "Use a distinct line of attack for calibration replicate "
                    f"{sample_index + 1}."
                    if mode == "exploratory"
                    else ""
                ),
            )
            result = generate_text(
                runtime,
                messages,
                seed=seed,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                store_raw=args.store_raw_generations,
            )
            score, judge_raw = judge_candidate(runtime, task, result.text, args, seed + 50_000)
            states = capture_sequence_states(
                runtime,
                result.full_token_ids,
                result.generated_tokens,
                args.state_tokens,
            )
            sample_id = make_candidate_id(runtime.spec.key, f"calibration-{mode}", task_id)
            sample = CalibrationSample(
                sample_id=sample_id,
                mode=mode,
                text=result.text,
                result=result,
                score=score,
                states=states,
            )
            samples[mode].append(sample)
            record = {
                "candidate_id": sample_id,
                "timestamp": utc_now(),
                "task_id": task_id,
                "task": task,
                "model_key": runtime.spec.key,
                "model_id": runtime.spec.model_id,
                "method": "shared-calibration",
                "stage": "calibration",
                "arm": mode,
                "sample_index": sample_index,
                "seed": seed,
                "text": result.text,
                "raw_text": result.raw_text,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "generation_seconds": result.seconds,
                "scores": score,
                "judge_raw": judge_raw,
            }
            store.add("candidates", record)
            LOGGER.info(
                "%s | %s | calibration %s %d/%d | overall %.2f",
                runtime.spec.key,
                task_id,
                mode,
                sample_index + 1,
                args.calibration_samples,
                score["overall"],
            )

    exploratory_sorted = sorted(
        samples["exploratory"], key=lambda sample: sample.score.get("overall", 0.0), reverse=True
    )
    positive_count = max(2, int(math.ceil(len(exploratory_sorted) * 0.60)))
    positive = exploratory_sorted[:positive_count]
    negative = list(samples["conventional"])
    low_count = max(0, min(len(exploratory_sorted) - positive_count, len(exploratory_sorted) // 4))
    if low_count:
        negative.extend(exploratory_sorted[-low_count:])
    if len(negative) < 2:
        negative.extend(exploratory_sorted[-2:])
    return CalibrationBundle(
        conventional=samples["conventional"],
        exploratory=samples["exploratory"],
        positive=positive,
        negative=negative,
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
    arms: list[tuple[str, TrajectoryGeometry, float, float]] = [
        ("baseline", learned_geometry, 0.0, 0.0),
        (
            "learned-inhibition",
            learned_geometry,
            args.inhibition_strength,
            args.tube_strength,
        ),
        (
            "random-subspace-control",
            random_geometry,
            args.inhibition_strength,
            args.tube_strength,
        ),
    ]

    for arm, geometry, default_strength, tube_strength in arms:
        for candidate_index in range(args.candidates_per_arm):
            seed = args.seed + 100_000 + candidate_index
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
            )
            intervention = TrajectoryIntervention(runtime, geometry, settings)
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
            reward = candidate_reward(score, diversity, repetition)
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
                "seed": seed,
                "text": result.text,
                "raw_text": result.raw_text,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "generation_seconds": result.seconds,
                "scores": score,
                "judge_raw": judge_raw,
                "lexical_novelty": diversity,
                "repetition_ratio": repetition,
                "reward": reward,
                "intervention": {
                    "geometry": geometry.geometry_name,
                    "selected_layers": geometry.selected_layers,
                    "layer_weights": geometry.layer_weights,
                    "default_strength": default_strength,
                    "tube_strength": tube_strength,
                    "max_delta_ratio": args.max_delta_ratio,
                    "rank": geometry.rank,
                    "positive_rank": geometry.positive_rank,
                    "token_start": args.intervention_token_start,
                    "token_end": args.intervention_token_end,
                },
                "trace_summary": trace_summary,
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
                    "method": "contrastive-trajectory-inhibition",
                    "arm": arm,
                },
            )
            outputs.append(record)
            LOGGER.info(
                "%s | %s | CTI %s %d/%d | overall %.2f | reward %.2f",
                runtime.spec.key,
                task_id,
                arm,
                candidate_index + 1,
                args.candidates_per_arm,
                score["overall"],
                reward,
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
            focus="Generate a distinctly mechanistic composition and make the falsifier unusually concrete.",
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
        "lexical_novelty": candidate.get("lexical_novelty"),
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
        seed=args.seed + 700_000 + round_index,
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
        "explore": "Create a new candidate from a fresh mechanistic composition.",
        "diversify": "Use a substantially different set of domains and mechanism than the prior candidates.",
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
                + "\n\nProduce one final, self-contained candidate. Preserve the strongest "
                "mechanistic composition, resolve avoidable weaknesses, state uncertainty, "
                "and include a discriminating falsification test. Do not claim verified novelty."
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
    conventional_references = [sample.text for sample in calibration.conventional]
    pool: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    # Seed the controller with the best exploratory calibration candidates, but mark them
    # as inherited context rather than method outputs.
    for sample in sorted(
        calibration.exploratory,
        key=lambda item: safe_float(item.score.get("overall"), 0.0),
        reverse=True,
    )[: min(2, len(calibration.exploratory))]:
        diversity = lexical_novelty(sample.text, conventional_references)
        rep = repetition_ratio(sample.text)
        pool.append(
            {
                "candidate_id": sample.sample_id,
                "round": -1,
                "arm": "calibration-seed",
                "text": sample.text,
                "scores": sample.score,
                "lexical_novelty": diversity,
                "repetition_ratio": rep,
                "reward": candidate_reward(sample.score, diversity, rep),
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
        )
        store.add(
            "controller_actions",
            {
                "timestamp": utc_now(),
                "task_id": task_id,
                "model_key": runtime.spec.key,
                "model_id": runtime.spec.model_id,
                "method": "autonomous-granular-controller",
                "round": round_index,
                "action": dataclasses.asdict(action),
                "controller_raw": controller_raw,
                "pool_size": len(pool),
            },
        )
        LOGGER.info(
            "%s | %s | AGC round %d | action=%s geometry=%s strength=%.2f",
            runtime.spec.key,
            task_id,
            round_index,
            action.action,
            action.geometry,
            action.inhibition_strength,
        )

        if action.action in {"halt", "answer"}:
            if action.answer.strip():
                final_text = action.answer.strip()
                final_result = None
            else:
                final_result = synthesize_final_answer(
                    runtime,
                    task,
                    pool,
                    args,
                    seed=args.seed + 850_000 + round_index,
                )
                final_text = final_result.text
            score, judge_raw = judge_candidate(
                runtime,
                task,
                final_text,
                args,
                seed=args.seed + 860_000 + round_index,
            )
            diversity = lexical_novelty(final_text, conventional_references)
            rep = repetition_ratio(final_text)
            reward = candidate_reward(score, diversity, rep)
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
                "arm": action.action,
                "round": round_index,
                "seed": args.seed + 850_000 + round_index,
                "text": final_text,
                "raw_text": final_result.raw_text if final_result else None,
                "prompt_tokens": final_result.prompt_tokens if final_result else None,
                "generated_tokens": final_result.generated_tokens if final_result else None,
                "generation_seconds": final_result.seconds if final_result else None,
                "scores": score,
                "judge_raw": judge_raw,
                "lexical_novelty": diversity,
                "repetition_ratio": rep,
                "reward": reward,
                "controller_action": dataclasses.asdict(action),
                "trace_summary": {},
            }
            store.add("candidates", record)
            outputs.append(record)
            halted = True
            break

        ranked_pool = sorted(pool, key=lambda row: safe_float(row.get("reward"), -999.0), reverse=True)
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
            seed = args.seed + 800_000 + round_index * 1000 + branch_index
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
            )
            intervention = TrajectoryIntervention(runtime, geometry, settings)
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
            references = conventional_references + [str(candidate.get("text", "")) for candidate in pool]
            diversity = lexical_novelty(result.text, references)
            rep = repetition_ratio(result.text)
            reward = candidate_reward(score, diversity, rep)
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
                "branch_index": branch_index,
                "parent_id": parent.get("candidate_id") if parent else None,
                "seed": seed,
                "text": result.text,
                "raw_text": result.raw_text,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "generation_seconds": result.seconds,
                "scores": score,
                "judge_raw": judge_raw,
                "lexical_novelty": diversity,
                "repetition_ratio": rep,
                "reward": reward,
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
                },
                "trace_summary": trace_summary,
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
                    "round": round_index,
                },
            )
            pool.append(record)
            outputs.append(record)
            LOGGER.info(
                "%s | %s | AGC round %d branch %d/%d | overall %.2f | reward %.2f",
                runtime.spec.key,
                task_id,
                round_index,
                branch_index + 1,
                action.num_branches,
                score["overall"],
                reward,
            )

    if not halted:
        final_result = synthesize_final_answer(
            runtime,
            task,
            pool,
            args,
            seed=args.seed + 899_000,
        )
        score, judge_raw = judge_candidate(
            runtime,
            task,
            final_result.text,
            args,
            seed=args.seed + 899_500,
        )
        diversity = lexical_novelty(final_result.text, conventional_references)
        rep = repetition_ratio(final_result.text)
        reward = candidate_reward(score, diversity, rep)
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
            "arm": "max-rounds-synthesis",
            "round": args.controller_rounds,
            "seed": args.seed + 899_000,
            "text": final_result.text,
            "raw_text": final_result.raw_text,
            "prompt_tokens": final_result.prompt_tokens,
            "generated_tokens": final_result.generated_tokens,
            "generation_seconds": final_result.seconds,
            "scores": score,
            "judge_raw": judge_raw,
            "lexical_novelty": diversity,
            "repetition_ratio": rep,
            "reward": reward,
            "trace_summary": {},
        }
        store.add("candidates", record)
        outputs.append(record)
    return outputs


# -----------------------------------------------------------------------------
# Optional cross-model jury
# -----------------------------------------------------------------------------


def select_cross_judge_candidates(
    candidate_records: Sequence[Mapping[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        if record.get("stage") not in {"search", "final"}:
            continue
        key = (
            str(record.get("task_id")),
            str(record.get("model_key")),
            str(record.get("method")),
        )
        groups[key].append(dict(record))
    selected: list[dict[str, Any]] = []
    for rows in groups.values():
        rows.sort(key=lambda row: safe_float(row.get("reward"), -999.0), reverse=True)
        selected.extend(rows[:top_k])
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
            for index, candidate in enumerate(selected_candidates):
                score, raw = judge_candidate(
                    runtime,
                    str(candidate.get("task", "")),
                    str(candidate.get("text", "")),
                    args,
                    seed=args.seed + 950_000 + index,
                )
                store.add(
                    "cross_judgments",
                    {
                        "timestamp": utc_now(),
                        "candidate_id": candidate.get("candidate_id"),
                        "task_id": candidate.get("task_id"),
                        "source_model_key": candidate.get("model_key"),
                        "source_method": candidate.get("method"),
                        "source_arm": candidate.get("arm"),
                        "judge_model_key": judge_key,
                        "judge_model_id": runtime.spec.model_id,
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


# -----------------------------------------------------------------------------
# Summaries and plotting
# -----------------------------------------------------------------------------


def aggregate_summary(candidate_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        if record.get("stage") not in {"search", "final"}:
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
        def values(path: str) -> list[float]:
            first, second = path.split(".", 1)
            result: list[float] = []
            for row in rows:
                container = row.get(first, {})
                if isinstance(container, Mapping):
                    result.append(safe_float(container.get(second), float("nan")))
            return [value for value in result if math.isfinite(value)]

        overall = values("scores.overall")
        novelty = values("scores.novelty")
        plausibility = values("scores.plausibility")
        testability = values("scores.testability")
        rewards = [safe_float(row.get("reward"), float("nan")) for row in rows]
        rewards = [value for value in rewards if math.isfinite(value)]
        summary.append(
            {
                "timestamp": utc_now(),
                "task_id": task_id,
                "model_key": model_key,
                "method": method,
                "arm": arm,
                "n": len(rows),
                "overall_mean": float(np.mean(overall)) if overall else None,
                "overall_std": float(np.std(overall)) if overall else None,
                "novelty_mean": float(np.mean(novelty)) if novelty else None,
                "plausibility_mean": float(np.mean(plausibility)) if plausibility else None,
                "testability_mean": float(np.mean(testability)) if testability else None,
                "reward_mean": float(np.mean(rewards)) if rewards else None,
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
    actions = read_jsonl(output_dir / "controller_actions.jsonl")
    layers = read_jsonl(output_dir / "layer_geometry.jsonl")
    traces = read_jsonl(output_dir / "intervention_traces.jsonl")
    judgments = read_jsonl(output_dir / "cross_judgments.jsonl")
    summary = read_jsonl(output_dir / "summary.jsonl")
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # 1. Mean overall score by model/method/arm.
    rows = [row for row in summary if row.get("overall_mean") is not None]
    if rows:
        labels = [f"{row['model_key']}\n{row['method']}\n{row['arm']}" for row in rows]
        values = [safe_float(row.get("overall_mean")) for row in rows]
        plt.figure(figsize=(max(10, len(rows) * 0.65), 6))
        plt.bar(range(len(rows)), values)
        plt.xticks(range(len(rows)), labels, rotation=75, ha="right")
        plt.ylabel("Mean self-judged overall score (0-10)")
        plt.title("Candidate Quality by Model, Method, and Arm")
        save_figure(plt, figures / "01_quality_by_model_method_arm")

    # 2. Novelty-plausibility tradeoff.
    search_candidates = [row for row in candidates if row.get("stage") in {"search", "final"}]
    if search_candidates:
        plt.figure(figsize=(9, 7))
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in search_candidates:
            groups[f"{row.get('model_key')} | {row.get('method')}"] .append(row)
        for label, group in groups.items():
            x = [safe_float(row.get("scores", {}).get("novelty")) for row in group]
            y = [safe_float(row.get("scores", {}).get("plausibility")) for row in group]
            plt.scatter(x, y, label=label, alpha=0.75)
        plt.xlabel("Novelty proxy (0-10)")
        plt.ylabel("Plausibility proxy (0-10)")
        plt.title("Novelty-Plausibility Tradeoff")
        plt.legend(fontsize=8)
        save_figure(plt, figures / "02_novelty_plausibility")

    # 3. Cross-domain integration vs testability.
    if search_candidates:
        plt.figure(figsize=(9, 7))
        groups = defaultdict(list)
        for row in search_candidates:
            groups[f"{row.get('model_key')} | {row.get('method')}"] .append(row)
        for label, group in groups.items():
            x = [safe_float(row.get("scores", {}).get("cross_domain_integration")) for row in group]
            y = [safe_float(row.get("scores", {}).get("testability")) for row in group]
            plt.scatter(x, y, label=label, alpha=0.75)
        plt.xlabel("Cross-domain integration proxy (0-10)")
        plt.ylabel("Testability proxy (0-10)")
        plt.title("Composition Strength versus Testability")
        plt.legend(fontsize=8)
        save_figure(plt, figures / "03_composition_testability")

    # 4. Layer signal profiles on normalized depth.
    if layers:
        plt.figure(figsize=(10, 6))
        groups = defaultdict(list)
        for row in layers:
            groups[(row.get("task_id"), row.get("model_key"))].append(row)
        for (task_id, model_key), group in groups.items():
            group = sorted(group, key=lambda row: safe_int(row.get("layer_index")))
            x = [safe_float(row.get("layer_fraction")) for row in group]
            y = [safe_float(row.get("smoothed_signal")) for row in group]
            plt.plot(x, y, marker="o", markersize=3, label=f"{model_key} | {task_id}")
        plt.xlabel("Normalized layer depth")
        plt.ylabel("Smoothed contrastive signal")
        plt.title("Layer-wise Conventional-versus-Composition Signal")
        plt.legend(fontsize=8)
        save_figure(plt, figures / "04_layer_signal_profiles")

    # 5. Selected band location by architecture.
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
        plt.title("Automatically Selected Intervention Bands")
        save_figure(plt, figures / "05_selected_layer_bands")

    # 6. Average cumulative singular energy.
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
            mean_curve = np.nanmean(matrix, axis=0)
            plt.plot(range(1, max_len + 1), mean_curve, marker="o", label=model_key)
        plt.xlabel("Rank")
        plt.ylabel("Mean cumulative delta energy")
        plt.ylim(0, 1.05)
        plt.title("Low-rank Saturation in Selected Layer Bands")
        plt.legend()
        save_figure(plt, figures / "06_cumulative_singular_energy")

    # 7. Drift by generated step.
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
            values = [float(np.mean(by_step[step])) for step in steps]
            plt.plot(steps, values, label=" | ".join(str(item) for item in key))
        plt.xlabel("Generation forward step")
        plt.ylabel("Mean normalized default-side drift")
        plt.title("Default-Trajectory Drift During Generation")
        plt.legend(fontsize=7)
        save_figure(plt, figures / "07_drift_over_generation")

    # 8. Intervention magnitude by generated step.
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
            values = [float(np.mean(by_step[step])) for step in steps]
            plt.plot(steps, values, label=" | ".join(str(item) for item in key))
        plt.xlabel("Generation forward step")
        plt.ylabel("Mean intervention delta / hidden-state norm")
        plt.title("Activation Intervention Magnitude")
        plt.legend(fontsize=7)
        save_figure(plt, figures / "08_intervention_magnitude")

    # 9. Controller reward by round.
    agc_candidates = [
        row
        for row in search_candidates
        if row.get("method") == "autonomous-granular-controller" and row.get("round") is not None
    ]
    if agc_candidates:
        plt.figure(figsize=(10, 6))
        groups = defaultdict(list)
        for row in agc_candidates:
            groups[str(row.get("model_key"))].append(row)
        for model_key, group in groups.items():
            by_round = defaultdict(list)
            for row in group:
                by_round[safe_int(row.get("round"))].append(safe_float(row.get("reward")))
            rounds = sorted(by_round)
            means = [float(np.mean(by_round[round_index])) for round_index in rounds]
            plt.plot(rounds, means, marker="o", label=model_key)
        plt.xlabel("Controller round")
        plt.ylabel("Mean candidate reward proxy")
        plt.title("Autonomous Search Feedback")
        plt.legend()
        save_figure(plt, figures / "09_controller_reward_by_round")

    # 10-13. Controller settings timelines as separate figures.
    if actions:
        settings_to_plot = [
            ("inhibition_strength", "Inhibition strength", "10_controller_inhibition_strength"),
            ("tube_strength", "Tube strength", "11_controller_tube_strength"),
            ("temperature", "Sampling temperature", "12_controller_temperature"),
            ("horizon_tokens", "Generation horizon (tokens)", "13_controller_horizon"),
        ]
        for field_name, ylabel, filename in settings_to_plot:
            plt.figure(figsize=(10, 6))
            groups = defaultdict(list)
            for row in actions:
                groups[str(row.get("model_key"))].append(row)
            for model_key, group in groups.items():
                group = sorted(group, key=lambda row: safe_int(row.get("round")))
                x = [safe_int(row.get("round")) for row in group]
                y = [safe_float(row.get("action", {}).get(field_name)) for row in group]
                plt.plot(x, y, marker="o", label=model_key)
            plt.xlabel("Controller round")
            plt.ylabel(ylabel)
            plt.title(ylabel + " Chosen by the Autonomous Controller")
            plt.legend()
            save_figure(plt, figures / filename)

    # 14. Architecture-by-method score heatmap.
    if summary:
        model_keys = sorted({str(row.get("model_key")) for row in summary})
        methods = sorted({str(row.get("method")) for row in summary})
        matrix = np.full((len(methods), len(model_keys)), np.nan)
        for method_index, method in enumerate(methods):
            for model_index, model_key in enumerate(model_keys):
                values = [
                    safe_float(row.get("overall_mean"), float("nan"))
                    for row in summary
                    if row.get("method") == method and row.get("model_key") == model_key
                ]
                values = [value for value in values if math.isfinite(value)]
                if values:
                    matrix[method_index, model_index] = float(np.mean(values))
        plt.figure(figsize=(8, 5))
        image = plt.imshow(matrix, aspect="auto")
        plt.colorbar(image, label="Mean overall score")
        plt.xticks(range(len(model_keys)), model_keys, rotation=30, ha="right")
        plt.yticks(range(len(methods)), methods)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if math.isfinite(value):
                    plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
        plt.title("Architecture × Search Method")
        save_figure(plt, figures / "14_architecture_method_heatmap")

    # 15. Output length versus quality.
    length_candidates = [
        row for row in search_candidates if row.get("generated_tokens") is not None
    ]
    if length_candidates:
        plt.figure(figsize=(9, 7))
        groups = defaultdict(list)
        for row in length_candidates:
            groups[str(row.get("model_key"))].append(row)
        for model_key, group in groups.items():
            x = [safe_float(row.get("generated_tokens")) for row in group]
            y = [safe_float(row.get("scores", {}).get("overall")) for row in group]
            plt.scatter(x, y, label=model_key, alpha=0.75)
        plt.xlabel("Generated tokens")
        plt.ylabel("Overall quality proxy")
        plt.title("Output Length versus Quality")
        plt.legend()
        save_figure(plt, figures / "15_output_length_quality")

    # 16. Generation time by model and method.
    timing_rows = [row for row in summary if row.get("generation_seconds_mean") is not None]
    if timing_rows:
        labels = [f"{row['model_key']}\n{row['method']}\n{row['arm']}" for row in timing_rows]
        values = [safe_float(row.get("generation_seconds_mean")) for row in timing_rows]
        plt.figure(figsize=(max(10, len(labels) * 0.65), 6))
        plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
        plt.ylabel("Mean generation seconds")
        plt.title("Generation Cost by Model, Method, and Arm")
        save_figure(plt, figures / "16_generation_time")

    # 17. Cross-judge source/judge matrix.
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
                ]
                values = [value for value in values if math.isfinite(value)]
                if values:
                    matrix[source_index, judge_index] = float(np.mean(values))
        plt.figure(figsize=(8, 6))
        image = plt.imshow(matrix, aspect="auto")
        plt.colorbar(image, label="Mean jury score")
        plt.xticks(range(len(judge_models)), judge_models, rotation=30, ha="right")
        plt.yticks(range(len(source_models)), source_models)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if math.isfinite(value):
                    plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
        plt.xlabel("Judge model")
        plt.ylabel("Source model")
        plt.title("Cross-model Jury Matrix")
        save_figure(plt, figures / "17_cross_jury_matrix")


def finalize_outputs(store: RecordStore, args: argparse.Namespace) -> None:
    candidate_records = read_jsonl(store.paths["candidates"])
    summary_records = aggregate_summary(candidate_records)
    for record in summary_records:
        store.add("summary", record)
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
            "candidates_per_arm": 1,
            "controller_rounds": 2,
            "controller_branches": 1,
            "controller_max_branches": 2,
            "max_new_tokens": 128,
            "controller_horizon_tokens": 96,
        },
        "standard": {
            "calibration_samples": 4,
            "candidates_per_arm": 3,
            "controller_rounds": 4,
            "controller_branches": 2,
            "controller_max_branches": 3,
            "max_new_tokens": 320,
            "controller_horizon_tokens": 220,
        },
        "serious": {
            "calibration_samples": 8,
            "candidates_per_arm": 8,
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

    parser.add_argument("--conventional-temperature", type=float, default=0.30)
    parser.add_argument("--exploratory-temperature", type=float, default=1.00)
    parser.add_argument("--search-temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--inhibition-strength", type=float, default=0.55)
    parser.add_argument("--tube-strength", type=float, default=0.08)
    parser.add_argument("--max-delta-ratio", type=float, default=0.12)
    parser.add_argument("--gate-temperature", type=float, default=0.75)
    parser.add_argument("--intervention-token-start", type=int, default=0)
    parser.add_argument("--intervention-token-end", type=int)
    parser.add_argument("--disable-adaptive-gate", action="store_true")
    parser.add_argument("--disable-rms-preservation", action="store_true")

    parser.add_argument("--controller-rounds", type=int, default=None)
    parser.add_argument("--controller-branches", type=int, default=None)
    parser.add_argument("--controller-max-branches", type=int, default=None)
    parser.add_argument("--controller-horizon-tokens", type=int, default=None)
    parser.add_argument("--controller-beam-width", type=int, default=4)

    parser.add_argument("--cross-judge", action="store_true")
    parser.add_argument("--cross-judge-top-k", type=int, default=2)
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
            "cti": "One-sided learned default-direction inhibition plus weak positive-trajectory tube, with baseline and random controls.",
            "agc": "Same-model feedback controller over layer/time/intervention/sampling controls with halt and answer actions.",
        },
        "epistemic_warning": (
            "Self-judged novelty and quality are proxies. Candidate outputs require independent "
            "literature review, domain expertise, and empirical or formal validation."
        ),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    apply_preset(args)
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
        selected = select_cross_judge_candidates(
            read_jsonl(store.paths["candidates"]), args.cross_judge_top_k
        )
        run_cross_jury(selected, args.models, args, store)

    finalize_outputs(store, args)
    LOGGER.info("Finished. Results written to %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
