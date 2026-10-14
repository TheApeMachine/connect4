"""Command-line experiment runner for the Rotoformer MVP."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping

import torch

from .adapter import RotoformerAdapter, RotoformerConfig
from .scoring import CandidateScore, choose_candidate, sweep_rotations


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def parse_rotations(text: str, prime: int) -> list[int]:
    raw = text.strip().lower()
    if raw == "all":
        # k=0..p-1 follows one full translation orbit while the multiplier
        # traverses its p-1 exponent positions.
        return list(range(prime))

    if raw.startswith("even:"):
        count = int(raw.split(":", 1)[1])
        if count < 1:
            raise ValueError("even:<count> requires count >= 1")
        if count == 1:
            return [0]
        return sorted(
            {
                round(i * (prime - 1) / (count - 1))
                for i in range(count)
            }
        )

    if raw.startswith("range:"):
        payload = raw.split(":", 1)[1]
        step = 1
        if ":" in payload:
            payload, step_text = payload.rsplit(":", 1)
            step = int(step_text)
        left, right = payload.split("-", 1)
        start, stop = int(left), int(right)
        if step == 0:
            raise ValueError("rotation range step cannot be zero")
        if start <= stop and step < 0:
            step = -step
        if start > stop and step > 0:
            step = -step
        stop_inclusive = stop + (1 if step > 0 else -1)
        return list(range(start, stop_inclusive, step))

    values = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if not values:
        raise ValueError("rotation list is empty")
    # Preserve user order while removing duplicates.
    return list(dict.fromkeys(values))


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        embedding = model.get_input_embeddings()
        if embedding is not None:
            return embedding.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_inputs(inputs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def build_prompt_inputs(
    tokenizer: Any,
    *,
    prompt: str,
    system: str | None,
    raw_prompt: bool,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], str]:
    rendered = prompt
    if not raw_prompt and hasattr(tokenizer, "apply_chat_template"):
        messages: list[dict[str, str]] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": prompt})
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            rendered = (system.strip() + "\n\n" if system and system.strip() else "") + prompt
    elif system and system.strip():
        rendered = system.strip() + "\n\n" + prompt

    encoded = tokenizer(rendered, return_tensors="pt")
    return _move_inputs(encoded, device), rendered


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    inputs: Mapping[str, torch.Tensor],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    set_seed(seed)
    do_sample = temperature > 0
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _ranked(scores: Iterable[CandidateScore], objective: str) -> list[CandidateScore]:
    values = list(scores)
    if objective == "prompt_nll":
        return sorted(values, key=lambda item: (item.prompt_nll, item.k))
    if objective == "entropy":
        return sorted(values, key=lambda item: (item.last_token_entropy, item.k))
    if objective == "margin":
        return sorted(values, key=lambda item: (-item.top2_margin, item.k))
    raise ValueError(objective)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face runner requires the optional dependencies. Install with: "
            "pip install -e '.[hf]'"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Brute-force finite-field rotations over a frozen causal LM, select a rotation "
            "from prompt-side inference metrics, then generate under the winning frame."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--system", default=None)
    parser.add_argument("--raw-prompt", action="store_true")

    parser.add_argument("--layer-spec", default="late")
    parser.add_argument("--scope", choices=["span", "layerwise"], default="span")
    parser.add_argument("--depth-stride", type=int, default=0)
    parser.add_argument("--prime", type=int, default=257)
    parser.add_argument("--generator", type=int, default=3)
    parser.add_argument("--translation-mode", choices=["k", "zero"], default="k")
    parser.add_argument(
        "--rotations",
        default="even:17",
        help="Comma list, all, even:N, or range:A-B[:STEP]",
    )
    parser.add_argument(
        "--objective", choices=["prompt_nll", "entropy", "margin"], default="prompt_nll"
    )

    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--device-map",
        default="none",
        help="Use 'auto' for Accelerate sharding/offload, or 'none' for model.to(device)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)

    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generate-top",
        type=int,
        default=1,
        help="Generate under the top N scored rotations, in addition to identity k=0",
    )
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("rotoformer_results"))
    parser.add_argument("--tag", default="run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.generate_top < 0:
        raise ValueError("--generate-top must be non-negative")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    assert prompt is not None and prompt.strip()

    rotations = parse_rotations(args.rotations, args.prime)
    AutoModelForCausalLM, AutoTokenizer = _load_transformers()

    dtype_map: dict[str, Any] = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype_map[args.dtype], **load_kwargs
        )
    except TypeError:
        # Compatibility with Transformers versions that still use torch_dtype.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype_map[args.dtype], **load_kwargs
        )

    if args.device_map == "none":
        device = _resolve_device(args.device)
        model.to(device)
    model.eval()
    input_device = _model_input_device(model)

    inputs, rendered_prompt = build_prompt_inputs(
        tokenizer,
        prompt=prompt,
        system=args.system,
        raw_prompt=args.raw_prompt,
        device=input_device,
    )
    if inputs["input_ids"].shape[1] < 2:
        raise ValueError("rendered prompt must tokenize to at least two tokens")

    config = RotoformerConfig(
        layer_spec=args.layer_spec,
        scope=args.scope,
        k=0,
        prime=args.prime,
        generator=args.generator,
        translation_mode=args.translation_mode,
        depth_stride=args.depth_stride,
    )

    run_id = f"{args.tag}_{int(time.time())}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = RotoformerAdapter(model, config)
    try:
        print("\nRotoformer attachment:")
        print(json.dumps(adapter.describe(), indent=2), flush=True)
        print(f"\nSweeping {len(rotations)} rotations...", flush=True)
        scores = sweep_rotations(model, adapter, inputs, rotations)
        winner = choose_candidate(scores, args.objective)
        ranked = _ranked(scores, args.objective)

        print("\nTop rotations:")
        for rank, item in enumerate(ranked[: min(10, len(ranked))], start=1):
            print(
                f"{rank:>2}. k={item.k:<5} a={item.multiplier:<4} "
                f"nll={item.prompt_nll:.6f} entropy={item.last_token_entropy:.5f} "
                f"margin={item.top2_margin:.5f}"
            )
        print(f"\nWinner by {args.objective}: k={winner.k}", flush=True)

        generations: list[dict[str, Any]] = []
        if not args.score_only:
            generation_candidates: list[CandidateScore] = []
            identity = next((item for item in scores if item.k == 0), None)
            if identity is None:
                adapter.set_rotation(0)
                identity_score = sweep_rotations(
                    model, adapter, inputs, [0], progress=False
                )[0]
                identity = identity_score
            generation_candidates.append(identity)
            for item in ranked[: args.generate_top]:
                if all(existing.k != item.k for existing in generation_candidates):
                    generation_candidates.append(item)

            print("\nGenerating selected frames...", flush=True)
            for item in generation_candidates:
                adapter.set_rotation(item.k)
                text = generate_text(
                    model,
                    tokenizer,
                    inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                )
                record = {
                    "k": item.k,
                    "multiplier": item.multiplier,
                    "translation": item.translation,
                    "is_identity": item.k == 0,
                    "is_winner": item.k == winner.k,
                    "score": item.to_dict(),
                    "text": text,
                }
                generations.append(record)
                label = "identity" if item.k == 0 else "winner" if item.k == winner.k else "candidate"
                print(f"\n--- {label}: k={item.k} ---\n{text}\n", flush=True)

        score_rows = [item.to_dict() for item in scores]
        _write_csv(output_dir / "scores.csv", score_rows)
        with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
            for row in generations:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = {
            "kind": "rotoformer_mvp_sweep",
            "model": args.model,
            "prompt": prompt,
            "system": args.system,
            "rendered_prompt": rendered_prompt,
            "input_token_count": int(inputs["input_ids"].shape[1]),
            "objective": args.objective,
            "rotations": rotations,
            "adapter": adapter.describe(),
            "winner": winner.to_dict(),
            "scores": score_rows,
            "generations": generations,
            "generation_config": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "prompt.txt").write_text(rendered_prompt, encoding="utf-8")
        print(f"\nWrote results to: {output_dir.resolve()}")
    finally:
        adapter.detach()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
