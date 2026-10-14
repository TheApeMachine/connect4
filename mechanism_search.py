#!/usr/bin/env python3
"""Mechanism Search: a functional-abstraction and transformation experiment.

A frozen LLM is used as a fallible source of candidate knowledge and hypotheses.
No fine-tuning, hidden-state steering, external retrieval, or claims of novelty.
Each model call is a NEW conversation containing only the information deliberately
passed by the search program; no generated chat history or KV cache is reused.

Stages:
  1. Decompose an open problem into functions, constraints, and frictions.
  2. Recall concrete mechanisms from several distinct domains.
  3. Explore structure-preserving transformations using individual mechanisms.
  4. Record hypotheses, preconditions, contradictions, and falsification tests.
  5. Compare with direct idea generation at an equal number of model calls.

The search does NOT prove mechanisms correct, improvements real, or ideas novel.

Requirements: Python >=3.10; for real inference pip install -U torch transformers
accelerate safetensors. Designed for local Hugging Face instruct models.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

DEFAULT_TASK = (
    "Find a new, testable approach that helps a language model discover useful, "
    "mechanistically grounded connections among knowledge already encoded in its weights. "
    "Avoid claiming proven novelty, and respect realistic inference and compute costs."
)
OPERATIONS = ("invert", "transfer", "iterate", "substitute", "factor", "relocate")

# Return concise JSON to leave headroom for full factual mechanisms and falsification.
DECOMPOSE_SCHEMA = {
    "problem": "one sentence",
    "desired_outcomes": ["measurable desired outcome"],
    "constraints": ["non-negotiable constraint"],
    "frictions": [{"name": "short name", "cause": "what creates it", "measurement": "observable metric"}],
    "required_functions": [{
        "name": "specific function name", "inputs": ["input"], "outputs": ["output"],
        "invariant": "property that must remain true", "constraints": ["operating constraint"]
    }],
    "search_domains": ["3 or more diverse domains (not simply synonyms for the target field)"]
}
RECALL_SCHEMA = {"mechanisms": [{
    "name": "established mechanism", "source_domain": "supplied domain",
    "known_basis": "specific known scientific/engineering basis; cite an author or field only when confident",
    "inputs": ["..."], "operation": "physical/computational causal steps, not analogy",
    "outputs": ["..."], "invariants": ["properties this mechanism preserves"],
    "preconditions": ["conditions required to operate"], "failure_modes": ["..."],
    "functional_roles": ["generic but operational role, e.g. local update of dependent outputs"]
}]}
CANDIDATE_SCHEMA = {
    "status": "candidate or rejected",
    "rejection_reason": "for rejected: why the transformation cannot work; otherwise empty string",
    "failed_preconditions": ["for rejected: violated precondition"],
    "invariant_at_risk": "for rejected: invariant that would fail",
    "conflict_test": "for rejected: a test that reveals the incompatibility",
    "title": "short descriptive name", "target_function": "one target function",
    "source_mechanism": "name of mechanism being transformed",
    "operation": "one operation", "abstract_shared_structure": "exact functional relationship",
    "mapping": [{"source_role": "...", "target_role": "...", "justification": "causal reason"}],
    "new_mechanism_steps": ["specific operational step 1", "step 2"],
    "preserved_invariants": [{"invariant": "...", "argument": "why preserved", "verification": "how to test"}],
    "preconditions": ["what must be true for this to work"],
    "frictions_reduced": [{"friction": "...", "mechanism": "why reduced", "measurement": "observable comparison"}],
    "frictions_added": [{"friction": "...", "mechanism": "why introduced", "measurement": "observable comparison"}],
    "new_capability": "what becomes possible, or 'none'",
    "prediction": "specific falsifiable prediction",
    "falsification_test": {"setup": "...", "baseline": "...", "experimental": "...",
                            "metric": "...", "failure_condition": "..."},
    "nearest_known_alternative": "closest conventional mechanism and distinction",
    "uncertainties": ["critical unverified assumptions"]
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def extract_object(text: str) -> dict[str, Any] | None:
    """Find complete JSON objects without splitting on nested braces or code fences."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def present(value: Any) -> bool:
    if isinstance(value, str):
        return len(value.strip()) >= 5
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, dict):
        return bool(value)
    return False


def validate_decomposition(d: Any) -> list[str]:
    if not isinstance(d, dict):
        return ["not an object"]
    missing = [f"missing {key}" for key in DECOMPOSE_SCHEMA if not present(d.get(key))]
    if isinstance(d.get("frictions"), list):
        if not all(isinstance(x, dict) and all(present(x.get(k)) for k in ("name", "cause", "measurement")) for x in d["frictions"]):
            missing.append("frictions lack cause/measurement")
    if isinstance(d.get("required_functions"), list):
        if not all(isinstance(x, dict) and all(present(x.get(k)) for k in ("name", "inputs", "outputs", "invariant", "constraints")) for x in d["required_functions"]):
            missing.append("target functions lack inputs/outputs/invariant")
    return missing


def validate_mechanism(m: Any) -> list[str]:
    if not isinstance(m, dict):
        return ["not an object"]
    return [f"missing {k}" for k in RECALL_SCHEMA["mechanisms"][0] if not present(m.get(k))]


def validate_candidate(c: Any, expected_operation: str | None = None,
                       expected_source: str | None = None) -> list[str]:
    if not isinstance(c, dict):
        return ["not an object"]
    status = c.get("status")
    if status not in ("candidate", "rejected"):
        return ["invalid status (candidate/rejected)"]
    if status == "rejected":
        required = ("operation", "source_mechanism", "rejection_reason",
                    "failed_preconditions", "invariant_at_risk", "conflict_test")
        errors = [f"missing {k}" for k in required if not present(c.get(k))]
    else:
        skip = {"status", "rejection_reason", "failed_preconditions",
                "invariant_at_risk", "conflict_test"}
        errors = [f"missing {k}" for k in CANDIDATE_SCHEMA if k not in skip and not present(c.get(k))]
    if expected_operation and str(c.get("operation", "")).lower() != expected_operation:
        errors.append(f"wrong operation (expected {expected_operation})")
    if expected_source and str(c.get("source_mechanism", "")).casefold().strip() != expected_source.casefold().strip():
        errors.append("source mechanism not preserved")
    if status == "rejected":
        return sorted(set(errors))
    for key, fields in (("mapping", ("source_role", "target_role", "justification")),
                        ("preserved_invariants", ("invariant", "argument", "verification")),
                        ("frictions_reduced", ("friction", "mechanism", "measurement")),
                        ("frictions_added", ("friction", "mechanism", "measurement"))):
        val = c.get(key)
        if not isinstance(val, list) or not all(isinstance(x, dict) and all(present(x.get(f)) for f in fields) for x in val):
            errors.append(f"{key} missing required causal fields")
    test = c.get("falsification_test")
    if not isinstance(test, dict) or not all(present(test.get(k)) for k in CANDIDATE_SCHEMA["falsification_test"]):
        errors.append("falsification_test incomplete")
    return sorted(set(errors))


def idea_signature(candidate: dict[str, Any]) -> str:
    """Lexical duplicate aid ONLY; not a semantic novelty or validity metric."""
    fields = [candidate.get("abstract_shared_structure", ""),
              " ".join(str(v) for v in candidate.get("new_mechanism_steps", []))]
    words = set(re.findall(r"[a-z]{4,}", " ".join(map(str, fields)).lower()))
    return hashlib.sha256(" ".join(sorted(words)).encode()).hexdigest()[:12]


@dataclasses.dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    seconds: float = 0.0


class Backend(Protocol):
    def generate(self, *, system: str, user: str, seed: int, max_new_tokens: int,
                 temperature: float) -> Generation: ...


class HuggingFaceBackend:
    """One-shot chat calls, no cross-call KV state. Inference only; model weights shared."""
    def __init__(self, model_id: str, device: str, dtype: str, loader: str,
                 trust_remote_code: bool = False) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError("Install torch transformers accelerate safetensors") from exc
        self.torch = torch
        self.tfm = transformers
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "mps":
            os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        selected_dtype = torch.float16 if dtype == "auto" and device in ("mps", "cuda") else (
            torch.float32 if dtype == "auto" else getattr(torch, dtype))
        self.processor = None
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code)
        except (OSError, ValueError, TypeError):
            pass
        self.tokenizer = getattr(self.processor, "tokenizer", None) or transformers.AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code)
        classes = (["AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"]
                   if loader in ("auto", "multimodal") else ["AutoModelForCausalLM"])
        errors = []
        self.model = None
        for name in classes:
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            kwargs: dict[str, Any] = {"dtype": selected_dtype,
                "trust_remote_code": trust_remote_code, "low_cpu_mem_usage": True}
            if self.device.type in ("mps", "cuda"):
                kwargs["device_map"] = "mps" if device == "mps" else "auto"
            try:
                self.model = cls.from_pretrained(model_id, **kwargs)
                break
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if self.model is None:
            raise RuntimeError("Cannot load model:\n" + "\n".join(errors[-3:]))
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        print(f"Loaded {model_id} on {self.input_device}", file=sys.stderr)

    def _encode(self, system: str, user: str) -> dict[str, Any]:
        plain = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        mm = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in plain]
        failures: list[str] = []
        for templater, variants in ((self.processor, (mm, plain)), (self.tokenizer, (plain,))):
            if templater is None or not hasattr(templater, "apply_chat_template"):
                continue
            for messages in variants:
                try:
                    result = templater.apply_chat_template(messages, tokenize=True,
                        add_generation_prompt=True, return_dict=True, return_tensors="pt")
                    if hasattr(result, "data"):
                        result = result.data
                    return {k: v.to(self.input_device) for k, v in dict(result).items()
                            if self.torch.is_tensor(v)}
                except Exception as exc:
                    failures.append(f"{type(exc).__name__}: {exc}")
        try:
            rendered = self.tokenizer.apply_chat_template(plain, tokenize=False, add_generation_prompt=True)
        except Exception:
            rendered = f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"
        encoded = self.tokenizer(rendered, return_tensors="pt")
        if "input_ids" not in encoded:
            raise RuntimeError("Could not encode prompt: " + str(failures[-3:]))
        return {k: v.to(self.input_device) for k, v in encoded.items()}

    def generate(self, *, system: str, user: str, seed: int, max_new_tokens: int,
                 temperature: float) -> Generation:
        torch = self.torch
        # New encoded context and new generation call on every invocation.
        inputs = self._encode(system, user)
        if "past_key_values" in inputs:
            raise AssertionError("Fresh calls must not inherit a KV cache")
        torch.manual_seed(seed)
        random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        if self.device.type == "mps":
            torch.mps.manual_seed(seed)
        prompt_len = int(inputs["input_ids"].shape[-1])
        token = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        kwargs: dict[str, Any] = dict(max_new_tokens=max_new_tokens,
            use_cache=True, do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or token)
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=0.95)
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)
        seconds = time.perf_counter() - start
        ids = out.sequences[0] if hasattr(out, "sequences") else out[0]
        generated = ids[prompt_len:]
        return Generation(text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                          prompt_tokens=prompt_len, generated_tokens=len(generated), seconds=seconds)


class MockBackend:
    """Deterministic test backend, never used as evidence of novelty or LLM behavior."""
    def generate(self, *, system: str, user: str, seed: int, max_new_tokens: int,
                 temperature: float) -> Generation:
        del system, max_new_tokens, temperature
        rnd = random.Random(seed)
        def dump(x: Any) -> Generation:
            return Generation(json.dumps(x), len(user) // 4, 175, 0.0)
        if "STAGE: DECOMPOSE" in user:
            return dump({"problem": "reduce wasted validation work", "desired_outcomes": ["fewer rebuilds"],
                "constraints": ["tests retain coverage"],
                "frictions": [{"name": "redundancy", "cause": "recompute unchanged dependencies", "measurement": "total work"}],
                "required_functions": [{"name": "maintain result", "inputs": ["changes"], "outputs": ["updated results"],
                                       "invariant": "correct results", "constraints": ["bounded latency"]}],
                "search_domains": ["database systems", "biology", "manufacturing"]})
        if "STAGE: RECALL" in user:
            dom = user.split("SOURCE DOMAIN:\n", 1)[-1].split("\n", 1)[0]
            return dump({"mechanisms": [{"name": "incremental updater in " + dom,
                "source_domain": dom, "known_basis": "update only changed dependencies",
                "inputs": ["changes"], "operation": "trace dependencies then update affected results",
                "outputs": ["updated values"], "invariants": ["unchanged outputs retained"],
                "preconditions": ["known dependencies"], "failure_modes": ["missing dependency"],
                "functional_roles": ["local dependent update"]}]})
        c = {
            "status": "candidate",
            "title": "Local change propagation " + str(rnd.randrange(100000)),
            "target_function": "maintain result", "source_mechanism": "incremental updater",
            "operation": "direct", "abstract_shared_structure": "update dependent outputs only",
            "mapping": [{"source_role": "dependency", "target_role": "test dependency", "justification": "both propagate change"}],
            "new_mechanism_steps": ["track dependencies", "recompute affected outputs"],
            "preserved_invariants": [{"invariant": "correct results", "argument": "dependents complete", "verification": "full sweep"}],
            "preconditions": ["dependency graph maintained"],
            "frictions_reduced": [{"friction": "redundancy", "mechanism": "skip unaffected work", "measurement": "work saved"}],
            "frictions_added": [{"friction": "graph overhead", "mechanism": "metadata upkeep", "measurement": "bytes stored"}],
            "new_capability": "selectively propagate changes", "prediction": "fewer tests per change",
            "falsification_test": {"setup": "repository corpus", "baseline": "full test run",
                "experimental": "incremental test run", "metric": "coverage and runtime",
                "failure_condition": "missed failures or no speedup"},
            "nearest_known_alternative": "ordinary full test run", "uncertainties": ["dependency completeness"]}
        if "STAGE: TRANSFORM" in user:
            match = re.search(r"OPERATION:\n(\w+)", user)
            op = match.group(1) if match else "transfer"
            name = re.search(r'SOURCE MECHANISM JSON:\n(\{.*?\})\n\n', user, flags=re.S)
            if name:
                val = extract_object(name.group(1))
                if val:
                    c["source_mechanism"] = val["name"]
            c["operation"] = op
        return dump(c)


SYSTEM = ("You are a mechanistic search instrument, not a novelty judge. Reuse real existing "
          "mechanisms; identify causes, inputs/outputs, constraints and measurable effects. "
          "Avoid loose metaphor, jargon and free-floating primitives. State uncertainty and "
          "failure modes explicitly. Output exactly ONE JSON object and no extra text. "
          "Do not assert historical priority or experimental success.")


def prompt_json(stage: str, instruction: str, schema: dict[str, Any]) -> str:
    return f"STAGE: {stage}\n\n{instruction}\n\nRETURN THIS JSON SHAPE with real contents (not placeholders):\n" + json.dumps(schema, indent=2)


def score_target_source(f: dict[str, Any], m: dict[str, Any]) -> float:
    """Soft matching heuristic only; cross-domain items always remain eligible."""
    a = set(re.findall(r"[a-z]{4,}", " ".join(map(str, [f.get("name", ""), f.get("invariant", ""),
                                                              f.get("inputs", []), f.get("outputs", [])])).lower()))
    b = set(re.findall(r"[a-z]{4,}", " ".join(map(str, [m.get("functional_roles", []),
                                                              m.get("invariants", []), m.get("inputs", []), m.get("outputs", [])])).lower()))
    return len(a & b) / max(1, len(a | b))


def select_pairs(problem: dict[str, Any], mechanisms: list[dict[str, Any]],
                 count: int, seed: int) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    functions = problem["required_functions"]
    combinations = [(f, m) for f, m in itertools.product(functions, mechanisms)]
    rng = random.Random(seed)
    rng.shuffle(combinations)
    # Round-robin domains and mechanisms to prioritize coverage over lexical closeness.
    combinations.sort(key=lambda pair: score_target_source(*pair), reverse=True)
    by_domain: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pair in combinations:
        by_domain.setdefault(str(pair[1]["source_domain"]), []).append(pair)
    domains = list(by_domain)
    rng.shuffle(domains)
    if not domains:
        return []
    result = []
    for j in range(count):
        # Ensure each domain gets considered, then cycle through its alternatives.
        domain = domains[j % len(domains)]
        candidates = by_domain[domain]
        pair = candidates[(j // len(domains)) % len(candidates)]
        result.append((*pair, OPERATIONS[j % len(OPERATIONS)]))
    return result


@dataclasses.dataclass
class Trial:
    stage: str
    seed: int
    prompt_tokens: int
    output_tokens: int
    seconds: float
    raw: str
    parsed: dict[str, Any] | None
    errors: list[str]
    details: dict[str, Any]


class Experiment:
    def __init__(self, args: argparse.Namespace, backend: Backend) -> None:
        self.args = args
        self.backend = backend
        self.out = Path(args.out)
        if self.out.exists() and any(self.out.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output directory already has files: {self.out} (use --overwrite or a new path)")
        self.out.mkdir(parents=True, exist_ok=True)
        for filename in ("calls.jsonl", "mechanisms.jsonl", "candidates.jsonl", "baseline.jsonl"):
            (self.out / filename).write_text("", encoding="utf-8")
        self.call_index = 0
        self.calls: list[Trial] = []
        self.candidates: list[dict[str, Any]] = []
        self.direct: list[dict[str, Any]] = []
        self.frontier: list[dict[str, Any]] = []

    def ask(self, stage: str, instruction: str, schema: dict[str, Any],
            validate: Any, max_new_tokens: int, details: dict[str, Any] | None = None,
            expected_op: str | None = None, expected_src: str | None = None,
            seed_override: int | None = None) -> Trial:
        seed = self.args.seed + self.call_index * 104729 if seed_override is None else seed_override
        self.call_index += 1
        user = prompt_json(stage, instruction, schema)
        gen = self.backend.generate(system=SYSTEM, user=user, seed=seed,
                                    max_new_tokens=max_new_tokens, temperature=self.args.temperature)
        parsed = extract_object(gen.text)
        if validate is validate_candidate:
            errors = validate_candidate(parsed, expected_op, expected_src)
        else:
            errors = validate(parsed)
        trial = Trial(stage, seed, gen.prompt_tokens, gen.generated_tokens, gen.seconds,
                      gen.text, parsed, errors, details or {})
        self.calls.append(trial)
        append_jsonl(self.out / "calls.jsonl", dataclasses.asdict(trial))
        flag = ("INCOMPATIBLE" if isinstance(parsed, dict) and parsed.get("status") == "rejected" and not errors
                else "OK" if not errors else "INVALID: " + ", ".join(errors[:3]))
        print(f"{stage:10s} | call={self.call_index:2d} seed={seed:9d} | "
              f"{gen.generated_tokens:4d} tokens | {flag}", flush=True)
        return trial

    def execute(self) -> dict[str, Any]:
        args = self.args
        decomp = self.ask("DECOMPOSE", f"TASK:\n{args.task}\n\n"
            "Extract required functional roles, invariants, constraints, and existing frictions. "
            "Do not propose an invention yet. Choose unrelated but mechanistically promising domains.",
            DECOMPOSE_SCHEMA, validate_decomposition, args.analysis_tokens)
        if decomp.errors:
            raise RuntimeError("Invalid task decomposition; see calls.jsonl; increase --analysis-tokens")
        problem = decomp.parsed
        assert problem is not None
        write_json(self.out / "problem.json", problem)
        suggested = [x for x in problem.get("search_domains", []) if isinstance(x, str) and x.strip()]
        domains = ([x.strip() for x in args.domains.split(",") if x.strip()] if args.domains else suggested)
        domains = list(dict.fromkeys(domains))[:args.domain_count]
        if len(domains) < 1:
            raise RuntimeError("No source domains; supply --domains")
        mechanisms: list[dict[str, Any]] = []
        for domain in domains:
            recall = self.ask("RECALL", f"ORIGINAL PROBLEM:\n{args.task}\n\nSOURCE DOMAIN:\n{domain}\n\n"
                f"Recall up to {args.mechanisms_per_domain} REAL, recognizable mechanisms from that domain. "
                "Describe how each works independently of the problem. Don't propose an application. "
                "Do not invent citations or assert that a mechanism is novel.", RECALL_SCHEMA,
                lambda x: (["mechanisms list missing"] if not isinstance(x, dict) or
                           not isinstance(x.get("mechanisms"), list) or not x["mechanisms"] else []),
                args.recall_tokens, {"domain": domain})
            if recall.errors or recall.parsed is None:
                continue
            for item in recall.parsed.get("mechanisms", [])[:args.mechanisms_per_domain]:
                errors = validate_mechanism(item)
                if errors:
                    append_jsonl(self.out / "mechanisms.jsonl", {"domain": domain, "valid": False,
                        "errors": errors, "mechanism": item})
                    continue
                item["source_domain"] = domain
                mechanisms.append(item)
                append_jsonl(self.out / "mechanisms.jsonl", {"domain": domain,
                    "valid": True, "mechanism": item})
        if not mechanisms:
            raise RuntimeError("No valid mechanisms recalled; examine mechanisms.jsonl")
        pairs = select_pairs(problem, mechanisms, args.ideas, args.seed + 17)
        for j, (initial_func, initial_source, initial_operation) in enumerate(pairs):
            func, source, operation = initial_func, initial_source, initial_operation
            parent: dict[str, Any] | None = None
            # Branch the search using newly exposed friction rather than only walking a static grid.
            if args.adaptive and self.frontier and j >= len(domains):
                parent = self.frontier.pop(0)
                alternatives = [m for m in mechanisms if m["source_domain"] != parent["source_domain"]]
                if alternatives:
                    added = parent["candidate"].get("frictions_added", [])
                    friction_names = " ".join(str(x.get("friction", "")) for x in added if isinstance(x, dict))
                    revised_need = dict(func)
                    revised_need["name"] = "reduce " + friction_names
                    source = max(alternatives, key=lambda m: score_target_source(revised_need, m))
                    operation = "substitute" if j % 2 else "factor"

            op_descriptions = {
                "invert": "Reverse a directional role or process while retaining its useful constraints.",
                "transfer": "Reimplement the same causal interaction in the target's substrate.",
                "iterate": "Feed an operation's output back as its next input; analyze stability/growth.",
                "substitute": "Replace an incumbent mechanism without violating its required invariant.",
                "factor": "Separate shared and local work so only necessary parts are recomputed.",
                "relocate": "Move a decision or operation earlier/later or closer to the information source."
            }
            parent_context = ""
            if parent:
                parent_context = ("PARENT RESEARCH HYPOTHESIS (not established):\n"
                                  + json.dumps(parent["candidate"], ensure_ascii=False)
                                  + "\n\nRESEARCH NEXT: Find a different mechanism which specifically "
                                  "addresses one of the parent's ADDED frictions. Preserve the "
                                  "original required function and test any new tradeoff.\n\n")
            instruction = (f"ORIGINAL PROBLEM:\n{args.task}\n\nPROBLEM DECOMPOSITION JSON:\n"
                f"{json.dumps(problem, ensure_ascii=False)}\n\n"
                f"{parent_context}TARGET FUNCTION JSON:\n{json.dumps(func, ensure_ascii=False)}\n\n"
                f"SOURCE MECHANISM JSON:\n{json.dumps(source, ensure_ascii=False)}\n\n"
                f"OPERATION:\n{operation}\n{op_descriptions[operation]}\n\n"
                "Find a function-preserving, mechanically specific application of the source to the target. "
                "Show concrete causal mappings; identify preconditions and what breaks; "
                "include a discriminating, realistic falsifier. Do not force a connection: "
                "if infeasible, return status=rejected with the violated precondition, "
                "invariant at risk and a conflict test. Do not invent a viable proposal. "
                "Reuse existing primitives. Every claim is a hypothesis, not established novelty.")
            trial = self.ask("TRANSFORM", instruction, CANDIDATE_SCHEMA,
                validate_candidate, args.candidate_tokens,
                {"index": j, "domain": source["source_domain"], "operation": operation,
                 "source": source["name"], "target": func["name"],
                 "parent": parent["index"] if parent else None},
                expected_op=operation, expected_src=source["name"])
            record = {"index": j,
                      "valid": not trial.errors and isinstance(trial.parsed, dict)
                               and trial.parsed.get("status") == "candidate",
                      "rejected": not trial.errors and isinstance(trial.parsed, dict)
                                  and trial.parsed.get("status") == "rejected",
                      "errors": trial.errors, "parent": parent["index"] if parent else None,
                      "candidate": trial.parsed, "source_domain": source["source_domain"],
                      "operation": operation, "call": self.call_index,
                      "signature": idea_signature(trial.parsed) if isinstance(trial.parsed, dict) else None}
            self.candidates.append(record)
            append_jsonl(self.out / "candidates.jsonl", record)
            if record["valid"] and isinstance(trial.parsed, dict):
                self.frontier.append({"index": j, "source_domain": source["source_domain"],
                                      "candidate": trial.parsed})
        search_calls = self.call_index
        if not args.skip_baseline:
            # Pair underlying sampling seeds with the search calls, prioritizing
            # actual idea-producing calls. This is a call-budget control, not a
            # claim that prompts have matching FLOPs or equivalent token length.
            seed_plan = ([x.seed for x in self.calls if x.stage == "TRANSFORM"]
                         + [x.seed for x in self.calls if x.stage != "TRANSFORM"])
            for j in range(search_calls):
                # Conservative baseline: same total model calls, each may create an idea.
                trial = self.ask("DIRECT", f"ORIGINAL PROBLEM:\n{args.task}\n\n"
                    "Propose ONE technically concrete, falsifiable improvement. Use recognizable existing "
                    "ingredients, explain the actual mechanism and preserved properties, evaluate new "
                    "frictions as well as benefits, and give the nearest alternative. No claim of proven novelty. "
                    "Do not use previous outputs; this is an independent fresh session. "
                    "For 'operation', use the literal string 'direct'.",
                    CANDIDATE_SCHEMA, validate_candidate, args.candidate_tokens, {"index": j},
                    expected_op="direct", seed_override=seed_plan[j])
                row = {"index": j,
                       "valid": not trial.errors and isinstance(trial.parsed, dict)
                                and trial.parsed.get("status") == "candidate",
                       "rejected": not trial.errors and isinstance(trial.parsed, dict)
                                   and trial.parsed.get("status") == "rejected",
                       "errors": trial.errors,
                       "candidate": trial.parsed,
                       "signature": idea_signature(trial.parsed) if isinstance(trial.parsed, dict) else None}
                self.direct.append(row)
                append_jsonl(self.out / "baseline.jsonl", row)
        summary = {
            "experiment": "functional abstraction + mechanistic transformation",
            "frozen_weights": True, "cross_call_chat_memory": False,
            "generation_calls_search": search_calls,
            "generation_calls_baseline": len(self.direct),
            "baseline_seed_matching": "candidate seeds paired first, other search-call seeds second",
            "tokens_generated_search": sum(x.output_tokens for x in self.calls[:search_calls]),
            "tokens_generated_baseline": sum(x.output_tokens for x in self.calls[search_calls:]),
            "prompt_tokens_search": sum(x.prompt_tokens for x in self.calls[:search_calls]),
            "prompt_tokens_baseline": sum(x.prompt_tokens for x in self.calls[search_calls:]),
            "valid_mechanisms": len(mechanisms),
            "structured_candidates_search": sum(row["valid"] for row in self.candidates),
            "mechanically_rejected_search": sum(row["rejected"] for row in self.candidates),
            "structured_candidates_baseline": sum(row["valid"] for row in self.direct),
            "mechanically_rejected_baseline": sum(row["rejected"] for row in self.direct),
            "search_lexical_signatures": len({x["signature"] for x in self.candidates if x["valid"]}),
            "baseline_lexical_signatures": len({x["signature"] for x in self.direct if x["valid"]}),
            "transforms": dict(Counter(x["operation"] for x in self.candidates)),
            "limits": ["LLM-recalled facts may be inaccurate", "schema completeness is not validity",
                       "lexical signatures do not establish semantic novelty",
                       "no results independently verified", "matching calls is not matching token FLOPs"]
        }
        write_json(self.out / "summary.json", summary)
        write_json(self.out / "config.json", vars(args))
        self.write_markdown(problem, mechanisms, summary)
        return summary

    def write_markdown(self, problem: dict[str, Any], mechanisms: list[dict[str, Any]],
                       summary: dict[str, Any]) -> None:
        md = ["# Mechanism Search — Hypotheses (NOT validated discoveries)", "",
              "All facts are model-recalled and unverified. Output formatting and a falsifier do not certify scientific merit.",
              "", "## Original problem", "", self.args.task, "", "## Decomposition", "",
              "```json", json.dumps(problem, ensure_ascii=False, indent=2), "```", "",
              "## Source mechanisms", ""]
        for m in mechanisms:
            md.extend([f"### {m['name']} ({m['source_domain']})", "", m["operation"], ""])
        for heading, rows in (("Transformed hypotheses", self.candidates),
                              ("Direct-generation control", self.direct)):
            md.extend([f"## {heading}", ""])
            for record in rows:
                c = record.get("candidate")
                md.extend([f"### {record['index'] + 1}. {c.get('title', 'Unparseable output') if isinstance(c, dict) else 'Unparseable output'}",
                           "", f"Schema: {'PASS' if record['valid'] else 'REJECT'}", ""])
                if record["errors"]:
                    md.extend(["Missing/invalid fields: " + "; ".join(record["errors"]), ""])
                if record.get("rejected"):
                    md.extend(["The transformation is **rejected** as incompatible, not counted as a discovery.", ""])
                md.extend(["```json", json.dumps(c, ensure_ascii=False, indent=2), "```", ""])
        md.extend(["## Machine diagnostics (not discovery scores)", "", "```json",
                   json.dumps(summary, indent=2), "```", ""])
        (self.out / "ideas.md").write_text("\n".join(md), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Search for mechanism transformations across unrelated domains")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--loader", choices=("auto", "multimodal", "causal"), default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--domains", help="Optional comma-separated domains; otherwise model selects domains")
    p.add_argument("--domain-count", type=int, default=3)
    p.add_argument("--mechanisms-per-domain", type=int, default=2)
    p.add_argument("--ideas", type=int, default=6)
    p.add_argument("--analysis-tokens", type=int, default=1000)
    p.add_argument("--recall-tokens", type=int, default=1000)
    p.add_argument("--candidate-tokens", type=int, default=1600)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--no-adaptive", action="store_false", dest="adaptive",
                   help="Disable expanding promising candidates via their newly added frictions")
    p.set_defaults(adaptive=True)
    p.add_argument("--mock", action="store_true", help="Run synthetic contract test; not scientific evidence")
    p.add_argument("--out", default="runs/mechanism_search")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.domain_count < 1 or args.mechanisms_per_domain < 1 or args.ideas < 1:
        raise SystemExit("--domain-count, --mechanisms-per-domain, --ideas must be >= 1")
    backend: Backend = MockBackend() if args.mock else HuggingFaceBackend(
        args.model, args.device, args.dtype, args.loader, args.trust_remote_code)
    result = Experiment(args, backend).execute()
    print("\n" + json.dumps(result, indent=2))
    print(f"\nWrote {args.out}/ideas.md plus raw logs and JSON files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
