#!/usr/bin/env python3
"""Multi-frame Mechanism Search: competing and revisable functional abstractions.

A frozen LLM is used as a fallible source of candidate knowledge and hypotheses.
No fine-tuning, hidden-state steering, external retrieval, or claims of novelty.
Each model call is a NEW conversation containing only the information deliberately
passed by the search program; no generated chat history or KV cache is reused.

Stages:
  1. Independently generate competing representations of the original problem.
  2. Recall mechanisms for frames while keeping user constraints immutable.
  3. Search transformations across frame/mechanism pairs.
  4. Reframe exposed friction or failed preconditions and expand the frontier.
  5. Compare against single-frame or direct generation at equal call budgets.

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



# Keep the original data contracts and model loader, but avoid reusing the old
# single-DECOMPOSE/RECALL/TRANSFORM orchestration.
FRAME_SCHEMA = {
    **{k:v for k,v in DECOMPOSE_SCHEMA.items() if k != "constraints"},
    "assumed_operating_conditions": ["testable working condition, not a user requirement"],
    "frame_name": "short, distinctive name for this way to model the problem",
    "problem_interpretation": "a representation of the underlying functional problem, NOT a solution",
    "state_variables": ["observable or operational variable in this representation"],
    "causal_relations": [{"source": "variable or function", "target": "variable or function",
                          "mechanism": "what causes this relationship"}],
    "assumptions": ["tentative framing assumption, NOT a requirement supplied by the user"],
    "blind_spots": ["what this representation may fail to reveal"]
}


def validate_frame(frame: Any) -> list[str]:
    if not isinstance(frame, dict):
        return ["not an object"]
    errors = [f"missing {key}" for key in FRAME_SCHEMA if not present(frame.get(key))]
    for key in ("desired_outcomes", "frictions", "required_functions", "search_domains",
                "state_variables", "causal_relations", "assumptions", "blind_spots",
                "assumed_operating_conditions"):
        if not isinstance(frame.get(key), list) or not frame[key]:
            errors.append(f"{key} must be a nonempty list")
    if isinstance(frame.get("frictions"), list) and not all(
        isinstance(x, dict) and all(present(x.get(k)) for k in ("name", "cause", "measurement"))
        for x in frame["frictions"]
    ):
        errors.append("frictions missing cause/measurement")
    if isinstance(frame.get("required_functions"), list) and not all(
        isinstance(x, dict) and all(present(x.get(k)) for k in ("name", "inputs", "outputs", "invariant", "constraints"))
        for x in frame["required_functions"]
    ):
        errors.append("required_functions incomplete")
    for key in ("frame_name", "problem_interpretation", "state_variables", "causal_relations",
                "assumptions", "blind_spots"):
        if not present(frame.get(key)):
            errors.append(f"missing {key}")
    relations = frame.get("causal_relations")
    if not isinstance(relations, list) or not all(
        isinstance(r, dict) and all(present(r.get(k)) for k in ("source", "target", "mechanism"))
        for r in relations
    ):
        errors.append("causal_relations missing source/target/mechanism")
    return errors


def _json_candidates(text: str) -> list[str]:
    """Find top-level brace spans, respecting quoted strings and escaped quotes.

    A malformed backslash does not interfere with brace balancing. We don't
    retrieve arbitrary nested JSON fragments from the middle of a broken root.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    quote = False
    escaped = False
    for i, ch in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quote = False
        elif ch == '"':
            quote = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start:i+1])
    return spans


def _escape_invalid_json_backslashes(text: str) -> tuple[str, int]:
    """Repair only INVALID JSON backslash escapes inside strings, never guesses.

    Legal escapes such as '\\t' are ambiguous with a raw LaTeX \\theta; they
    remain untouched and prompts request ASCII/escaped LaTeX to avoid ambiguity.
    Returns repair count for provenance; does not claim semantic equivalence.
    """
    out: list[str] = []
    quote = False
    repairs = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            # This quote may be escaped by a run of preceding backslashes.
            preceding = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                preceding += 1
                j -= 1
            if preceding % 2 == 0:
                quote = not quote
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and quote:
            nxt = text[i+1] if i + 1 < len(text) else ""
            if nxt not in '"\\/bfnrtu' or (nxt == 'u' and
                not re.fullmatch(r"[0-9a-fA-F]{4}", text[i+2:i+6])):
                out.append("\\\\")
                repairs += 1
            else:
                out.append("\\")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), repairs


def parse_root_object(text: str, keys: tuple[str, ...] = ()) -> tuple[dict[str, Any] | None, str]:
    """Parse a complete top-level response, with auditable conservative repair.

    If a root cannot be parsed, return None rather than silently accepting a
    nested mapping as a complete candidate.
    """
    for candidate in _json_candidates(text):
        for version, mode in ((candidate, "exact"), (*_escape_invalid_json_backslashes(candidate),)):
            # The second tuple is (fixed_text, number_of_repairs).
            label = mode if isinstance(mode, str) else f"invalid-escape-repair:{mode}"
            try:
                root = json.loads(version)
            except (ValueError, TypeError):
                continue
            if isinstance(root, dict) and (not keys or any(k in root for k in keys)):
                return root, label
    return None, "unparseable-root"


class MockBackend:
    """Synthetic mechanics backend; NEVER evidence of discovery or model behavior."""
    def __init__(self) -> None:
        self.requests: list[tuple[str, int]] = []

    def generate(self, *, system: str, user: str, seed: int, max_new_tokens: int,
                 temperature: float) -> Generation:
        del system, max_new_tokens, temperature
        stage = user.split("STAGE: ", 1)[1].split("\n", 1)[0]
        self.requests.append((user, seed))
        def emit(value: Any) -> Generation:
            return Generation(json.dumps(value), len(user)//4, 175, 0.0)
        if stage in ("FRAME", "REFRAME"):
            name = ("reframed" if stage == "REFRAME" else "independent") + f"-{seed}"
            return emit({"frame_name": name, "problem_interpretation": "local changes to dependent results",
                "problem": "reduce redundant validation effort", "desired_outcomes": ["lower recomputation cost"],
                "assumed_operating_conditions": ["propagation has bounded runtime"],
                "frictions": [{"name": "redundant work", "cause": "reprocessing unaffected outputs",
                               "measurement": "work per change"}],
                "required_functions": [{"name": "maintain dependent results", "inputs": ["changed input"],
                    "outputs": ["updated result"], "invariant": "complete outputs",
                    "constraints": ["all dependencies considered"]}],
                "search_domains": ["database systems", "manufacturing", "biology"],
                "state_variables": ["changed inputs", "dependent outputs"],
                "causal_relations": [{"source": "change", "target": "recalculation",
                    "mechanism": "dependency propagation"}],
                "assumptions": ["dependencies may be tracked"],
                "blind_spots": ["hidden dependencies"]})
        if stage == "RECALL":
            domain = user.split("SOURCE DOMAIN:\n", 1)[1].split("\n", 1)[0]
            return emit({"mechanisms": [{"name": "incremental view maintenance in " + domain,
                "source_domain": domain, "known_basis": "recompute affected dependents only",
                "inputs": ["changed records"], "operation": "traverse dependencies and update impacted results",
                "outputs": ["updated materialized view"], "invariants": ["view consistency"],
                "preconditions": ["dependencies are complete"], "failure_modes": ["missing edges"],
                "functional_roles": ["local update of dependent outputs"]}]})
        candidate = {"status": "candidate", "rejection_reason": "", "failed_preconditions": [],
            "invariant_at_risk": "", "conflict_test": "",
            "title": "Selective dependency propagation " + str(seed),
            "target_function": "maintain dependent results", "source_mechanism": "incremental view maintenance",
            "operation": "direct", "abstract_shared_structure": "local propagation of changed dependencies",
            "mapping": [{"source_role": "view dependency", "target_role": "test dependency",
                         "justification": "changes propagate through explicit dependency edges"}],
            "new_mechanism_steps": ["track dependencies", "recompute only affected results"],
            "preserved_invariants": [{"invariant": "complete results", "argument": "every dependency is checked",
                                      "verification": "compare with full recompute"}],
            "preconditions": ["dependency graph complete"],
            "frictions_reduced": [{"friction": "redundant work", "mechanism": "skip unaffected entries",
                                    "measurement": "work saved"}],
            "frictions_added": [{"friction": "graph storage overhead", "mechanism": "metadata maintenance",
                                  "measurement": "bytes stored"}],
            "new_capability": "incremental recomputation", "prediction": "reduced total processing time",
            "falsification_test": {"setup": "representative repository", "baseline": "full test suite",
                                   "experimental": "incremental test run", "metric": "coverage and runtime",
                                   "failure_condition": "missed failures or no speedup"},
            "nearest_known_alternative": "ordinary full test run",
            "uncertainties": ["unknown dependencies"]}
        if stage == "TRANSFORM":
            op = user.split("OPERATION:\n", 1)[1].split("\n", 1)[0]
            source_text = user.split("SOURCE MECHANISM JSON:\n", 1)[1].split("\n\n", 1)[0]
            source, _ = parse_root_object(source_text, ("name",))
            candidate["source_mechanism"] = source["name"] if source else "unknown"
            candidate["operation"] = op
        return emit(candidate)


SYSTEM = ("You are a mechanistic research-search instrument, not a novelty judge. "
          "Use known mechanisms, identify exact causes, prerequisites, invariants and potential conflicts. "
          "Do not claim a result has been experimentally established. "
          "Do not treat speculative assumptions as user-specified constraints. "
          "Output ONE complete JSON object only. Use plain ASCII mathematics inside JSON, "
          "or double-escape backslashes; do not emit raw LaTeX escapes.")


def prompt_json(stage: str, instruction: str, schema: dict[str, Any]) -> str:
    return f"STAGE: {stage}\n\n{instruction}\n\nRETURN THIS JSON SHAPE (not placeholder text):\n" + json.dumps(schema, indent=2)


def words(text: Any) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", str(text).lower()))


def pair_score(f: dict[str, Any], m: dict[str, Any]) -> float:
    a = words([f.get("name"), f.get("invariant"), f.get("inputs"), f.get("outputs")])
    b = words([m.get("functional_roles"), m.get("invariants"), m.get("inputs"), m.get("outputs")])
    return len(a & b)/max(1, len(a|b))


@dataclasses.dataclass
class Trial:
    stage: str
    seed: int
    prompt_tokens: int
    output_tokens: int
    seconds: float
    raw: str
    parsed: dict[str, Any] | None
    parse_mode: str
    errors: list[str]
    details: dict[str, Any]


@dataclasses.dataclass
class Frame:
    id: str
    parent_id: str | None
    origin: str
    data: dict[str, Any]
    attempts: int = 0
    created_at: int = 0
    max_prior_frame_overlap: float = 0.0  # lexical diagnostic, not semantic identity


@dataclasses.dataclass
class SearchNode:
    id: str
    kind: str  # frame, candidate, failed_constraint, friction, exhausted
    frame_id: str
    parent_id: str | None
    payload: dict[str, Any]


class MultiFrameExperiment:
    def __init__(self, args: argparse.Namespace, backend: Backend) -> None:
        self.args, self.backend = args, backend
        self.out = Path(args.out)
        if self.out.exists() and any(self.out.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Nonempty directory: {self.out}. Use a different --out or --overwrite.")
        self.out.mkdir(parents=True, exist_ok=True)
        for fn in ("calls.jsonl", "frames.jsonl", "mechanisms.jsonl", "frontier.jsonl",
                   "candidates.jsonl", "baseline.jsonl"):
            (self.out/fn).write_text("", encoding="utf-8")
        self.calls: list[Trial] = []
        self.frames: list[Frame] = []
        self.nodes: list[SearchNode] = []
        self.mechanisms: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        self.baseline: list[dict[str, Any]] = []
        self.signal_queue: list[str] = []
        self.pair_usage: Counter = Counter()
        self.reframes = 0
        self.search_calls = 0
        self.search_seeds: list[int] = []

    def _requirements(self) -> str:
        return (f"ORIGINAL USER TASK (never replace):\n{self.args.task}\n\n"
                "USER-IMPOSED CONSTRAINTS (immutable, may be empty):\n"
                + json.dumps(self.args.constraint, ensure_ascii=False) + "\n\n"
                "You may add TESTABLE ASSUMPTIONS but never portray them as requirements set by the user.\n")

    def ask(self, stage: str, instruction: str, schema: dict[str, Any],
            validator: Any, max_new_tokens: int, details: dict[str, Any] | None = None,
            expected_op: str | None = None, expected_source: str | None = None,
            seed_override: int | None = None, in_search: bool = True) -> Trial:
        index = len(self.calls)
        seed = self.args.seed + self.search_calls*104729 if seed_override is None else seed_override
        if in_search and self.search_calls >= self.args.search_calls:
            raise RuntimeError("search call budget exceeded")
        if in_search:
            self.search_calls += 1
            self.search_seeds.append(seed)
        prompt = prompt_json(stage, self._requirements() + "\n" + instruction, schema)
        gen = self.backend.generate(system=SYSTEM, user=prompt, seed=seed,
                                    max_new_tokens=max_new_tokens,
                                    temperature=self.args.temperature)
        preferred = (("status", "operation") if validator is validate_candidate else
                     ("mechanisms",) if stage == "RECALL" else ("frame_name",))
        parsed, mode = parse_root_object(gen.text, preferred)
        if validator is validate_candidate:
            errors = validator(parsed, expected_op, expected_source)
        else:
            errors = validator(parsed)
        trial = Trial(stage, seed, gen.prompt_tokens, gen.generated_tokens,
                      gen.seconds, gen.text, parsed, mode, errors, details or {})
        self.calls.append(trial)
        append_jsonl(self.out/"calls.jsonl", dataclasses.asdict(trial))
        flag = ("INVALID: " + ", ".join(errors[:2]) if errors else
                "INCOMPATIBLE" if isinstance(parsed, dict) and parsed.get("status") == "rejected" else "OK")
        print(f"{stage:9s} | call={index+1:2d} seed={seed:9d} | {gen.generated_tokens:4d} tokens | {flag}",
              flush=True)
        return trial

    def add_node(self, kind: str, frame_id: str, parent_id: str | None,
                 payload: dict[str, Any]) -> SearchNode:
        node = SearchNode(f"n{len(self.nodes)}", kind, frame_id, parent_id, payload)
        self.nodes.append(node)
        append_jsonl(self.out/"frontier.jsonl", dataclasses.asdict(node))
        return node

    def add_frame(self, parsed: dict[str, Any], parent_id: str | None, origin: str) -> Frame:
        def signature(data: dict[str, Any]) -> set[str]:
            return words([data.get("problem_interpretation"), data.get("state_variables"),
                          data.get("causal_relations"), data.get("required_functions")])
        now = signature(parsed)
        overlap = max((len(now & signature(f.data))/max(1, len(now | signature(f.data)))
                       for f in self.frames), default=0.0)
        frame = Frame(f"f{len(self.frames)}", parent_id, origin, parsed,
                      created_at=len(self.calls), max_prior_frame_overlap=round(overlap, 5))
        self.frames.append(frame)
        append_jsonl(self.out/"frames.jsonl", dataclasses.asdict(frame))
        self.add_node("frame", frame.id, parent_id,
                      {"frame_name": parsed["frame_name"], "origin": origin})
        return frame

    def _frame_instruction(self, index: int) -> str:
        return (f"FRAME SLOT: {index}\n"
                "Independently formulate ONE functional/causal representation of the original problem. "
                "Do not propose a solution or an implementation. The framing must include state variables, "
                "causal dependencies, required functions, measurable frictions, working assumptions, "
                "and blind spots. Search for mechanisms by function rather than superficial domain words. "
                "Your output is independent: you have no access to any other frame or earlier generated idea. "
                "Treat all added restrictions as optional ASSUMPTIONS, not immutable requirements. "
                "Do not require the solution to involve model activations, vector spaces or any other "
                "particular technical approach unless the original user task requires it.")

    def recall(self, frame: Frame, domain: str) -> None:
        if self.search_calls >= self.args.search_calls:
            return
        instruction = (f"SOURCE DOMAIN:\n{domain}\n\n"
                       f"FUNCTIONAL NEEDS (not a proposed solution):\n"
                       f"{json.dumps(frame.data['required_functions'], ensure_ascii=False)}\n\n"
                       f"Recall up to {self.args.mechanisms_per_domain} real, recognizable mechanisms "
                       "from the named domain. Describe each independently of the proposed target "
                       "solution; give inputs, outputs, causal operations, invariants and failure modes. "
                       "Do not invent papers or treat analogies as equivalent mechanisms.")
        def validate_recall(v: Any) -> list[str]:
            return [] if isinstance(v, dict) and isinstance(v.get("mechanisms"), list) and v["mechanisms"] else ["missing mechanisms"]
        r = self.ask("RECALL", instruction, RECALL_SCHEMA, validate_recall,
                     self.args.recall_tokens, {"frame_id": frame.id, "domain": domain})
        if r.errors or r.parsed is None:
            return
        for m in r.parsed["mechanisms"][:self.args.mechanisms_per_domain]:
            issues = validate_mechanism(m)
            record = {"frame_id": frame.id, "domain": domain, "valid": not issues,
                      "errors": issues, "mechanism": m}
            append_jsonl(self.out/"mechanisms.jsonl", record)
            if not issues:
                m = dict(m)
                m["source_domain"] = domain
                m["_origin_frame"] = frame.id
                self.mechanisms.append(m)

    def initial_frames(self) -> None:
        n = 1 if self.args.mode == "single" else self.args.frames
        for i in range(n):
            trial = self.ask("FRAME", self._frame_instruction(i), FRAME_SCHEMA,
                             validate_frame, self.args.analysis_tokens, {"slot": i})
            if not trial.errors and trial.parsed:
                self.add_frame(trial.parsed, None, "independent")
        if not self.frames:
            raise RuntimeError("No valid frames were generated; inspect calls.jsonl")
        # Mechanisms are recalled from each frame separately. Each domain selection
        # uses only the independent frame and optional explicit user domain override.
        recalled_domains: set[str] = set()
        for f in list(self.frames):
            explicit = [v.strip() for v in self.args.domains.split(",") if v.strip()] if self.args.domains else []
            domains = list(dict.fromkeys(explicit or [s for s in f.data["search_domains"] if isinstance(s, str)]))
            # Prefer different source domains across independently generated
            # frames, without inventing replacement domains absent from the frame.
            domains = [d for d in domains if d not in recalled_domains] + [d for d in domains if d in recalled_domains]
            n_domain = self.args.single_domains if self.args.mode == "single" else self.args.domains_per_frame
            for dom in domains[:n_domain]:
                if self.search_calls >= self.args.search_calls - 1:
                    break
                self.recall(f, dom)
                recalled_domains.add(dom)
        if not self.mechanisms:
            raise RuntimeError("No valid recalled mechanisms; inspect mechanisms.jsonl")

    def _next_signal(self) -> SearchNode | None:
        while self.signal_queue:
            nid = self.signal_queue.pop(0)
            node = next((x for x in self.nodes if x.id == nid), None)
            if node:
                return node
        # If we've exhausted the existing frames, permit a new viewpoint even
        # without an identified failure. This is explicitly labeled 'exhausted'.
        tried = [f for f in self.frames if f.attempts >= 2]
        if tried:
            old = max(tried, key=lambda f: f.attempts)
            return self.add_node("exhausted", old.id, None,
                                 {"reason": "frame has multiple explored mechanisms; seek an unrelated functional representation"})
        return None

    def maybe_reframe(self, candidate_index: int) -> Frame | None:
        if self.args.mode == "single" or self.reframes >= self.args.max_reframes:
            return None
        # Reframe only after evidence from actual transformations exists. Budget
        # reserves a REFRAME, one RECALL, and at least one later TRANSFORM call.
        if candidate_index < len([f for f in self.frames if f.origin == "independent"]):
            return None
        if self.search_calls + 3 > self.args.search_calls:
            return None
        signal = self._next_signal()
        if signal is None:
            return None
        parent = next(f for f in self.frames if f.id == signal.frame_id)
        instr = (f"PARENT FRAME (a fallible hypothesis, not a mandate):\n"
                 f"{json.dumps(parent.data, ensure_ascii=False)}\n\n"
                 f"TRIGGER KIND: {signal.kind}\n"
                 f"TRIGGER EVIDENCE:\n{json.dumps(signal.payload, ensure_ascii=False)}\n\n"
                 "Construct a NEW functional representation of the ORIGINAL user problem which can "
                 "make progress on this failure, introduced friction, or exhausted branch. "
                 "Change the state variables, causal decomposition or problem boundary; "
                 "do not just rename the previous frame, switch jargon, or solve the trigger alone. "
                 "Preserve the original user goals, with user constraints unchanged. "
                 "Treat all new restrictions as testable assumptions, not user demands. "
                 "Do not propose a solution yet. State the new frame's likely blind spots.")
        r = self.ask("REFRAME", instr, FRAME_SCHEMA, validate_frame,
                     self.args.analysis_tokens,
                     {"trigger_node": signal.id, "parent_frame": parent.id})
        self.reframes += 1
        if r.errors or not r.parsed:
            return None
        new = self.add_frame(r.parsed, parent.id, f"reframed-from-{signal.kind}")
        domains = [d for d in r.parsed.get("search_domains", []) if isinstance(d, str)]
        unused = [d for d in domains if d not in {m["source_domain"] for m in self.mechanisms}]
        if self.search_calls < self.args.search_calls - 1 and domains:
            self.recall(new, (unused or domains)[0])
        return new

    def _choose_frame(self, preferred: Frame | None) -> Frame:
        if preferred is not None:
            return preferred
        # Breadth-first with deterministic tie breaking; no LLM self-scored novelty.
        return min(self.frames, key=lambda f: (f.attempts, f.created_at))

    def transform(self, frame: Frame, j: int) -> None:
        if not self.mechanisms:
            return
        funcs = frame.data["required_functions"]
        options = [(f, m) for f in funcs for m in self.mechanisms]
        rng = random.Random(self.args.seed + j * 7919)
        rng.shuffle(options)
        options.sort(key=lambda fm: (self.pair_usage[(frame.id, fm[0]["name"], fm[1]["name"])],
                                     -pair_score(*fm)))
        func, mech = options[0]
        self.pair_usage[(frame.id, func["name"], mech["name"])] += 1
        op = OPERATIONS[j % len(OPERATIONS)]
        desc = {
            "invert": "Reverse causal direction or order if reversible; otherwise reject.",
            "transfer": "Transfer the causal interaction to a new physical or computational substrate.",
            "iterate": "Reuse output as input; derive conditions for stability, amplification or convergence.",
            "substitute": "Replace incumbent mechanism without violating its necessary invariants.",
            "factor": "Separate shared from local work or independently changeable components.",
            "relocate": "Move a decision/operation earlier, later, or nearer its information source."
        }[op]
        instr = (f"FRAME ID: {frame.id}\nFRAME (working representation, not a solved design):\n"
                 f"{json.dumps(frame.data, ensure_ascii=False)}\n\n"
                 f"TARGET FUNCTION JSON:\n{json.dumps(func, ensure_ascii=False)}\n\n"
                 f"SOURCE MECHANISM JSON:\n{json.dumps({k:v for k,v in mech.items() if not k.startswith('_')}, ensure_ascii=False)}\n\n"
                 f"OPERATION:\n{op}\n{desc}\n\n"
                 "Construct an operationally specific transformation, not a metaphor. "
                 "State the preserved function, falsifiable consequences, preconditions, "
                 "new frictions and a discriminating control. "
                 "REJECTION IS VALID: if a necessary condition is not met, return status='rejected' "
                 "with an actual counterexample/conflict test rather than inventing a solution. "
                 "A schema-valid candidate is a hypothesis, NEVER an independently established result.")
        r = self.ask("TRANSFORM", instr, CANDIDATE_SCHEMA, validate_candidate,
                     self.args.candidate_tokens,
                     {"frame_id": frame.id, "attempt": j, "operation": op,
                      "mechanism": mech["name"]}, expected_op=op, expected_source=mech["name"])
        frame.attempts += 1
        good = not r.errors and isinstance(r.parsed, dict)
        status = r.parsed.get("status") if good else "invalid"
        record = {"index": j, "frame_id": frame.id, "frame_name": frame.data["frame_name"],
                  "valid": good and status == "candidate", "rejected": good and status == "rejected",
                  "errors": r.errors, "candidate": r.parsed, "operation": op,
                  "source_domain": mech["source_domain"], "source_mechanism": mech["name"],
                  "parse_mode": r.parse_mode,
                  "signature": idea_signature(r.parsed) if isinstance(r.parsed, dict) else None}
        self.candidates.append(record)
        append_jsonl(self.out/"candidates.jsonl", record)
        parent_id = next((n.id for n in reversed(self.nodes) if n.kind == "frame" and n.frame_id == frame.id), None)
        if record["rejected"]:
            c = r.parsed
            signal = self.add_node("failed_constraint", frame.id, parent_id,
                {"failed_preconditions": c.get("failed_preconditions", []),
                 "invariant_at_risk": c.get("invariant_at_risk"),
                 "conflict_test": c.get("conflict_test")})
            self.signal_queue.insert(0, signal.id)
        elif record["valid"]:
            node = self.add_node("candidate", frame.id, parent_id,
                                 {"candidate_index": j, "title": r.parsed.get("title")})
            for friction in r.parsed.get("frictions_added", []):
                if isinstance(friction, dict) and friction.get("friction"):
                    signal = self.add_node("friction", frame.id, node.id,
                        {"name": friction["friction"], "cause": friction.get("mechanism"),
                         "measurement": friction.get("measurement")})
                    self.signal_queue.append(signal.id)
        else:
            # Missing or unparsable JSON is a format failure, not evidence of a
            # physically impossible transformation. Don't manufacture a reframe.
            self.add_node("invalid_output", frame.id, parent_id, {"errors": r.errors})

    def execute(self) -> dict[str, Any]:
        self.initial_frames()
        attempted = 0
        while self.search_calls < self.args.search_calls:
            available = self.args.search_calls - self.search_calls
            if available < 1:
                break
            preferred = self.maybe_reframe(attempted) if available >= 3 else None
            if self.search_calls >= self.args.search_calls:
                break
            self.transform(self._choose_frame(preferred), attempted)
            attempted += 1
        if not self.args.skip_baseline:
            # Pair candidate seeds first; after that, pair preparation seeds.
            chosen = [x.seed for x in self.calls[:self.search_calls] if x.stage == "TRANSFORM"]
            chosen += [x.seed for x in self.calls[:self.search_calls] if x.stage != "TRANSFORM"]
            if len(chosen) != self.search_calls:
                raise AssertionError("baseline seed plan does not match search calls")
            for j, seed in enumerate(chosen):
                r = self.ask("DIRECT", "Generate ONE concrete, testable candidate directly from the "
                             "original problem. Use established mechanisms, explain operational steps, "
                             "failure modes and constraints. This is a fresh independent generation; "
                             "you have no previous outputs. Set the literal operation field to 'direct'.",
                             CANDIDATE_SCHEMA, validate_candidate, self.args.candidate_tokens,
                             {"index": j}, expected_op="direct", seed_override=seed, in_search=False)
                valid = not r.errors and isinstance(r.parsed, dict)
                record = {"index": j, "valid": valid and r.parsed.get("status") == "candidate",
                          "rejected": valid and r.parsed.get("status") == "rejected",
                          "errors": r.errors, "candidate": r.parsed, "parse_mode": r.parse_mode,
                          "signature": idea_signature(r.parsed) if r.parsed else None}
                self.baseline.append(record)
                append_jsonl(self.out/"baseline.jsonl", record)
        summary = {
            "experiment": "multi-frame mechanism search" if self.args.mode == "multi" else "fixed single-frame mechanism search",
            "model": self.args.model, "frozen_weights": True, "cross_call_chat_memory": False,
            "search_call_budget": self.args.search_calls,
            "generation_calls_search": self.search_calls,
            "generation_calls_baseline": len(self.baseline),
            "independent_initial_frames": sum(f.origin == "independent" for f in self.frames),
            "new_reframes": len(self.frames) - sum(f.origin == "independent" for f in self.frames),
            "reframe_attempts": self.reframes,
            "frames_total": len(self.frames), "valid_mechanisms": len(self.mechanisms),
            "unique_recalled_domains": len({m["source_domain"] for m in self.mechanisms}),
            "frame_lexical_overlaps": {f.id: f.max_prior_frame_overlap for f in self.frames},
            "transform_attempts": len(self.candidates),
            "search_structured_candidates": sum(r["valid"] for r in self.candidates),
            "search_explicit_rejections": sum(r["rejected"] for r in self.candidates),
            "baseline_structured_candidates": sum(r["valid"] for r in self.baseline),
            "baseline_explicit_rejections": sum(r["rejected"] for r in self.baseline),
            "search_lexical_signatures": len({r["signature"] for r in self.candidates if r["valid"]}),
            "baseline_lexical_signatures": len({r["signature"] for r in self.baseline if r["valid"]}),
            "prompt_tokens_search": sum(t.prompt_tokens for t in self.calls[:self.search_calls]),
            "generated_tokens_search": sum(t.output_tokens for t in self.calls[:self.search_calls]),
            "prompt_tokens_baseline": sum(t.prompt_tokens for t in self.calls[self.search_calls:]),
            "generated_tokens_baseline": sum(t.output_tokens for t in self.calls[self.search_calls:]),
            "search_seconds": round(sum(t.seconds for t in self.calls[:self.search_calls]), 2),
            "baseline_seconds": round(sum(t.seconds for t in self.calls[self.search_calls:]), 2),
            "transforms": dict(Counter(r["operation"] for r in self.candidates)),
            "caveats": ["Frame constraints and mechanical claims are not externally verified",
                        "A valid schema is not a valid invention",
                        "Repaired invalid JSON escapes may conceal ambiguous LaTeX; inspect raw outputs",
                        "Same model-call budget does not equal same prompt-token FLOPs",
                        "Lexical signatures do not measure conceptual novelty"]
        }
        write_json(self.out/"summary.json", summary)
        write_json(self.out/"config.json", vars(self.args))
        self._write_markdown(summary)
        return summary

    def _write_markdown(self, summary: dict[str, Any]) -> None:
        md = ["# Multi-frame mechanism search — unverified hypotheses", "",
              "Schema acceptance is not causal validity or novelty. Model-recalled facts remain unverified.",
              "", "## Original user task", "", self.args.task,
              "", "## Immutable user constraints", "", "```json",
              json.dumps(self.args.constraint, ensure_ascii=False, indent=2), "```", "",
              "## Competing frames", ""]
        for f in self.frames:
            md += [f"### {f.id}: {f.data.get('frame_name')} ({f.origin})", "",
                   f"Parent frame: {f.parent_id or 'none'}; transformation attempts: {f.attempts}; "
                   f"max prior lexical overlap: {f.max_prior_frame_overlap:.3f}", "",
                   "```json", json.dumps(f.data, ensure_ascii=False, indent=2), "```", ""]
        md += ["## Source mechanisms", ""]
        for m in self.mechanisms:
            md += [f"### {m['name']} ({m['source_domain']}; frame {m['_origin_frame']})",
                   "", str(m.get("operation", "")), ""]
        for title, rows in (("Transformation attempts", self.candidates),
                            ("Direct-generation control", self.baseline)):
            md += [f"## {title}", ""]
            for r in rows:
                c = r["candidate"]
                name = c.get("title", "Unparseable") if isinstance(c, dict) else "Unparseable"
                md += [f"### {r['index']+1}. {name}", "",
                       f"Record status: {'CANDIDATE' if r['valid'] else 'INCOMPATIBLE' if r['rejected'] else 'INVALID'}; "
                       f"JSON parse: {r['parse_mode']}", ""]
                if r.get("frame_id"):
                    md += [f"Frame: {r['frame_id']} ({r['frame_name']})", ""]
                if r["errors"]:
                    md += ["Format/structural errors: " + ", ".join(r["errors"]), ""]
                md += ["```json", json.dumps(c, ensure_ascii=False, indent=2), "```", ""]
        md += ["## Search frontier (including rejected paths and new frictions)", ""]
        for n in self.nodes:
            if n.kind in ("friction", "failed_constraint", "exhausted"):
                md += [f"- {n.kind} [{n.frame_id}]: " + json.dumps(n.payload, ensure_ascii=False)]
        md += ["", "## Machine diagnostics (not discovery scores)", "", "```json",
               json.dumps(summary, indent=2), "```", ""]
        (self.out/"ideas.md").write_text("\n".join(md), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Search mechanisms across competing, revisable problem frames")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--constraint", action="append", default=[],
                   help="Explicit, immutable user constraint. Can repeat.")
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--loader", choices=("auto", "multimodal", "causal"), default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--mode", choices=("multi", "single"), default="multi",
                   help="Single frame freezes one abstraction; multi permits competing frames and reframes.")
    p.add_argument("--frames", type=int, default=3, help="Independent initial frames in multi mode")
    p.add_argument("--max-reframes", type=int, default=1,
                   help="Maximum attempts to reframe from discovered friction/failure")
    p.add_argument("--domains", default=None,
                   help="Optional explicit comma-separated source domains (applies to all frames)")
    p.add_argument("--domains-per-frame", type=int, default=1)
    p.add_argument("--single-domains", type=int, default=3)
    p.add_argument("--mechanisms-per-domain", type=int, default=2)
    p.add_argument("--search-calls", type=int, default=16,
                   help="Exact model-call budget for search; DIRECT uses same count")
    p.add_argument("--analysis-tokens", type=int, default=1350)
    p.add_argument("--recall-tokens", type=int, default=1100)
    p.add_argument("--candidate-tokens", type=int, default=1600)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--mock", action="store_true", help="Synthetic harness smoke test only")
    p.add_argument("--out", default="runs/multiframe_search")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    n = args.frames if args.mode == "multi" else 1
    if (n < 1 or args.search_calls < n + 2 or args.max_reframes < 0 or
        args.domains_per_frame < 1 or args.single_domains < 1 or args.mechanisms_per_domain < 1):
        raise SystemExit("Invalid frames/search-calls/reframes/domains count; need budget for frames, recall and search")
    model: Backend = MockBackend() if args.mock else HuggingFaceBackend(
        args.model, args.device, args.dtype, args.loader, args.trust_remote_code)
    results = MultiFrameExperiment(args, model).execute()
    print("\n" + json.dumps(results, indent=2))
    print(f"\nWrote {args.out}/ideas.md, frames.jsonl, frontier.jsonl and raw logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
