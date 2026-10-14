"""Model backend copied verbatim from the supplied multi-frame experiment.
The search controller, not this loader, is new.
"""
from __future__ import annotations
import dataclasses
import os
import random
import sys
import time
from typing import Any, Protocol

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
