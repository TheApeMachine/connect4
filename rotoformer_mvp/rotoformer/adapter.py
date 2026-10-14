"""Attach affine rotors to decoder stacks in Hugging Face-style causal LMs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Sequence

from torch import nn

from .affine import AffineChannelRotor, TranslationMode
from .layers import RotoEnterWrapper, RotoExitWrapper, RotoLayerWrapper

Scope = Literal["span", "layerwise"]


KNOWN_LAYER_PATHS = (
    "model.layers",          # Some bare decoder models
    "model.model.layers",    # Llama, Mistral, Qwen, Gemma causal-LM wrappers
    "transformer.h",         # GPT-2 and several GPT-like models
    "gpt_neox.layers",       # GPT-NeoX / Pythia
    "model.decoder.layers",  # OPT/BART-like wrappers
    "decoder.layers",
    "layers",
)


def _get_by_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise AttributeError(path)
        current = getattr(current, part)
    return current


def _config_int(model: nn.Module, names: Sequence[str]) -> int | None:
    configs = [getattr(model, "config", None)]
    root_cfg = configs[0]
    if root_cfg is not None:
        configs.append(getattr(root_cfg, "text_config", None))

    for cfg in configs:
        if cfg is None:
            continue
        for name in names:
            value = getattr(cfg, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def get_hidden_width(model: nn.Module) -> int:
    value = _config_int(model, ("hidden_size", "n_embd", "d_model", "dim"))
    if value is None:
        raise ValueError(
            "Could not determine hidden width from model.config. Expected hidden_size, "
            "n_embd, d_model, or dim."
        )
    return value


def get_num_layers(model: nn.Module) -> int | None:
    return _config_int(
        model, ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
    )


def find_decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """Return the most likely decoder ``ModuleList`` and its dotted path."""

    for path in KNOWN_LAYER_PATHS:
        try:
            value = _get_by_path(model, path)
        except AttributeError:
            continue
        if isinstance(value, nn.ModuleList) and len(value) > 0:
            return path, value

    expected = get_num_layers(model)
    candidates: list[tuple[int, str, nn.ModuleList]] = []
    for path, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            continue
        score = 0
        lower = path.lower()
        if expected is not None and len(module) == expected:
            score += 100
        if any(token in lower for token in ("layers", "blocks", ".h", "decoder")):
            score += 20
        if all(isinstance(child, nn.Module) for child in module):
            score += 1
        candidates.append((score, path, module))

    if not candidates:
        raise ValueError("No decoder ModuleList could be found in the model")

    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    _, path, layers = candidates[0]
    return path, layers


def resolve_layer_spec(spec: str, num_layers: int) -> list[int]:
    """Resolve a compact decoder-layer selection expression.

    Supported forms include ``all``, ``early``, ``middle``, ``late``,
    ``single:12``, ``range:8-15``, ``tail:12`` (layer 12 through the end),
    ``last:4``, ``every:3``, ``band:0.5-1.0``, and ``list:1,4,7``.
    """

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    raw = spec.strip().lower()
    if raw == "all":
        return list(range(num_layers))
    if raw == "early":
        return list(range(0, max(1, num_layers // 3)))
    if raw == "middle":
        return list(range(num_layers // 3, max(num_layers // 3 + 1, 2 * num_layers // 3)))
    if raw == "late":
        return list(range(2 * num_layers // 3, num_layers))

    if raw.startswith("single:"):
        result = [int(raw.split(":", 1)[1])]
    elif raw.startswith("range:"):
        left, right = raw.split(":", 1)[1].split("-", 1)
        lo, hi = sorted((int(left), int(right)))
        result = list(range(lo, hi + 1))
    elif raw.startswith("tail:"):
        start = int(raw.split(":", 1)[1])
        result = list(range(start, num_layers))
    elif raw.startswith("last:"):
        count = int(raw.split(":", 1)[1])
        if count <= 0:
            raise ValueError("last:<count> requires a positive count")
        result = list(range(max(0, num_layers - count), num_layers))
    elif raw.startswith("every:"):
        step = int(raw.split(":", 1)[1])
        if step <= 0:
            raise ValueError("every:<step> requires a positive step")
        result = list(range(0, num_layers, step))
    elif raw.startswith("band:"):
        left, right = raw.split(":", 1)[1].split("-", 1)
        lo, hi = sorted((float(left), float(right)))
        if lo < 0 or hi > 1:
            raise ValueError("band fractions must lie in [0, 1]")
        start = min(num_layers - 1, max(0, math.floor(lo * num_layers)))
        stop = min(num_layers, max(start + 1, math.ceil(hi * num_layers)))
        result = list(range(start, stop))
    elif raw.startswith("list:"):
        result = sorted(
            {int(piece.strip()) for piece in raw.split(":", 1)[1].split(",") if piece.strip()}
        )
    elif "," in raw:
        result = sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()})
    else:
        raise ValueError(f"unsupported layer spec: {spec!r}")

    if not result:
        raise ValueError(f"layer spec selected no layers: {spec!r}")
    invalid = [idx for idx in result if idx < 0 or idx >= num_layers]
    if invalid:
        raise ValueError(
            f"layer indices out of range for {num_layers} layers: {invalid}"
        )
    return result


@dataclass
class RotoformerConfig:
    layer_spec: str = "late"
    scope: Scope = "span"
    k: int = 0
    prime: int = 257
    generator: int = 3
    translation_mode: TranslationMode = "k"
    depth_stride: int = 0

    def validate(self) -> None:
        if self.scope not in ("span", "layerwise"):
            raise ValueError(f"unsupported scope: {self.scope!r}")
        if self.scope == "span" and self.depth_stride != 0:
            raise ValueError("depth_stride is only meaningful for scope='layerwise'")


class RotoformerAdapter:
    """Temporarily replace decoder layers with reversible rotation wrappers."""

    def __init__(self, model: nn.Module, config: RotoformerConfig) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.layer_path, self.layers = find_decoder_layers(model)
        self.width = get_hidden_width(model)
        self.layer_indices = resolve_layer_spec(config.layer_spec, len(self.layers))
        self._originals: dict[int, nn.Module] = {}
        self._wrappers: list[tuple[int, Any]] = []
        self._attached = False
        self.attach()

    def _new_rotor(self, k: int) -> AffineChannelRotor:
        return AffineChannelRotor(
            self.width,
            k=k,
            prime=self.config.prime,
            generator=self.config.generator,
            translation_mode=self.config.translation_mode,
        )

    def attach(self) -> None:
        if self._attached:
            return

        selected = self.layer_indices
        if self.config.scope == "span":
            expected = list(range(selected[0], selected[-1] + 1))
            if selected != expected:
                raise ValueError(
                    "scope='span' requires a contiguous layer selection; use range:, tail:, "
                    "early/middle/late, or switch to scope='layerwise'"
                )

            first, last = selected[0], selected[-1]
            if first == last:
                original = self.layers[first]
                self._originals[first] = original
                wrapper = RotoLayerWrapper(original, self._new_rotor(self.config.k))
                self.layers[first] = wrapper
                self._wrappers.append((0, wrapper))
            else:
                first_original = self.layers[first]
                last_original = self.layers[last]
                self._originals[first] = first_original
                self._originals[last] = last_original

                enter = RotoEnterWrapper(first_original, self._new_rotor(self.config.k))
                exit_ = RotoExitWrapper(last_original, self._new_rotor(self.config.k))
                self.layers[first] = enter
                self.layers[last] = exit_
                self._wrappers.extend(((0, enter), (0, exit_)))
        else:
            for relative, index in enumerate(selected):
                original = self.layers[index]
                self._originals[index] = original
                effective_k = self.config.k + relative * self.config.depth_stride
                wrapper = RotoLayerWrapper(original, self._new_rotor(effective_k))
                self.layers[index] = wrapper
                self._wrappers.append((relative, wrapper))

        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        for index, original in self._originals.items():
            self.layers[index] = original
        self._originals.clear()
        self._wrappers.clear()
        self._attached = False

    def set_rotation(self, k: int) -> None:
        self.config.k = int(k)
        for relative, wrapper in self._wrappers:
            effective_k = self.config.k + relative * self.config.depth_stride
            wrapper.set_k(effective_k)

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def current_spec(self):
        if not self._wrappers:
            raise RuntimeError("Rotoformer adapter is detached")
        return self._wrappers[0][1].rotor.spec

    def describe(self) -> dict[str, Any]:
        first_rotor = self._wrappers[0][1].rotor
        spec = first_rotor.spec
        return {
            "layer_path": self.layer_path,
            "num_decoder_layers": len(self.layers),
            "selected_layer_indices": list(self.layer_indices),
            "hidden_width": self.width,
            "config": asdict(self.config),
            "multiplier": spec.multiplier,
            "translation": spec.translation,
            "full_fields": spec.full_fields,
            "rotated_channels": spec.rotated_channels,
            "remainder_channels": spec.remainder_channels,
            "coverage": spec.coverage,
        }

    def __enter__(self) -> "RotoformerAdapter":
        self.attach()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.detach()
