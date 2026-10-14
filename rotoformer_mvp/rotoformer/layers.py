"""Transparent wrappers that conjugate transformer layers by an affine rotor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import torch
from torch import Tensor, nn

from .affine import AffineChannelRotor


HiddenTransform = Callable[[Tensor], Tensor]


def _transform_hidden_input(
    args: tuple[Any, ...], kwargs: dict[str, Any], transform: HiddenTransform
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args_list = list(args)
    kwargs_copy = dict(kwargs)

    if args_list and torch.is_tensor(args_list[0]):
        args_list[0] = transform(args_list[0])
        return tuple(args_list), kwargs_copy

    hidden = kwargs_copy.get("hidden_states")
    if torch.is_tensor(hidden):
        kwargs_copy["hidden_states"] = transform(hidden)
        return tuple(args_list), kwargs_copy

    raise TypeError(
        "Could not locate decoder hidden states. Expected the first positional argument "
        "or a 'hidden_states' keyword tensor."
    )


def _transform_hidden_output(output: Any, transform: HiddenTransform) -> Any:
    if torch.is_tensor(output):
        return transform(output)

    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("decoder layer returned a tuple without a tensor in position 0")
        return (transform(output[0]), *output[1:])

    if isinstance(output, list):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("decoder layer returned a list without a tensor in position 0")
        copied = list(output)
        copied[0] = transform(copied[0])
        return copied

    # This is uncommon for individual decoder blocks, but a few custom models use
    # dict-like outputs. Preserve their concrete type when practical.
    if isinstance(output, Mapping):
        for key in ("hidden_states", "last_hidden_state"):
            if key in output and torch.is_tensor(output[key]):
                try:
                    copied = output.copy()
                    copied[key] = transform(output[key])
                    return copied
                except Exception as exc:  # pragma: no cover - model-specific fallback
                    raise TypeError(
                        f"could not replace {key!r} in decoder output type {type(output)!r}"
                    ) from exc

    raise TypeError(
        "Unsupported decoder-layer output shape. Expected Tensor, tuple/list with the "
        f"hidden tensor first, or a mapping; got {type(output)!r}."
    )


class _TransparentLayerWrapper(nn.Module):
    """Delegate unknown attributes to the wrapped decoder layer."""

    def __init__(self, inner: nn.Module, rotor: AffineChannelRotor) -> None:
        super().__init__()
        self.inner = inner
        self.rotor = rotor

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as original:
            inner = self.__dict__.get("_modules", {}).get("inner")
            if inner is not None and hasattr(inner, name):
                return getattr(inner, name)
            raise original

    def set_k(self, k: int) -> None:
        self.rotor.set_k(k)


class RotoLayerWrapper(_TransparentLayerWrapper):
    """Conjugate one layer: ``P_k^-1 · layer · P_k``."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        rotated_args, rotated_kwargs = _transform_hidden_input(
            args, kwargs, self.rotor.rotate
        )
        output = self.inner(*rotated_args, **rotated_kwargs)
        return _transform_hidden_output(output, self.rotor.inverse)


class RotoEnterWrapper(_TransparentLayerWrapper):
    """Rotate at the beginning of a contiguous layer span."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        rotated_args, rotated_kwargs = _transform_hidden_input(
            args, kwargs, self.rotor.rotate
        )
        return self.inner(*rotated_args, **rotated_kwargs)


class RotoExitWrapper(_TransparentLayerWrapper):
    """Undo the rotation at the end of a contiguous layer span."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.inner(*args, **kwargs)
        return _transform_hidden_output(output, self.rotor.inverse)
