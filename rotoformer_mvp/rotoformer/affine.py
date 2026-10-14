"""Finite-field affine channel rotations.

The default family follows the geometry discussed for Six::

    slot(i) = (3**k * i + k) mod 257

For a hidden width larger than the field, the same bijection is applied to every
complete 257-channel block. Any remainder is left untouched, keeping the
operation exactly invertible without padding or information loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Literal

import torch
from torch import Tensor, nn

TranslationMode = Literal["k", "zero"]


@dataclass(frozen=True)
class AffineRotationSpec:
    """Description of one member of the affine rotation family."""

    width: int
    k: int
    prime: int = 257
    generator: int = 3
    translation_mode: TranslationMode = "k"

    @property
    def multiplier(self) -> int:
        return pow(self.generator, self.k, self.prime)

    @property
    def translation(self) -> int:
        if self.translation_mode == "zero":
            return 0
        return self.k % self.prime

    @property
    def full_fields(self) -> int:
        return self.width // self.prime

    @property
    def rotated_channels(self) -> int:
        return self.full_fields * self.prime

    @property
    def remainder_channels(self) -> int:
        return self.width - self.rotated_channels

    @property
    def coverage(self) -> float:
        return self.rotated_channels / self.width if self.width else 0.0


def _validate_spec(spec: AffineRotationSpec) -> None:
    if spec.width <= 0:
        raise ValueError(f"width must be positive, got {spec.width}")
    if spec.prime <= 2:
        raise ValueError(f"prime must be greater than 2, got {spec.prime}")
    if not 0 < spec.generator < spec.prime:
        raise ValueError(
            f"generator must be in [1, prime-1], got {spec.generator} for GF({spec.prime})"
        )
    if gcd(spec.multiplier, spec.prime) != 1:
        raise ValueError(
            "affine multiplier is not invertible; choose a prime field and a non-zero generator"
        )
    if spec.translation_mode not in ("k", "zero"):
        raise ValueError(f"unsupported translation mode: {spec.translation_mode!r}")


def build_affine_indices(spec: AffineRotationSpec) -> tuple[Tensor, Tensor]:
    """Build gather indices for the rotation and its inverse.

    Returns two int64 tensors of shape ``[width]``:

    * ``rotate_index`` makes ``rotated = x.index_select(-1, rotate_index)``.
    * ``inverse_index`` restores ``x = rotated.index_select(-1, inverse_index)``.

    The source-to-destination map inside one field is::

        destination(i) = (a*i + b) mod p

    where ``a = generator**k mod p`` and ``b = k mod p`` by default.
    """

    _validate_spec(spec)

    p = spec.prime
    source = torch.arange(p, dtype=torch.long)
    destination = (spec.multiplier * source + spec.translation) % p

    # Gather form: output[destination] = input[source].
    source_for_destination = torch.empty_like(source)
    source_for_destination[destination] = source

    rotate_index = torch.arange(spec.width, dtype=torch.long)
    inverse_index = torch.arange(spec.width, dtype=torch.long)

    for field_idx in range(spec.full_fields):
        start = field_idx * p
        stop = start + p
        rotate_index[start:stop] = start + source_for_destination
        inverse_index[start:stop] = start + destination

    return rotate_index, inverse_index


class AffineChannelRotor(nn.Module):
    """A reversible affine permutation over the final tensor dimension.

    The rotor has no trainable parameters. ``set_k`` changes the active member of
    the family without touching model weights.
    """

    def __init__(
        self,
        width: int,
        *,
        k: int = 0,
        prime: int = 257,
        generator: int = 3,
        translation_mode: TranslationMode = "k",
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.prime = int(prime)
        self.generator = int(generator)
        self.translation_mode: TranslationMode = translation_mode
        self.k = int(k)

        rotate_index, inverse_index = build_affine_indices(self.spec)
        self.register_buffer("rotate_index", rotate_index, persistent=True)
        self.register_buffer("inverse_index", inverse_index, persistent=True)

    @property
    def spec(self) -> AffineRotationSpec:
        return AffineRotationSpec(
            width=self.width,
            k=self.k,
            prime=self.prime,
            generator=self.generator,
            translation_mode=self.translation_mode,
        )

    def set_k(self, k: int) -> None:
        """Select a new affine rotation while preserving the current buffer device."""

        self.k = int(k)
        rotate_index, inverse_index = build_affine_indices(self.spec)
        self.rotate_index = rotate_index.to(self.rotate_index.device)
        self.inverse_index = inverse_index.to(self.inverse_index.device)

    def _index_on(self, index: Tensor, device: torch.device) -> Tensor:
        # Accelerate/offload may move hidden states independently of the wrapper.
        # Index tensors are tiny, so moving lazily is safer than assuming one device.
        if index.device != device:
            return index.to(device=device, non_blocking=True)
        return index

    def rotate(self, hidden_states: Tensor) -> Tensor:
        self._validate_hidden(hidden_states)
        index = self._index_on(self.rotate_index, hidden_states.device)
        return hidden_states.index_select(-1, index)

    def inverse(self, hidden_states: Tensor) -> Tensor:
        self._validate_hidden(hidden_states)
        index = self._index_on(self.inverse_index, hidden_states.device)
        return hidden_states.index_select(-1, index)

    def forward(self, hidden_states: Tensor, *, inverse: bool = False) -> Tensor:
        return self.inverse(hidden_states) if inverse else self.rotate(hidden_states)

    def _validate_hidden(self, hidden_states: Tensor) -> None:
        if not torch.is_tensor(hidden_states):
            raise TypeError(f"expected a Tensor, got {type(hidden_states)!r}")
        if hidden_states.ndim == 0:
            raise ValueError("hidden state tensor must have at least one dimension")
        if hidden_states.shape[-1] != self.width:
            raise ValueError(
                f"expected hidden width {self.width}, got shape {tuple(hidden_states.shape)}"
            )

    def extra_repr(self) -> str:
        spec = self.spec
        return (
            f"width={self.width}, k={self.k}, GF({self.prime}), generator={self.generator}, "
            f"a={spec.multiplier}, b={spec.translation}, coverage={spec.coverage:.3f}"
        )
