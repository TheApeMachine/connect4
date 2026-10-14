from __future__ import annotations

import torch

from rotoformer.affine import (
    AffineChannelRotor,
    AffineRotationSpec,
    build_affine_indices,
)


def test_identity_k_zero() -> None:
    rotor = AffineChannelRotor(width=37, prime=17, generator=3, k=0)
    x = torch.randn(2, 4, 37)
    assert torch.equal(rotor.rotate(x), x)
    assert torch.equal(rotor.inverse(x), x)


def test_rotation_is_exactly_invertible_with_tail() -> None:
    rotor = AffineChannelRotor(width=39, prime=17, generator=3, k=5)
    x = torch.randn(3, 7, 39)
    y = rotor.rotate(x)
    restored = rotor.inverse(y)
    assert torch.equal(restored, x)
    # Two complete fields rotate; the five-channel remainder stays fixed.
    assert torch.equal(y[..., 34:], x[..., 34:])


def test_indices_follow_affine_source_to_destination_formula() -> None:
    spec = AffineRotationSpec(width=17, prime=17, generator=3, k=4)
    rotate_index, inverse_index = build_affine_indices(spec)
    x = torch.arange(17)
    y = x.index_select(0, rotate_index)

    a = pow(3, 4, 17)
    b = 4
    for source in range(17):
        destination = (a * source + b) % 17
        assert y[destination].item() == source

    restored = y.index_select(0, inverse_index)
    assert torch.equal(restored, x)


def test_set_k_changes_and_can_return_to_identity() -> None:
    rotor = AffineChannelRotor(width=34, prime=17, generator=3, k=0)
    x = torch.arange(34).reshape(1, 34)
    rotor.set_k(8)
    moved = rotor.rotate(x)
    assert not torch.equal(moved, x)
    rotor.set_k(0)
    assert torch.equal(rotor.rotate(x), x)
