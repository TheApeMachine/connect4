from __future__ import annotations

import torch
from torch import nn

from rotoformer.affine import AffineChannelRotor
from rotoformer.layers import RotoEnterWrapper, RotoExitWrapper, RotoLayerWrapper


class TupleBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden_states: torch.Tensor, marker: int = 3):
        return (self.norm(hidden_states + self.linear(hidden_states)), marker)


def test_single_layer_wrapper_matches_explicit_conjugation() -> None:
    torch.manual_seed(2)
    width = 34
    block = TupleBlock(width)
    rotor = AffineChannelRotor(width, prime=17, generator=3, k=6)
    wrapper = RotoLayerWrapper(block, rotor)
    x = torch.randn(2, 5, width)

    explicit = block(rotor.rotate(x))
    explicit_hidden = rotor.inverse(explicit[0])
    actual = wrapper(x)

    assert torch.allclose(actual[0], explicit_hidden)
    assert actual[1] == explicit[1]


def test_enter_and_exit_conjugate_a_span() -> None:
    torch.manual_seed(3)
    width = 34
    first = TupleBlock(width)
    second = TupleBlock(width)
    rotor_a = AffineChannelRotor(width, prime=17, generator=3, k=7)
    rotor_b = AffineChannelRotor(width, prime=17, generator=3, k=7)
    enter = RotoEnterWrapper(first, rotor_a)
    exit_ = RotoExitWrapper(second, rotor_b)
    x = torch.randn(2, 4, width)

    explicit_first = first(rotor_a.rotate(x))[0]
    explicit_second = second(explicit_first)[0]
    explicit = rotor_b.inverse(explicit_second)

    actual = exit_(enter(x)[0])[0]
    assert torch.allclose(actual, explicit)


def test_wrapper_delegates_inner_attributes() -> None:
    block = TupleBlock(17)
    wrapper = RotoLayerWrapper(
        block, AffineChannelRotor(17, prime=17, generator=3, k=1)
    )
    assert wrapper.linear is block.linear
