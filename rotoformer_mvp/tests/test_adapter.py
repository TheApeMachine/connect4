from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from rotoformer.adapter import RotoformerAdapter, RotoformerConfig, resolve_layer_spec
from rotoformer.layers import RotoEnterWrapper, RotoExitWrapper, RotoLayerWrapper


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor):
        return (hidden_states + self.act(self.linear(hidden_states)),)


class DummyModel(nn.Module):
    def __init__(self, width: int = 34, depth: int = 5) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width, num_hidden_layers=depth)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(width) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


def test_layer_spec_tail_and_band() -> None:
    assert resolve_layer_spec("tail:3", 8) == [3, 4, 5, 6, 7]
    assert resolve_layer_spec("last:2", 8) == [6, 7]
    assert resolve_layer_spec("band:0.25-0.75", 8) == [2, 3, 4, 5]


def test_span_identity_rotation_is_exact_and_detach_restores() -> None:
    torch.manual_seed(5)
    model = DummyModel()
    x = torch.randn(2, 6, 34)
    baseline = model(x)
    original_first = model.model.layers[2]
    original_last = model.model.layers[4]

    adapter = RotoformerAdapter(
        model,
        RotoformerConfig(
            layer_spec="tail:2", scope="span", prime=17, generator=3, k=0
        ),
    )
    assert isinstance(model.model.layers[2], RotoEnterWrapper)
    assert isinstance(model.model.layers[4], RotoExitWrapper)
    assert torch.equal(model(x), baseline)

    adapter.set_rotation(4)
    rotated = model(x)
    assert not torch.allclose(rotated, baseline)

    adapter.detach()
    assert model.model.layers[2] is original_first
    assert model.model.layers[4] is original_last
    assert torch.equal(model(x), baseline)


def test_layerwise_rotation_and_depth_stride() -> None:
    model = DummyModel(depth=4)
    adapter = RotoformerAdapter(
        model,
        RotoformerConfig(
            layer_spec="all",
            scope="layerwise",
            prime=17,
            generator=3,
            k=2,
            depth_stride=3,
        ),
    )
    assert all(isinstance(layer, RotoLayerWrapper) for layer in model.model.layers)
    assert [layer.rotor.k for layer in model.model.layers] == [2, 5, 8, 11]
    adapter.set_rotation(4)
    assert [layer.rotor.k for layer in model.model.layers] == [4, 7, 10, 13]
    adapter.detach()
