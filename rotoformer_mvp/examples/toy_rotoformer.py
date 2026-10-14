#!/usr/bin/env python3
"""Self-contained proof that a layer span can be conjugated and restored."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from rotoformer import RotoformerAdapter, RotoformerConfig


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden_states + self.ff(self.norm(hidden_states)),)


class ToyModel(nn.Module):
    def __init__(self, width: int = 34, depth: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width, num_hidden_layers=depth)
        self.model = SimpleNamespace()
        # Register the stack through an nn.Module container while preserving the
        # familiar model.layers path used by many decoder-only architectures.
        self.layers = nn.ModuleList([ResidualBlock(width) for _ in range(depth)])
        self.model.layers = self.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)[0]
        return x


def main() -> None:
    torch.manual_seed(7)
    model = ToyModel()
    x = torch.randn(2, 5, 34)
    baseline = model(x)

    adapter = RotoformerAdapter(
        model,
        RotoformerConfig(
            layer_spec="tail:1",
            scope="span",
            prime=17,
            generator=3,
            k=0,
        ),
    )
    identity = model(x)
    adapter.set_rotation(5)
    rotated = model(x)
    adapter.detach()
    restored = model(x)

    print(f"identity max delta: {(baseline - identity).abs().max().item():.3e}")
    print(f"rotated max delta:  {(baseline - rotated).abs().max().item():.3e}")
    print(f"restore max delta:  {(baseline - restored).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
