"""Rotoformer: inference-time finite-field rotations for transformer layers."""

from .adapter import (
    RotoformerAdapter,
    RotoformerConfig,
    find_decoder_layers,
    get_hidden_width,
    resolve_layer_spec,
)
from .affine import AffineChannelRotor, AffineRotationSpec, build_affine_indices
from .layers import RotoEnterWrapper, RotoExitWrapper, RotoLayerWrapper
from .scoring import CandidateScore, choose_candidate, score_prompt, sweep_rotations

__all__ = [
    "AffineChannelRotor",
    "AffineRotationSpec",
    "build_affine_indices",
    "RotoLayerWrapper",
    "RotoEnterWrapper",
    "RotoExitWrapper",
    "RotoformerConfig",
    "RotoformerAdapter",
    "find_decoder_layers",
    "get_hidden_width",
    "resolve_layer_spec",
    "CandidateScore",
    "score_prompt",
    "sweep_rotations",
    "choose_candidate",
]

__version__ = "0.1.0"
