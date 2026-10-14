# The Rotoformer

**Rotoformer** is an inference-time experiment in making frozen transformer layers rotatable.

The first implementation uses the finite-field affine family:

\[
\operatorname{slot}_k(i) = \left(3^k i + k\right) \bmod 257.
\]

That map is converted into an exact channel permutation \(P_k\). A selected transformer layer becomes:

\[
F_k(x) = P_k^{-1} F(P_k x).
\]

No weights are edited. The same frozen layer is exposed through a structured family of reversible coordinate frames.

The especially interesting mode is a **span**—a layer and its tail. For layers \(F_j,\ldots,F_n\), Rotoformer evaluates:

\[
T_{j:n,k}(x) = P_k^{-1}\left(F_n \circ \cdots \circ F_j\right)(P_k x).
\]

The representation enters the tail in a rotated basis, stays there through the selected depth, and is returned to the model's ordinary basis at the end.

## What this MVP does

- Applies the exact `slot(i) = (3^k * i + k) mod 257` family.
- Rotates complete 257-channel fields in a model's hidden dimension.
- Leaves any remainder channels untouched, so the transform remains lossless and invertible.
- Supports one conjugated layer, a contiguous span/tail, or independent layerwise rotors.
- Brute-forces a list—or all 257 initial rotations—without retraining.
- Scores each frame by prompt-side next-token loss, final-token entropy, or logit margin.
- Generates under the winning frame and the identity frame for immediate comparison.
- Restores the original module stack exactly when detached.

This is intentionally a probe, not yet a trained architecture. The first question is simple: **does a frozen model have input-dependent preferred affine frames?**

## Install

```bash
cd rotoformer_mvp
python -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,dev]'
```

The core rotor and wrappers only require PyTorch. The Hugging Face sweep runner uses the optional `hf` dependencies.

## Quick mechanical test

This demo needs no model download:

```bash
python examples/toy_rotoformer.py
```

Expected shape of the result:

```text
identity max delta: 0.000e+00
rotated max delta:  non-zero
restore max delta: 0.000e+00
```

## First real run

Use any compatible local or Hugging Face causal language model:

```bash
rotoformer-sweep \
  --model <MODEL_ID_OR_LOCAL_PATH> \
  --prompt "Maybe you, and your tail" \
  --layer-spec late \
  --scope span \
  --rotations even:17 \
  --objective prompt_nll \
  --max-new-tokens 160 \
  --output-dir results
```

For a complete first orbit over `k = 0..256`:

```bash
rotoformer-sweep \
  --model <MODEL_ID_OR_LOCAL_PATH> \
  --prompt "Explain what a movable weight geometry might make possible." \
  --layer-spec tail:12 \
  --scope span \
  --rotations all \
  --objective prompt_nll \
  --generate-top 3 \
  --output-dir results
```

`tail:12` means layer 12 through the final decoder layer. Other useful selections are `single:8`, `range:8-15`, `last:4`, `early`, `middle`, `late`, and `all`.

For a layerwise phase walk instead of one shared tail frame:

```bash
rotoformer-sweep \
  --model <MODEL_ID_OR_LOCAL_PATH> \
  --prompt "A short test prompt" \
  --layer-spec late \
  --scope layerwise \
  --depth-stride 1 \
  --rotations range:0-64:4
```

With `--depth-stride 1`, selected layer zero uses `k`, the next selected layer uses `k+1`, and so on.

## Device examples

Single-device Apple Silicon:

```bash
rotoformer-sweep \
  --model <MODEL_ID_OR_LOCAL_PATH> \
  --prompt "Test" \
  --device mps \
  --device-map none \
  --dtype float16
```

Accelerate-managed sharding or offload:

```bash
rotoformer-sweep \
  --model <MODEL_ID_OR_LOCAL_PATH> \
  --prompt "Test" \
  --device-map auto \
  --dtype auto
```

## Outputs

Each run writes a timestamped directory containing:

- `scores.csv` — one row per rotation.
- `summary.json` — configuration, attachment details, winner, metrics, and generations.
- `generations.jsonl` — identity and top-frame outputs.
- `prompt.txt` — the exact rendered prompt seen by the tokenizer.

The logged rotation record includes `k`, multiplier \(3^k \bmod 257\), translation, prompt loss, perplexity, final-token entropy, logit margin, and score latency.

## Python API

```python
from rotoformer import RotoformerAdapter, RotoformerConfig

adapter = RotoformerAdapter(
    model,
    RotoformerConfig(
        layer_spec="tail:12",
        scope="span",
        prime=257,
        generator=3,
        k=0,
    ),
)

adapter.set_rotation(84)
output = model(**inputs)

adapter.detach()  # restores the original decoder modules
```

## Why blockwise GF(257)?

Most hidden widths are not themselves 257. The MVP treats the channel axis as a row of complete 257-element fields:

```text
[ GF(257) ][ GF(257) ][ GF(257) ] ... [ untouched remainder ]
```

Every complete field receives the same affine permutation. The untouched remainder is deliberate: padding and then dropping rotated coordinates would not be bijective on the original hidden state.

The attachment summary reports the exact field count and channel coverage. A width of 4096, for example, contains 15 complete fields and a 241-channel identity remainder.

## Current limits

The selected `k` is static during one generation. The sweep is prompt-level, not token-level. Prompt NLL is a convenient first selector, not a claim that it is the final or best selection signal. The wrappers target common decoder `ModuleList` layouts and may need a small adapter for unusual remote-code architectures. Running all 257 candidates multiplies the prompt-scoring cost by 257.

Most importantly, this version tests **structured reversible re-expression**. It does not yet implement your affine resonance/neighbor statistic over hidden states. That belongs in the next selector once we decide what an activation-space “neighbor” should mean without smuggling in an arbitrary binning choice.

## Tests

```bash
pytest
```

The test suite verifies exact inversion, identity at `k=0`, the affine source-to-destination formula, explicit layer conjugation, tail-span behavior, layerwise phase walks, and exact restoration of the original stack.
