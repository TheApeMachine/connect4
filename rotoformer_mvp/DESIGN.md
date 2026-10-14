# Rotoformer Design Notes

## 1. The object being rotated

The MVP rotates the hidden-channel coordinate system, not token order and not parameter storage.

For a hidden state `x[..., d]`, the affine map supplies a permutation `P_k` over complete `GF(257)` blocks in `d`. Because it is a permutation, it preserves every scalar value exactly. Only addresses change.

A layer wrapper performs:

```text
ordinary basis ──P_k──► rotated basis ──frozen layer──► rotated basis ──P_k^-1──► ordinary basis
```

For a span, only the entry and exit boundaries rotate. The entire interior tail runs in the new basis.

## 2. Why this is not weight jitter

Jitter adds a perturbation to parameter values and must later subtract it. Rotoformer leaves parameters untouched. It changes the function presented by a frozen layer through conjugation:

```text
W                ordinary linear layer
P^-1 W P         rotated linear layer
```

For a nonlinear transformer block, the same construction applies to the whole block function. The family is deterministic, reversible, and indexed by one small integer `k`.

## 3. The first selector

The sweep uses teacher-forced prompt NLL because it is immediately available from a causal LM with no labels beyond the prompt itself. It asks:

> Under which frame does this frozen model find the observed prompt least surprising?

Entropy and top-two logit margin are logged as alternative selectors. They are deliberately kept separate because low entropy can mean confidence without correctness.

## 4. “You, and your tail”

The span mode is the direct implementation of that phrase.

For `tail:j`, the model rotates once at layer `j`, carries the rotated representation through every later decoder layer, and unrotates after the final decoder layer. The tail is treated as one movable object rather than a bag of independently perturbed matrices.

## 5. Minimum experiment matrix

Run the same prompts under:

1. Identity `k=0`.
2. Sparse affine sweep such as `even:17`.
3. Full `k=0..256` sweep.
4. Early, middle, late, and several `tail:j` pivots.
5. Shared span rotation versus layerwise rotation.
6. Layerwise depth strides `0`, `1`, `-1`, and a few larger jumps.
7. Multiplicative-plus-translation (`translation_mode=k`) versus multiplier-only (`zero`).

Record whether winning rotations are:

- better than identity on prompt NLL,
- stable across paraphrases,
- stable across continuations from the same document,
- localized to certain layer ranges,
- sticky across adjacent tokens or prompt chunks,
- associated with visibly different but coherent generations.

## 6. Controls worth adding next

The strongest next control is an equally sized bank of random fixed permutations. That distinguishes “coordinate freedom helps” from “this particular affine family supplies useful structure.”

Other controls:

- cyclic shifts only,
- multiplier-only affine maps,
- random block permutations,
- one global 257-block versus repeated fields,
- a learned router over the same affine bank,
- a router over affine plus random banks with equal candidate counts.

## 7. Next architecture steps

### Activation resonance selector

Capture the hidden state at the rotation boundary and score candidate frames through a statistic related to the observed affine resonance geometry. The open design choice is what constitutes an activation neighbor. Plausible first versions include sign patterns, rank bins, vector-quantized codes, or token-to-token cosine neighborhoods.

### Chunk-sticky inference

Choose `k` once per prompt chunk, keep it for several generated tokens, then reopen selection when surprise rises. This avoids a 257-way search every token while allowing the frame to move.

### Headwise Rotoformer

Partition the hidden width by attention head and let heads use distinct rotations. This is more invasive because head dimensions are usually smaller than 257; it may need a smaller prime or a cross-head field.

### Trainable router

Freeze the base model and train only a tiny router that predicts `k` from boundary activations. The brute-force winner becomes the supervision signal. This converts expensive search into one routing pass while retaining the same rotor bank.

### Weight-materialized Rotoformer

For deployment, frequently selected conjugations could be materialized into weight layouts, avoiding runtime gather operations. The search model remains conceptual; the deployed model can compile favored frames.

## 8. A useful success criterion

The first meaningful result is not necessarily a benchmark gain. It is a repeatable answer to:

> Related inputs prefer related rotations, and those preferences are not explained by identity or arbitrary random permutations.

That would make `k` an observable inference coordinate rather than a decorative transformation.
