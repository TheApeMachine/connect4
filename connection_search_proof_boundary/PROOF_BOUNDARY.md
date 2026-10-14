# Proof-construction boundary and global call-horizon fix

This build addresses `events(4).jsonl` / `summary(9).json`. The uploaded run was
not solved: 32 real model calls, zero complete candidates, zero concrete task
cases, one partial refutation, and one revision. It is not evidence that graph
search improves discovery. This build has not been run with the user's Gemma/MPS
checkpoint.

## The reproduced failures

1. The partial-output check stopped a state-preserving recurrence at call six.
   That part of the previous change operated as intended.
2. An 18-selection completion stopped at call 26 with `law_unknown`, because
   the polynomial checker tried to interpret `zip` while constructing a callback.
   The sequence arguments in this particular callback are unused: its result is
   the left projection. An unsupported *construction argument* is different from
   an unsupported *algebraic result*.
3. Calls 27–32 took place outside the completion-loop budget guard. The original
   syntax-only witness procedure estimated 8, 10, 7, 9, 9 and 14 additional calls
   after these selections; only 5, 4, 3, 2, 1 and 0 remained. Those estimates are
   witnesses, not proven minimum costs. They suffice to show the inconsistent
   application of the existing budget policy, not mathematical impossibility.

## Changes

### Callback laws versus callback construction

`algebra.py` can carry non-algebraic construction values as opaque placeholders
while it evaluates higher-order functions to polynomial results. Opaque values
are never serialized as polynomial certificates; if the result actually depends
on an unsupported value, the law remains unknown. Type and complete-function
checks run before issuing an associativity certificate. A certificate explicitly
states that the original strict construction must evaluate successfully.

The runtime program is never rewritten. Strict argument evaluation, zip length
preconditions, runtime work limits, nested scan-law obligations, task outputs,
and span limits remain enforced on the original AST. In particular, an invalid
zip or a non-associative nested scan is not bypassed just because a surrounding
function ignores its result. The enumerator retains its original strict
polynomial-normalization policy; this change is opt-in inside the law checker.

This is an extension of a bounded polynomial checker, not a verified general
normalization-by-evaluation implementation. Background: Abel and Chapman,
*Normalization by Evaluation in the Delay Monad*, arXiv:1406.2059.

### One budget policy at both decision boundaries

Typed graph expansions and revisions now use the existing closure-witness
predicate, not just bounded completion steps. The candidate's own selection
costs one call. Unclosable partial states are retained but deferred for the
current call horizon. The witness procedure is conservative: it can miss a
possible closure. A deferral is neither a proof of impossibility nor a negative
mathematical trial.

When graph reuse is enabled, known hole-free components participate in the
witness calculation with the same one-selection cost as their offered menu
entries. Costs are refreshed when the graph grows. Direct/repair paths without
graph access do not receive those components. Plans are never submitted as model
answers. Token and time limits can still interrupt a call-feasible continuation.

### Checker limitations receive repair opportunities

A `law_unknown` result gets a first revision turn and is presented as a
`checker_limitation`, not a refutation. Its completion remains censored rather
than increasing the measured mathematical-failure count. This preserves the
original uncertainty distinction while making it actionable.

The HTML report names the new proof and global-horizon policies. It displays
outside-rollout filtered-choice and horizon-deferred-state counts.

## Validation

- 207 tests passed; one optional Transformers integration test was skipped.
- 22 added regression cases, including 3,610 concrete associativity comparisons
  of sequence-wrapped callbacks across moduli 2, 3 and 5.
- Two prior tests changed because they explicitly required the old behavior:
  rejection of an unused sequence capture, and creation of an over-budget
  initial shell. The latter still tests zero-call deferral of a pre-existing
  partial state. No output/span requirement was changed.
- All 59 uploaded states were checked. The existing partial refutation remains;
  three old `law_unknown` states become partial, not solved. No recorded program
  was rewritten.
- A synthetic controller replay reproduces the first 26 observed operation
  selections and then supplies two explicitly hand-authored `bs` selections.
  It reaches one complete candidate and five concrete case evaluations. That
  candidate fails. This tests the previously blocked execution route, not new
  model behavior, better solve rates, or original discovery.

Run the dependency-free audit with:

```bash
python validation/proof_boundary/run_audit.py
```

Run the regression suite (pytest required) with:

```bash
python -m pytest -q
```

Fresh run, in the same working model environment:

```bash
python run_connection_search.py run \
  --mode graph \
  --proposal-mode typed \
  --task affine_prefix \
  --model google/gemma-4-12B-it \
  --device mps \
  --max-calls 32 \
  --token-budget 65536 \
  --max-new-tokens 1024 \
  --out runs/affine_proof_boundary
```

No package upgrade or new feature flag is needed. JSON proposal mode remains the
CLI default for compatibility; keep `--proposal-mode typed` for this experiment.
The unchanged loader, constrained decoder, interpreter, task generator, concrete
verifier and enumeration grammar are recorded in
`validation/proof_boundary/source_integrity.json`.

## Remaining limitation

Fixing rejection, budget handling and proof coverage does not establish useful
model guidance. This prototype still asks the model to select small syntax
steps. The uploaded trace spends many decisions on functions that discard their
arguments. This patch does not solve that search-quality problem, implement a
normal-form synthesis grammar, or establish a gain over direct/repair controls.
