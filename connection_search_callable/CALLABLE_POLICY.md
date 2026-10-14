# Planner boundary fix and optional existing-callable proposals

## Recorded result, not a new model run

The supplied `events(5).jsonl` / `summary(10).json` run used 32 Gemma/MPS calls,
51,610 prompt tokens and 160 generated tokens. It evaluated four cases for one
complete candidate, remained unsolved, and had one partial refutation and one
revision. The completed candidate returned `[1, 1, 1]` where `[0, 0, 1]` was
required. The reusable library was empty.

Between calls 14 and 15, `ClosingPlanner.admissible` raised `Malformed hole type`
while inspecting an unselected alternative. The accepted construction itself
passes `validate_shape` and type inference. The raw grammar introduced another
function layer beyond the existing type-depth limit; the planner resolved that
alternative outside a per-alternative validation boundary and aborted the whole
completion. The recorded final model response was therefore not responsible for
the malformed alternative.

## Default correction

The production generator omits prerequisites that exceed the DSL's existing
valid-type bound. The closing planner resolves and type-checks alternatives
individually, including alternatives for map-source witnesses. An invalid
alternative no longer poisons other choices. The type-depth limit is unchanged.
No accepted expression is rewritten and no failed mathematical check is waived.

Run with the previous arguments to isolate this correction. The default typed
application policy remains `unrestricted`.

## Optional inference experiment

`--typed-application-policy existing` removes speculative unary apply steps whose
function head is itself an unresolved hole. Instead, apply can select an actual
unary function with the required return type from lexical scope (including pair
projections), the explicit starting library, or completed, scope-compatible graph
components. Its argument can still be a typed hole.

The model can still create new lambdas, compose map/scan/zip/arithmetic/pairs, and
reuse functions after constructing them. Newly reusable functions need not have a
standalone task-correctness proof: the graph labels their evidence as type checking,
and the resulting whole construction still passes through the existing checks.
Callables with multiple parameters remain usable in operators such as scan; this
experiment keeps the proposal grammar's existing unary-apply restriction.

The completion witness builder uses the same policy and component inventory as
the proposer. Its witness never becomes a submitted model answer. The policy is
recorded in prompts, configuration, summary, and a visible report notice.

This is a search restriction, NOT full normalization, an e-graph, semantic
uniqueness, or a completeness result. Useful programs requiring construction of
an unknown application head can be excluded. Already completed components may
contain wrappers. The old policy is retained for a controlled comparison.
No strict runtime argument evaluation, arithmetic, associativity obligation,
output test, work limit or dependency-span condition has changed.

Research context: *NSynC: Normalised Synthesis of Computation*, arXiv:2606.30703,
searches normal forms to address semantic duplication. This implementation does
not reproduce that algorithm or inherit its guarantees or benchmark results.

## Run

Retain the existing working Python/model environment. No new dependency is added.

```bash
unzip connection_search_callable.zip
cd connection_search_callable

python run_connection_search.py run \
  --mode graph \
  --proposal-mode typed \
  --typed-application-policy existing \
  --task affine_prefix \
  --model google/gemma-4-12B-it \
  --device mps \
  --max-calls 32 \
  --token-budget 65536 \
  --max-new-tokens 1024 \
  --out runs/affine_existing_callable
```

Omit `--typed-application-policy existing` for the bug-fix-only behavior. The option
is supported by typed direct and repair controls as well as typed graph search.
The model-free enumerator remains unrestricted; its summary does not claim to use
this typed-only policy. JSON proposals cannot enable the typed-only option.

A source patch against the proof-boundary build can be applied from that build's
root with `patch -p1 < ../connection_search_callable.patch`.

## Validation and limits

- 233 tests passed, one optional Transformers test skipped because Transformers
  is not installed in this environment. All previous 207 tests remain unchanged.
- All 35 uploaded states were rechecked. Their checks, including certificate and
  counterexample evidence, match the recorded results apart from timing.
- The original exception is reproduced against the unmodified package. The
  corrected package offers one next step at the same boundary, with a 16-step
  remaining closure witness after that step; the accepted construction is intact.
- At the recorded call-9 function hole, the optional policy offers a lambda with
  an unfinished body or the already recorded left-projection lambda. The old
  speculative apply is absent. This does not tell us which option Gemma will pick.
- New tests cover every type depth from 0 to 8, invalid alternatives, scoped and
  library callables, map witnesses, callable reuse, policy wiring in all three
  typed controllers, and random construction closure. A known affine solution
  is still reachable through explicitly hand-authored test choices; that is a
  reachability regression, not model discovery.

Reproduce the offline audit without loading a model:

```bash
python validation/callable_policy/audit.py --out audit.json
python -m pytest -q
```

`validation/callable_policy/` includes the before/after audit, clean-patch test
output, and preservation hashes. No fresh Gemma/MPS inference was performed.
There is no demonstrated improvement in solve rate, compute efficiency, or novel
mathematical discovery. This remains a bounded program-synthesis diagnostic for
the larger connection-search hypothesis.
