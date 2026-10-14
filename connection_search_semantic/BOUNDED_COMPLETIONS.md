# Version 0.1.3: bounded completions and initial-branch scheduling

## What the uploaded run established

The third uploaded run used the real Gemma/MPS backend. It made 32 valid typed
selections, consumed 47,340 prompt tokens and 160 generated tokens, and ended
after 168.64 seconds with no complete candidate and no evaluated task cases.
Twenty-eight calls were made inside a single `complete_typed` invocation. The
other distinct initial branch was retained but never selected for expansion.
The growing hypergraph was therefore not evidence of improved discovery.

The failure moved from syntax handling to controller liveness. A structural
step could replace one hole with several more. The old 32-step completion cap
was as large as the entire run's model-call cap; its first rollout could consume
all remaining calls before frontier selection ran again. Its unfinished attempt
also incremented the completion-failure denominator when the global budget
raised an exception, without emitting the final `completion_evidence` event.

`validation/bounded_completion/uploaded_run_audit.json` records the calculations.
The regression fixture retains the uploaded decisions, not generated substitutes.

## Patch scope

The model loader, prompt encoder, constrained-generation backend, DSL, algebraic
checker, test cases, task reference implementations, graph storage and grammar
productions are unchanged. There is no new model, answer template, alias repair,
auto-filled program, or relaxed span bound.

### 1. Reserve calls for the other initial branches

In typed graph mode, distinct retained initial frames get an expansion turn
before random selection revisits their rollout descendants, provided the frame
is still eligible and resources remain. Initial prompts remain independent.

A rollout with `R` calls remaining and `P` other initial branches still awaiting
that turn gets at most `floor(R/(1+P))` calls, further capped by
`--completion-steps`. In the uploaded configuration, the first allocation would
be 14 calls, not all 28. This is startup fairness, **not** a claim of globally
fair scheduling of every graph node or an exact MCTS/SMC implementation.

### 2. Keep a concrete path to a hole-free result within that allocation

`ClosingPlanner` composes a syntax-only witness using existing grammar steps:
available terminals and bound projections, lambda introduction, pair construction,
zip, and map over an available sequence. A witness records how many additional
model selections can construct one complete extension. It neither looks at the
task's reference answer nor tries to satisfy the task.

Before presenting a rollout choice, the controller checks that the resulting
whole construction has a witness fitting in the remaining call allocation.
If `h` calls remain before the choice, its successor must have a known closure
using at most `h-1` further selections. This accounts for **all** remaining holes,
not just the focused one. A choice cannot repeatedly add obligations while
ignoring the finite remaining budget.

The model still selects every actual operation and terminal. Witness programs
are not submitted, executed as proposals, or inserted as answers. A well-typed
complete candidate may be entirely wrong; the original checker decides that.
Partial expressions remain unchanged outside the selected holes.

The witness procedure is deliberately conservative. Its cost is an upper bound
for a known construction, **not** the shortest possible completion or a lower
bound on all completions. A missing witness means unknown; it does not prove
that the program or the mathematics is impossible. The filter can exclude a
useful longer path or a shorter path the witness procedure cannot recognize.
This changes the bounded proposal distribution and must be considered when
comparing methods. Direct and repair use the same horizon filter; graph alone
has initial-branch scheduling and cross-branch component access.

A feasible rollout reaches a hole-free candidate within its allocated selection
count, unless an earlier checker rejection, global token/time/check cap, backend
failure, or unsupported condition interrupts it. No claim is made that every
candidate runs successfully or passes its tests. Scope/type/size checks remain
active. If no witness fits initially, the controller defers the rollout without
spending model calls. Direct/repair terminate with an explicit horizon-unavailable
stop reason rather than spin in a zero-call retry loop.

### 3. Distinguish observed failures from censored trials

For typed rollouts, an output, constraint or algebraic counterexample updates
the failed-trial count. A successful checked completion updates the success
count. An interrupted or unsupported/unknown result increments the separate
`censored_completions` count and does not become negative mathematical evidence.
The `completion_evidence` event is emitted from `finally`, including on global
budget interruption. Controller deferrals do not blame a stale model response.

JSON mode retains its earlier success-per-generated-attempt accounting; invalid
JSON completion attempts still count as failed attempts. The event text states
that different interpretation explicitly.

## Running

Use the same working local model environment. No inference package upgrade is
needed for this patch. After extracting the ZIP:

```bash
cd connection_search_bounded
python run_connection_search.py run \
  --mode graph \
  --proposal-mode typed \
  --task affine_prefix \
  --model google/gemma-4-12B-it \
  --device mps \
  --max-calls 32 \
  --token-budget 65536 \
  --max-new-tokens 1024 \
  --out runs/affine_bounded
```

For a controlled comparison replace `run --mode graph` with
`compare --modes direct,repair,graph`. Each method has its own budget. Use a new
output directory. No experiment resumes or silently imports a previous library.

The existing `--completion-steps` cap defaults to 32. There is no extra activation
flag beyond `--proposal-mode typed`. JSON remains the CLI default.

Inspect `complete_candidates`, `case_evaluations`, `observed_completion_trials`,
`censored_completions`, `initial_branches_expanded`, and
`completion_horizon_filtered_choices`. New timeline events show allocations,
witness costs, deferred trials and horizon-filtered menu sizes. Node and edge
counts alone are not progress measurements.

## Validation and limits

165 tests passed and one optional Transformers integration test was skipped.
The prior suite had 122 passing tests. Two old tests were revised because they
explicitly expected the now-prevented behavior of spending calls on an
unfinishable short-horizon rollout; 43 new cases test the new contract.

New tests include a policy that deliberately tries to add more holes, twenty
random-policy seeds, witness reachability through the actual typed menus,
budget accounting across all holes, competing initial branches, token-cap
interruption and state preservation. The uploaded rollout reaches its first
counterfactual horizon exclusion at historical call 12 with a 14-call allocation.
That audit does not predict Gemma's next response or assume it repeats the prefix
under changed prompts.

Existing synthetic affine-construction and sequential-span rejection tests still
pass. Those are hand-authored target policies, not discovered solutions. No real
Gemma/MPS run has been performed for this patch. Transformers is absent from the
build environment; its optional integration test remains skipped. This patch
changes no Transformers call or decoding code.

The patch is a liveness and accounting correction, not a new mathematical
capability result. It does not implement a general partial-program semantic
interpreter, prove that graph search beats its controls, or establish discovery
of an unknown result. Shorter completions can still be wrong. The purpose is to
reach checkable outcomes before the budget disappears into unfinished syntax.

## Related research, not validation of this build

ExeDec uses execution subgoals to guide compositional synthesis:
https://deepmind.google/research/publications/49061/

Neural-Guided Deductive Search combines symbolic deductions with neural guidance:
https://www.microsoft.com/en-us/research/blog/neural-guided-deductive-search-best-worlds-approach-program-synthesis/

These are conceptual context only. This frozen-model prototype reproduces
neither trained method and inherits none of their experimental performance claims.
