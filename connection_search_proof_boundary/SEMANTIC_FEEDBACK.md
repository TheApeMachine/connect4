# Semantic feedback patch

This build addresses the uploaded bounded run (`events(3).jsonl`, `summary(8).json`). That run produced six complete candidates and 22 concrete case evaluations in 32 model calls, but no solution and no revision calls. Failure evidence remained attached to particular states; subsequent calls on other ancestors did not receive it.

## Changes

`connection_search/feasibility.py` performs conservative abstract execution of partial programs against the same public examples already included in prompts. Scalars are sets of residues, pairs preserve separate fields, and sequences have either exact-length elements or an unknown-length element enclosure. Known functions preserve lexical scope; unknown functions return the full output domain. Unknown-length folds use a post-fixpoint over possible states, and distinguish reachable initial states from emitted post-update states. Scans follow the same ordered prefix circuit as the concrete interpreter.

If the enclosure excludes a required public output, no well-typed filling of the remaining holes can satisfy that example without changing a fixed decision. The status is `partial_output_counterexample`. Membership in an enclosure means UNKNOWN, not a successful test. The analysis does not localize a unique guilty choice, infer a correct replacement, certify whole programs, or assess novelty.

The analyzer supports residue moduli 2 through 17 and uses an operation limit. Unsupported moduli, recursion limits and exhausted analysis budgets return unknown. Correlations are deliberately approximated, so some impossible constructions remain in the search. The analyzer is not mechanically proved sound; its enclosure behavior is checked by differential tests against the original interpreter. Concrete verification, including larger/composite moduli and dependency span, is unchanged.

`Search.submit` applies the analysis only to type-checked typed-mode partial programs. Direct and repair typed completion paths receive the same public-example check. It is enabled by default in typed mode; `--no-partial-pruning` disables it. JSON proposals keep the original concrete checker path. No hidden or held-out cases are fed to partial analysis.

Adaptive graph prompts now include up to two recent observed failures, each paired with its exact source program and checker evidence. A failure does not declare every shared component invalid. Initial independent frames and the independent direct baseline do not receive other attempts' observations. The typed inference wrapper retains the search instruction rather than replacing it with output-format instructions.

A demonstrated failure receives a first revision turn after pending initial branch turns, alternating reserved repair turns with ordinary selection. A pending first repair preempts a different branch's long rollout; that partial child is retained. A refuted partial program is revised at fixed syntax, not expanded at another hole. Fixed leaves are also revisable. `--no-revision` prevents revisiting refuted partial constructions as well as failed complete ones. Frontier and global budget limits still apply.

## Evidence and accounting

The offline audit reevaluates the uploaded expressions, not Gemma. Its first refutation is the state reached after call 6: the callback has the form `x0 * HOLE`. In the fourth public example, `x0 = -4 = 0 (mod 2)`, so every emitted value must be zero, regardless of the remaining scalar, initial-state and sequence holes. The required output is `[0, 0, 1]`.

The uploaded run later completed that branch and found the same output contradiction. The new analyzer can reject it before those additional syntax choices. A different prompt/controller can change all later model choices; this audit is not a prediction of the new model trajectory.

Abstract example evaluations and refutations are reported separately from concrete `case_evaluations`. A partial refutation is conclusive negative evidence for the sampled completion attempt; an unfinished or unknown attempt remains censored. Neither creates a solution or increments concrete passed-case counts. Reports expose partial-analysis cost, revision calls and calls containing observed failures.

## Validation

Run `python -m pytest -q`. The suite includes the original tests (with two liveness assertions extended to recognize sound early refutations), 8,000 scalar concrete executions checked against one-hole abstractions, composite/higher-order/scan differential tests, hole-erased known-correct affine constructions, public-only analysis, limit handling, feedback propagation, and repair scheduling.

Run `python validation/semantic_feedback/run_diagnostics.py NEW_OUTPUT_DIRECTORY` for an offline historical-state audit and two SYNTHETIC controller runs. The synthetic policy replays the first six recorded choices and then chooses randomly from legal options. No solution template guides its later choices. Both synthetic outcomes remain unsolved. The on/off comparison holds the new feedback/scheduling code fixed and toggles only partial pruning; it is not a model-performance comparison. Its token units are UTF-8 bytes, not pretrained-tokenizer counts.

`validation/semantic_feedback/` contains recorded results. The optional real-Transformers test is skipped when that package is unavailable. This build was not run with the user's Gemma/MPS checkpoint.

## Run in the existing model environment

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
  --out runs/affine_semantic_feedback
```

No new package is required. The model loader, constrained-token implementation, grammar, algebraic law checker, concrete interpreter, task definitions and concrete verifier are unchanged. `compare` supports the same flag for all typed methods; do not compare a pruned graph method against unpruned controls without labeling that difference.

## Research context, not validation of this implementation

Abstraction-based synthesis pruning encloses possible outputs of incomplete programs and rejects a candidate only when the desired output is outside that enclosure. See Johnson, Krishnan, Reps and D'Antoni, *Automating Pruning in Top-Down Enumeration for Program Synthesis Problems with Monotonic Semantics* (2024), https://arxiv.org/abs/2408.15822. This small manually defined residue abstraction is not a reproduction of their automated general-purpose framework.

See also Yoon, Lee and Yi, *Inductive Program Synthesis via Iterative Forward-Backward Abstract Interpretation* (PLDI 2023), https://arxiv.org/abs/2304.10768. This patch does not implement their full forward-backward method.
