# 0.1.1: rejected-proposal feedback and bounded retries

## What the uploaded run actually showed

The supplied `affine_prefix` run made 32 model calls, used 30,363 prompt tokens
and 2,129 generated tokens, and recorded approximately 248 seconds of model time.
It ended with the original typed goal hole, no hyperedges, no complete candidate,
and zero evaluated task cases. All 32 transitions were rejected before execution:
29 unbound `bs_` names, two malformed `["x0"]` variable references, and one JSON
parse error. There were only two distinct system/user prompt pairs across all
32 calls. Every adaptive expansion received the accepted root's
`partial_type_checked` result, not the rejected proposal's diagnostic.

These are observations from the uploaded logs, not an inference about general
Gemma ability. The logs do not establish that the discovery hypothesis failed,
because no executable construction reached task testing. They also do not show
the effective model generation configuration; 62–67 output tokens per call are
well below the logged 1,024-token maximum, but near-identical outputs alone do
not establish a decoding or device fault.

## Root cause and patch scope

`Search.ask_expansion` caught `Invalid`, wrote `rejected_transition`, and returned
`None`. Its next call built feedback from `state.check`; the invalid proposal
had never become that state's expression or check. Thus the error was visible
to the user but not to the model. The repair baseline similarly lost format
failures before any `previous` program existed.

The patch retains a separate, bounded rejected-proposal record with the raw
response excerpt, precise error path, relevant type evidence and call index.
The next adaptive call at the same state/focus receives up to three such
records. Accepted-state evidence is not overwritten with evidence about another
expression. Independent initial frames and direct-sampling calls stay independent.
Completion format/contract failures and the repair baseline now receive the
same kind of deliberately supplied diagnostic context.

A default four-attempt **adaptive** non-progress limit applies to each graph
state/focus. Unchanged or already-seen proposals count as no progress. Other
focus points and states remain eligible; successful new constructions reset
the corresponding counter. `proposal_stalled` is a budget-policy stop, not a
proof that any expression is impossible. No retry is free or unlogged.

The DSL guide explicitly specifies exact name spelling and callback types.
The type checker now reports in-scope names and expected/actual `accumulate`
callback types without accepting any previously invalid program. `zip` produces
pair-valued elements: the callback consumes `(state, pair)`, not three separate
scalars. Fixing spelling alone cannot fix that signature. A sequential program
with both defects corrected still has to satisfy the original span bound.

Reports expose escaped raw outputs and exact prompts and warn when no task
case was evaluated. The requested temperature is now recorded with each call.

No dependencies were added. The original model-loading and generation classes
are byte-for-byte preserved and remain covered by the provenance test. Task
specifications, reference functions, evaluator implementation, algebraic proof
checks, and held-out isolation were not changed. No aliases, automatic variable
renaming, tuple unpacking, reference-answer injection, or check bypasses were
added. This is not grammar- or type-constrained token decoding.

## Validation and its limits

The original 69-test suite passes unchanged. Three targeted regression cases
fail against the old controller: adaptive rejection feedback and two malformed
output paths in the repair control. With the patch, **87 tests pass**: all 69
existing cases plus 18 new cases. New coverage includes focus-local rejection
memory, actual/expected callback types, bounded stalls, duplicate handling,
independence of controls, budget accounting, completion contracts, and HTML
escaping of model outputs.

The complete uploaded reply stream is included as a verbatim fixture with the
SHA-256 of its source `events.jsonl`. Replaying those **unchanged** replies now
stops after seven synthetic calls: three independent frames plus four adaptive
failed expansions. It remains `not_solved` with zero task cases. This proves
bounded retry behavior, not recovery by the real model.

A separate regression injects hand-authored corrections after checking that each
preceding error appears in the next prompt. It progresses from the bad name to
the bad callback type to an executable sequential program. That program is
correctly rejected by the original dependency-span constraint. These corrected
replies are synthetic test inputs, never proposed by the patched system as a
model discovery.

No new Gemma/MPS inference was executed in this patch environment. The supplied
run is real-model evidence; the patch validation is local unit/replay testing.
Whether Gemma uses the new diagnostics effectively needs a fresh real-model run.
The original release validation files are retained separately from this patch's
new evidence in `validation/feedback_fix/`.

## Run

Extract the full bundle into a fresh directory and use the same working Python
and model environment. Do not overwrite the old run.

```bash
cd connection_search_feedback_fix
python run_connection_search.py run \
  --mode graph --task affine_prefix \
  --model google/gemma-4-12B-it --device mps \
  --max-calls 32 --token-budget 65536 --max-new-tokens 1024 \
  --out runs/affine_graph_feedback_fix
```

`--max-stalled-attempts` defaults to 4; this limit can be explicitly changed.
No new model flags or precision changes are required by this patch. Inspect the
HTML report's model-call sections to see the actual diagnostics delivered.

The full archive includes test and replay logs. The source-only patch can be
applied from the old project root with `git apply connection_search_feedback_fix.patch`;
run `python -m pytest -q` there to regenerate the test result. It includes the
replay script and fixture, but not the generated validation logs.

## External context checked

The diagnosis above comes from the supplied code and logs. External research is
context, not evidence of this patch's performance:

- Mündler et al., *Type-Constrained Code Generation with Language Models*:
  https://arxiv.org/abs/2504.09246. Distinguishes syntax constraints from typing
  constraints, relevant to valid JSON containing invalid variable names or
  callback types. Its decoding algorithm was not integrated into this patch.
- Poesia et al., *Synchromesh: Reliable code generation from pre-trained language
  models*: https://arxiv.org/abs/2201.11227. Studies constrained semantic decoding
  including scope and typing. Again, precedent rather than an implemented feature.
- Hugging Face, *Generation*:
  https://huggingface.co/docs/transformers/main_classes/text_generation.
  Documents sampling controls and output-token limits. This patch records the
  requested temperature but does not change the original generation method or
  claim to capture all inherited generation settings.
