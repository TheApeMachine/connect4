# Typed-choice inference patch

The second uploaded Gemma/MPS run made seven calls, consumed 82.63 seconds, and evaluated no task cases. The feedback patch delivered diagnostics, but the model still emitted malformed JSON, `Fn` instead of `fn`, and occasional unbound names. These failures happened before mathematical evaluation.

The new path changes what inference emits. It selects a short ID for a type-checked structural step. The controller builds the actual expression. The old JSON path remains available and is still the default for backward compatibility. **Use `--proposal-mode typed` to activate the new path.**

## Run in the existing environment

```bash
python run_connection_search.py run \
  --mode graph --proposal-mode typed \
  --task affine_prefix \
  --model google/gemma-4-12B-it --device mps \
  --max-calls 32 --token-budget 65536 --max-new-tokens 1024 \
  --out runs/affine_typed_choices
```

There is no new package requirement beyond the existing PyTorch/Transformers inference environment. The original model loader, prompt encoder, and unconstrained `generate()` method are unchanged. Typed mode uses the same loaded weights and encoder through a separate constrained generation call. Decoder-only text generation is supported, including text generation by a compatible multimodal model. Unsupported generation interfaces fail explicitly; there is no backend, dtype, parsing, or free-text fallback.

For an ingredient probe rather than the recurrence diagnostic, change the task to `scalar_add`. The existing `probe` command also accepts `--proposal-mode typed` and runs the five ingredient tasks. Each task has its own configured budget.

## Mechanism and boundaries

At a focused typed hole, the existing model-free grammar supplies legal one-constructor expansions. Function parameters and their arity are generated from the expected function type. Bound variables and transparent bound-pair projections can be selected directly. Compatible cross-branch components remain available when reuse is enabled. Every option is resolved, shape-checked, type-checked, and checked against the surrounding expression's size/depth limits before it enters the menu.

The menu is derived from the grammar, types, in-scope names, and available components—not the reference evaluator or an affine answer template. Adaptive expansion excludes already-submitted resulting constructions; this does not globally forbid reusing an operator. Initial frames retain independence. Menu ordering is reproducibly shuffled, and the complete ID-to-expression mapping is logged.

The model receives the task, construction, relevant feedback, and menu. A token trie restricts next-token generation to the offered IDs and permits EOS only at a complete ID. The output must round-trip and match one entire allowed token path. Invalid outputs are not silently renamed, repaired, or interpreted as another action. The selected expression then passes through the original checker.

This guarantees neither mathematical correctness nor that a partial expression has a successful completion. Associativity, outputs, modular arithmetic, and dependency span remain independent obligations. The grammar is deliberately bounded: constants -1/0/1, task/context carrier types plus `Pair(R,R)`, unary function-application productions, and the existing operators. It is not all of the supported DSL or an unconstrained search of the model's knowledge. IDs and local token masking also do not implement an exact posterior over programs.

## Search, completion, and controls

Each structural decision is a separate charged model call. Full construction may therefore require many calls. Shorter generated text is not automatically lower total inference cost; prompt prefills still count. Output budget is reserved for the longest offered ID plus EOS, using the actual tokenizer. Menu preparation contributes to end-to-end wall time.

Typed completion fills holes without changing fixed syntax. `--completion-steps` defaults to 32 and bounds decisions in one completion attempt; the global call/token/time limits remain in force. Graph completion trials persist intermediate constructions so budget interruption does not erase the work. Failed algebraic or task checks remain failures. The report records each constrained call, its selected ID, the materialized expression, and evaluated case counts.

`direct` and `repair` accept the same typed proposal mode. They construct complete attempts through the same interface, without graph component feedback. Use the same proposal mode for the model-based methods when comparing graph organization, and compare observed time and total tokens rather than call count alone:

```bash
python run_connection_search.py compare \
  --proposal-mode typed --modes enum,direct,repair,graph \
  --task affine_prefix --model google/gemma-4-12B-it --device mps \
  --max-calls 32 --token-budget 65536 \
  --out runs/affine_typed_comparison
```

Enumeration continues to use its explicit CPU/size/expansion limits. No superiority claim is justified by a single diagnostic task or hand-authored replay.

## Validation actually performed

`validation/typed_choices/pytest.txt` records 122 passing tests and one skipped test. The original 87 tests still pass. The skipped test calls real Transformers `generate()` on a tiny, randomly initialized GPT-2; it requires Transformers, which was unavailable in the build environment. Package installation was also unavailable there. It remains in the suite and does not download weights.

Tests cover trie prefixes, terminal EOS, malformed tokenizers, invalid-backend refusal, callback signatures, lexical scope, immutable task specifications, CLI wiring, duplicate exclusion, randomized legal construction walks, budget accounting, and interrupted-partial retention. A deterministic stand-in exercises the HF call boundary with an illegal token assigned the highest raw logit; this is a contract test, not a real HF/Gemma run.

Two explicit hand-authored target policies exercise the typed graph/controller. One constructs the known affine-scan program through separate legal choices and passes 28 search and 64 held-out cases. The other constructs a sequential recurrence and is rejected by the original span constraint. These test reachability and checking, not discovery or pretrained model selection. The target policies live in tests/validation only.

`validation/typed_choices/uploaded_call5_audit.json` separately records a manual diagnostic changing only the fifth uploaded proposal's `Fn` to `fn`. The resulting sequential program executes, but for n=129 its recorded span is 260 against the original limit of 108. That edit is not in the production parser and is not counted as a successful model output.

To reproduce the local checks:

```bash
python -m pytest -q -rs
python validation/typed_choices/reproduce.py
```

**No Gemma checkpoint or MPS execution was performed in the build environment.** The next real run must establish whether the checkpoint chooses useful legal steps. This patch removes the observed serialization burden; it does not establish search efficacy.

## External references

Hugging Face documents `prefix_allowed_tokens_fn` and `PrefixConstrainedLogitsProcessor`, which are the mechanisms used by this constrained path:

https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.PrefixConstrainedLogitsProcessor

Mündler et al., *Type-Constrained Code Generation with Language Models* (2025), motivates distinguishing syntactic constraints from typing constraints. This implementation is a finite typed-action-menu method, not a reproduction of that paper's general prefix-automata algorithm:

https://arxiv.org/abs/2504.09246
