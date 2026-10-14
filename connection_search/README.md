# Executable Connection Search

A runnable first implementation of discovery-oriented inference: a frozen local model proposes **typed, executable connections**; an external controller retains alternatives, composes compatible subexpressions, checks algebraic obligations, executes task tests, and revises explicit decision points.

This is not another mechanism-description generator. The old frame/recall/lexical-overlap orchestration is not used. The original `Generation`, `Backend`, and `HuggingFaceBackend` classes are retained verbatim, with source hashes in `PROVENANCE.json` and a regression test that checks those hashes. The supplied original file is not modified.

**What has and has not been demonstrated:** the executable kernel, search-controller paths, CLI, accounting, and model-free controls have been run here. The replay demonstration uses hand-authored responses and is conspicuously labeled synthetic. No Hugging Face model checkpoint was available in the build environment, so this bundle does not contain a real-model benchmark or evidence that graph inference outperforms sampling. There is no hidden-state intervention, fine-tuning, retrieval during discovery, Lean integration, or claim of historical novelty.

## Run it in your existing model environment

Extract the ZIP and enter the directory containing this README and `run_connection_search.py`. Keep the working Transformers/PyTorch environment from the original experiment; no package upgrade is necessary merely to run this bundle.

```bash
cd connection_search

# Dependency-free controller/algebra regression; NOT model discovery.
python run_connection_search.py demo --out runs/controller_demo

# Real graph search, preserving the original Gemma/MPS loader settings.
python run_connection_search.py run \
  --mode graph \
  --task affine_prefix \
  --model google/gemma-4-12B-it \
  --device mps \
  --max-calls 32 \
  --token-budget 65536 \
  --max-new-tokens 1024 \
  --out runs/affine_graph
```

`python -m connection_search ...` is equivalent when run from this directory. No API key or external service is required by the controller. Model files must already be locally available or loadable by the original Hugging Face backend, including any model access requirements. The controller itself does not retrieve outside knowledge.

The kernel, enumeration, audit and replay commands use only the Python standard library. For a fresh environment rather than your already-working one, optional dependency groups are provided:

```bash
python -m pip install -e '.[inference,test]'
```

This installs model dependencies without claiming that arbitrary new dependency versions have been tested with your particular checkpoint. The minimum Python version is 3.10. For the existing environment, running directly from the extracted directory avoids installation and import-path ambiguity.

## Compare against actual controls

```bash
python run_connection_search.py compare \
  --task affine_prefix \
  --modes enum,direct,repair,graph \
  --model google/gemma-4-12B-it --device mps \
  --max-calls 32 --token-budget 65536 --max-new-tokens 1024 \
  --out runs/affine_comparison
```

`direct` samples independent complete programs and checks them. `repair` generates a complete program and supplies the previous program plus checker feedback on the next fresh call. `enum` performs model-free, type-directed bounded enumeration with exact polynomial-equivalence pruning. `graph` maintains the persistent typed frontier, measured completion trials, structural revision, and cross-branch bridges.

All methods receive the same task specification, small public examples, primitive catalogue and initial reusable-function library. `compare` loads the model once, shuffles method order reproducibly, and clones the same starting library for every run. One method's discoveries are **not** silently given to later methods. The first search-test-passing solution is frozen before the holdout is opened; a holdout failure does not resume search.

For a task/seed matrix:

```bash
python run_connection_search.py compare \
  --tasks prefix_sum,prefix_square_sum,affine_prefix,affine_square_coefficient,affine_suffix \
  --seeds 1729,42,2718 \
  --model google/gemma-4-12B-it --device mps \
  --max-calls 32 --token-budget 65536 \
  --out runs/composition_matrix
```

The optional matrix is explicit because those budgets apply **per task, per seed, per method**. `comparison.json` reports outcomes and observed costs, not an automatic scientific conclusion.

Probe component competence separately:

```bash
python run_connection_search.py probe \
  --model google/gemma-4-12B-it --device mps \
  --max-calls 4 --token-budget 16384 \
  --out runs/component_probes
```

The five probes test addition, multiplication, pair projection, mapping a square, and zipping inputs. They help distinguish elementary DSL/component failures from composition failures. They are not automatically inserted as extra solution hints into the comparison.

## The executable representation

An expression is a scalar literal, a bound name, or a JSON operator array. The types are scalars `R`, pairs, sequences and functions. Arithmetic is exact modulo the modulus supplied by each case; floats and division are intentionally absent.

For example, this is a *known-answer diagnostic*, not a newly discovered result:

```json
["scan", ["fn", [["a", "R"], ["b", "R"]], ["add", "a", "b"]], "xs"]
```

It computes inclusive prefix sums. General primitives are `add`, `sub`, `mul`, `pair`, `fst`, `snd`, `zip`, `reverse`, `fn`, `apply`, `map`, `scan`, and `accumulate`. There is no built-in affine-recurrence solution, task-name dispatch in the searcher, answer lookup in the model backend, or hand-written affine transition offered in real search prompts.

`["hole", TYPE]` is an unresolved prerequisite. Replacing an output hole with a typed operation and new input holes is backward construction. Previously generated hole-free subexpressions provide the forward frontier. Matching a component to a hole requires matching its type and in-scope free variables; the resulting construction is checked again.

`scan` evaluates an inclusive Hillis–Steele-style prefix circuit with ordered earlier-left/later-right arguments. Its associativity obligation must normalize to an exact integer polynomial identity on scalar/pair carriers. That identity implies the same law for modular instances. A failure can produce a replayable numeric counterexample; unsupported proof forms or resource exhaustion remain unproved, not “verified.”

The interpreter executes the circuit serially in Python. Its recorded **dependency span** describes the evaluated circuit; it is not measured GPU speedup. The staged scan performs logarithmically many rounds and can use O(n log n) work. `accumulate` is a general sequential prefix fold; a recurrence that depends on its preceding state can satisfy the output tests and still fail the task's span constraint.

The full grammar is in `connection_search/dsl.py` and is supplied consistently to every model-based method. Generated host-language code is never executed: there is no `eval`, `exec`, generated import, subprocess, filesystem operation or network operation in the expression interpreter. Expression depth/size, integer literal size, symbolic expansion and execution work are bounded. This is an intentionally restricted language, not a general Python sandbox or universal theorem prover.

## How inference differs from repeated full answers

Each call starts from a fresh context using the original backend. The controller explicitly supplies the immutable task and the selected construction; it never relies on a generated chat transcript or cross-call KV cache.

Initial frame proposals are independent. After that, a state is a typed partial or complete program, not a paragraph. The controller selects a hole or an expression cut, requests a replacement of the corresponding type, and preserves everything outside that path. A local failed scan law supplies a specific repair location. An ordinary output counterexample does **not** identify a responsible line; nonlocal cuts remain exploratory and are labeled that way.

A bounded completion trial asks the model to fill every remaining hole without changing existing syntax. Its success is determined by the checker. A beta-distribution-based frontier policy allocates attempts using those measured trial outcomes, with an explicit exploration fraction and a visitation penalty. This is a search heuristic, **not an exact Sequential Monte Carlo implementation or an unbiased posterior sampler**.

Type-compatible expressions from other branches are actually substituted and executed by the bridge operation, not merely listed as inspiring concepts. Graph nodes are content-addressed *expression templates*, not runtime tensor/value nodes; free-variable interfaces and lambda bindings remain explicit. Operator applications are hyperedges whose ordered tails are jointly required. Revisions create immutable alternative roots, sharing unchanged expression nodes. A non-associative function can still be a valid ordinary function: rejection of its scan obligation does not erase it as a generally usable component.

The controller does not reward lexical novelty or low model probability. It does not claim that the entire model's knowledge is a directly addressable graph.

## Verification and held-out evaluation

The checker separates these outcomes:

- Typed partial construction: holes remain; no claim of executed task correctness.
- Algebraic-law evidence: exact polynomial identity, concrete law counterexample, or an unproved/unsupported obligation.
- Whole-program evidence: passed search tests, failed output/constraint tests, and the separate final holdout audit.

The task oracle is independently written ordinary-loop Python in `tasks.py`; it is not the candidate AST or the candidate's own explanation. Its source is not included in prompts. Search uses 28 deterministic cases. The final audit uses 64 different cases, including different moduli, composite rings, negatives, empty sequences and longer inputs. These are reproducible, inspectable local holdouts, **not cryptographically secret tests or contamination-resistant research benchmarks**.

Built-in compositional tasks include prefix sums/products, prefix sums of squares, shifted sums, affine recurrence prefixes, squared-coefficient affine recurrences and suffix-direction recurrences. The latter variations change the task specification, rather than merely swapping numbers into an unchanged answer.

Passing the finite task tests is not a proof of universal program correctness. Polynomial law certificates are relative to this small normalizer, not independently kernel-checked Lean certificates. New task domains require an independently authored specification/reference evaluator and any additional primitive verification rules; arbitrary natural-language problems are not silently treated as verified executable tasks.

## Reusable compositions

A successful program can yield closed, parameterized functions in `library.json`. The source body, types and provenance are retained. Evidence distinguishes “source program passed its tests” from “this subexpression has an independent goal proof”; the latter is not inferred automatically.

Reuse is explicit:

```bash
python run_connection_search.py run \
  --task affine_square_coefficient --mode graph \
  --library-in runs/affine_graph/library.json \
  --model google/gemma-4-12B-it --device mps \
  --out runs/affine_transfer
```

The syntax `["use","lib_ID"]` transparently expands to the stored `fn` before type checking, graph construction and execution. It does not introduce an opaque capability. Library imports recheck identities, closedness, types and supported scan obligations; source evidence is provenance, not trusted truth. For a fair comparison, give the same `--library-in` to the whole `compare` command.

## Accounting and ablations

Every generation is logged, including invalid JSON, failed candidates and completion trials. An exact preflight token count reserves prompt plus maximum output tokens before a real call. There are no characters-per-token estimates in the real backend. Repeated `_encode` preflight overhead is separately timed.

Use `--max-calls`, `--token-budget`, `--model-seconds` and `--wall-seconds` to bound experiments. MPS/CUDA calls are synchronized around timing outside the preserved backend. Time caps are checked between indivisible calls and can overshoot by one call; token caps reserve the maximum output in advance. Equal token budgets are **not** claimed to be equal FLOPs. Reports expose actual prompt/output tokens, model-call time, verification time and end-to-end time. Selection stops at the first search-test pass; compare cost-to-success rather than pretending each method consumed exactly the same compute.

Enumeration has explicit type, size, bucket, candidate, expansion and CPU-time limits. Its constants are -1, 0 and 1; its carrier set contains task carriers plus scalar pairs. It uses polynomial normal forms to avoid redundant algebraic expressions. It does not exhaust the entire language; truncation and pruning statistics are reported.

Ablations are `--no-backward`, `--no-revision`, `--no-reuse`, and `--rollouts 0`. `--no-reuse` disables new cross-branch bridging while keeping any explicitly supplied initial library constant. `--no-backward` requires hole-free replacements. No ablation silently changes the immutable task or reference implementation.

## Outputs and regression tests

Every completed run writes `report.html`, `summary.json`, `events.jsonl`, `states.json`, `graph.json`, `graph.dot`, and `library.json`. A solved run also writes `solution.json`. Existing nonempty output directories are refused, never overwritten. The HTML report is local, escapes generated text, requires no remote assets, and exposes the timeline and evidence. `graph.dot` can be rendered with Graphviz; the JSON is the full typed hypergraph.

```bash
python -m pytest -q

# Independently execute a saved program again:
python run_connection_search.py audit \
  --task affine_prefix \
  --program runs/affine_graph/solution.json \
  --out runs/affine_audit
```

The tests cover algebraic normalization, noncommutative scan ordering, rejection of non-associative laws, resource and type failures, preserved lexical scope, immutable requirements, genuine cross-branch composition, focused repair, completion preservation, token accounting, independent contexts, held-out isolation, transparent library reuse, model-free enumeration, CLI invocation, and preservation of the supplied backend source. The optional PyTorch generation-path test uses a deterministic fake model, not a downloaded checkpoint.

`validation/` contains the actual build-time run reports and test output. The affine controller replay intentionally proposes a bad law, observes its counterexample, replaces only the offending function, and executes the repaired construction against both case sets. It is a regression test with known answers, not evidence that a model found the connection.

## Research references and boundaries

The model/evaluator separation is informed by **FunSearch**, which pairs language-model proposals with an automated evaluator. The frontier/partial-trajectory motivation also relates to **Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo**. This implementation does not reproduce that paper's learned twist functions or importance-weighted sampler and does not inherit either project's performance claims.

Primary sources consulted for this build:

- Google DeepMind, *FunSearch: Making new discoveries in mathematical sciences using Large Language Models*: https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/
- Zhao, Brekelmans, Makhzani and Grosse, *Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo* (2024): https://arxiv.org/abs/2404.17546
- Hugging Face, *Generation*: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
- Hugging Face, *Chat templates*: https://huggingface.co/docs/transformers/main/en/chat_templating

The frozen local loader comes from the supplied file, not from replacing it with a new research backend. Soft-embedding or hidden-state inference is deliberately absent from this first build: it should be tested as a separate intervention against this executable baseline, not bundled into a result that cannot identify which change helped.
