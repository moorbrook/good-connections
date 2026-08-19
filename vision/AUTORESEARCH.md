# Vision model-search contract

This is a local adaptation of the autoresearch loop: propose one falsifiable model or data-contract
change, train under a fixed budget, evaluate on a fixed distribution, and retain only measured
improvements.

The chronological experiment record is in [`RESEARCH_LOG.md`](RESEARCH_LOG.md). Update it when an
experiment is proposed and again when its result is known; rejected hypotheses are part of the
record, not cleanup noise.

“Model search” is the accurate name. This is not a scaling-law experiment: the run count and model
sizes are far too small to infer a power law or predictable return from parameter count. It is also
not runtime neural architecture search or program synthesis. The outer development loop proposes
ordinary source changes; FrankenTorch trains the resulting fixed policy.

The loop is counterexample-guided in the ordinary engineering sense: clustered validation failures
motivate the next hypothesis. It is only CEGIS-like; no symbolic synthesizer proves or constructs the
policy.

## Objective

Build the fastest learned pixel-only player that makes zero wrong selections on production-rendered
12-card boards and generalizes to unseen procedural renderer seeds and raster sizes. The extension
may not read card labels, application state, network payloads, or use a symbolic Set solver at
runtime.

Rank candidates lexicographically:

1. Fewer `board_failures` on the fixed procedural validation distribution.
2. Higher `triple_accuracy` when board failures tie.
3. Lower measured end-to-end local inference latency when correctness ties.
4. Smaller WASM and simpler code when all prior measurements tie.

The useful search axes are operator choice, topology, spatial inductive bias, representation, input
contract, and training distribution. Width or depth is allowed as a controlled candidate, but “the
larger model won one run” must not be described as a scaling law.

## Shared implementation boundary

The deployed policy is `../vision-policy/src/lib.rs`. Both the native reference evaluator and the
WebAssembly runtime depend on it and load the same FrankenTorch `.ftsv` checkpoint. Never add a
JavaScript inference copy or a second export format.

Fast training currently has a second, training-only expression of that topology in
`../vision-mpsgraph/src/policy_graph.rs`. MPSGraph performs the tensor operations on Apple GPU
hardware; its checkpoint is converted into the same FrankenTorch state dictionary used everywhere
else. This duplication is a deliberate temporary cost, not a shared implementation. Every topology
change must update both Rust modules and pass the fixed-batch forward/loss/gradient parity probe
before its result is comparable. A future shared declarative topology description would remove this
hazard without adding a second runtime.

The trainer in `src/` owns data loading, balanced triple generation, losses, optimization, and
evaluation. Training flags may change the optimizer recipe, but an accepted experiment must record
them. Rejected auxiliary inference branches and alternative checkpoint formats should be removed,
not left behind as dormant switches.

## Fixed board-v4 protocol

The current research evaluator uses `../vision-artifacts/validation-v2-atlases`, with renderer seeds
9001–9008 and widths 104, 120, 152, 192, 216, 272, 344, and 376 across all four themes. Its aggregate
digest is `f883b4c2f509fd49c4d831bf44e3ad809c6ad1c50f031a58555a4c77d73c0d68`. It evaluates
32,768 balanced triples and 4,096 12-card boards with fixed RNG seeds. The reported board metric
scores the policy's highest-logit triple, which is the deployed action.

Board-v3 is historical. Its sealed holdout was exposed by the first candidate that cleared its
research split, so neither that holdout nor its seven failures may guide board-v4 model selection.

Training sampling is uniform over theme, then source-raster bucket, then atlas within that bucket.
This matters: sampling all files uniformly overweights any bucket that happens to contain more
procedural seeds. The evaluator samples its individual atlases uniformly.

During one run, do not edit:

- `src/data.rs`, `src/config.rs`, or the evaluator in `src/main.rs`;
- `../src/vision/Atlas.tsx`, the production `Card` renderer, or preprocessing contract;
- validation or final-holdout atlas files; or
- `../tools/run-vision-mps-experiment.sh`.

Changing one of those between runs creates a new explicitly named protocol or input-distribution
hypothesis. Do not silently compare metrics from different protocols.

## One experiment

From the repository root:

```sh
tools/run-vision-mps-experiment.sh <tag> "<one falsifiable hypothesis>"
```

The runner compiles first, trains for exactly 20,000 optimizer steps (1,280,000 sampled triples),
captures noisy output in an artifact log, evaluates 4,096 fixed boards, and appends to
`vision-artifacts/autoresearch-v2/results-board-v4.tsv`. `keep` means the candidate beats earlier
results under the lexicographic research metric; it does not mean the checkpoint is ready for
deployment. Training seconds and steps per second are recorded as diagnostics. The fixed step budget
keeps examples seen comparable while topology throughput changes; correctness, not elapsed time, is
the research selection metric. Candidates tied on correctness need an explicit browser/WASM
benchmark before latency can decide promotion.

After each experiment:

1. Read the summary and inspect failure records by theme, seed, width, and attribute.
2. Decide whether evidence points to topology, representation, input coverage, or optimization.
3. Change one thing and state why it should address that failure mode.
4. Keep historical logs and the append-only result row; do not keep dead runtime branches.
5. Rebuild the extension only after a checkpoint passes promotion gates.

Use names that describe the actual intervention, such as `width-balanced` or `local-refinement-conv`.
Do not call a convolutional policy a transformer, a training loop program synthesis, or a sequence of
small candidates a scaling study.

## Promotion and final holdout

A research candidate is promotable only after zero failures on the fixed 4,096-board evaluator.
Then generate a fresh final-holdout directory with seeds and source widths absent from both training
and research validation. Evaluate 10,000 boards exactly once. Do not use those results to select a
follow-up model; doing so turns the final holdout into another validation set.

The historical board-v3 holdout is `../vision-artifacts/final-multiraster-atlases`. It is no longer a
sealed selection-independent test after its one permitted evaluation. Board-v4 has a new sealed
holdout in `../vision-artifacts/final-v2-atlases`, rendered from seeds 10001–10004 at widths 112, 181,
299, and 367 across all themes. Its aggregate digest is
`b5ba3578ee54eb3c0eebdbdd85cd7180ad34e3a5b10c8cea66614a1915450a52`. Representative atlases
from every theme and all four widths were visually audited before the first board-v4 research result.
Evaluate it exactly once only after zero failures on board-v4 research validation.

For a promoted checkpoint, record:

- exact training and evaluation distributions;
- board failures and denominator;
- triple and per-attribute accuracy;
- direct FrankenTorch/WASM cached and pixel-transfer latency;
- generated WASM size; and
- responsive browser-path cases.

Finite testing supports a statement such as “0 failures in 10,000 boards from this sealed renderer
distribution.” It never proves universal 100% reliability.

## Board-v5 robustness cycle (locked before results)

Board-v4's sealed holdout is exposed and cannot be reused. Board-v5 tests capture-path robustness
without changing the 41,918-parameter specialist topology. Training remains 20,000 MPSGraph
steps, batch size 64, no augmentation, and the existing `vision-artifacts/atlases` corpus.

### Independent training replications

Run these eight predeclared `(data seed, initialization seed)` pairs. The first pair reproduces the
former hard-coded recipe; the other seven measure seed sensitivity rather than seed lottery.

| Run | Data seed | Initialization seed |
|---:|---:|---:|
| 1 | 99005 | 11400714819323198485 |
| 2 | 190001 | 6364136223846793005 |
| 3 | 290003 | 1442695040888963407 |
| 4 | 390007 | 3202034522624059733 |
| 5 | 490009 | 3935559000370003845 |
| 6 | 590011 | 2862933555777941757 |
| 7 | 690019 | 7046029254386353131 |
| 8 | 790021 | 9650029242287828579 |

Every checkpoint is evaluated on the same board-v5 research corpus with 4,096 boards and 512
balanced triple batches. Rank recipes by fewest board failures, then highest triple accuracy, then
lowest run number. Only the selected checkpoint may receive one board-v5 sealed evaluation.

### Fresh capture-shift corpora

The research directory is `vision-artifacts/board-v5-research-atlases` (240 PNGs):

- seeds 11001–11002, all four themes, CSS widths 96, 137, 229, and 353, profiles `clean`, `blur`,
  `crop`, `shadow`, `lowcontrast`, and JPEG quality 52, DPR 1 (192 PNGs); and
- the same seeds/themes, widths 113 and 287, profiles `clean`, `crop`, and `jpeg`, true Chrome DPR
  2 screenshots (48 PNGs).

The sealed directory is `vision-artifacts/board-v5-sealed-atlases` (240 PNGs), rendered and hashed
before any training result but never passed to an evaluator until selection:

- seeds 12001–12002, all themes, CSS widths 104, 173, 257, and 371, all six profiles, DPR 1
  (192 PNGs); and
- the same seeds/themes, widths 121 and 299, profiles `blur`, `shadow`, and `jpeg`, DPR 2
  (48 PNGs).

The locked sorted per-file aggregate SHA-256 digests are:

- research: `fb43f82a14ff3507267adf7ed754136b0fdc55ec5a30222cfe62cf36095625ac`;
- sealed: `5caf2e34172ac3171c74426c874524968181644c5be404738f7dd26103fe8fac`.

The loader derives physical padding and gaps from the explicit `-dpr2` filename token. Visual QA
must cover every profile, theme, width extreme, and DPR before hashes are recorded. File count,
sorted per-file SHA-256 aggregate, and image dimensions are immutable after that point.

### Abstention and closed-loop gate

Each research board records relation score, top-one/top-two relation margin, normalized mean
attribute entropy, weakest selected-attribute probability margin, and correctness. Any fitted guard
uses only the selected checkpoint's board-v5 research samples. It must report accepted coverage and
accepted errors; the sealed data cannot choose coefficients or thresholds.

The browser also requires two captures whose selected triples agree and whose card pixels are
stable. After clicking, it clears its highlight and requires two stable captures showing a
material pixel-level board change. An unchanged/stale frame, unstable animation, disagreement, or
guard rejection stops autoplay.

### Architecture decision

Do not change width, depth, resolution, or branches merely because another candidate might score
better. First cluster board-v5 counterexamples by profile, theme, raster, DPR, and failed attribute.
Change architecture only if those counterexamples identify a repeatable representational limit.
If the unchanged topology satisfies the locked action and abstention gates, retaining it is the
required outcome for this cycle.

### Declared follow-up after eight replications (before candidate result)

All eight unchanged-topology runs failed systematically under scale shifts. The locked control is
replicate 5 (data seed 490009, initialization seed 3935559000370003845) with 487 failures in 4,096
research boards. Across runs, crop/zoom-in caused 253–322 failures and shadow/scale-down caused
179–279; clean and blur caused at most two each. This repeatable bidirectional scale sensitivity is
evidence against the exact flattened specialist representation, not a single unlucky seed.

Test one targeted candidate named `specialist-spatial-plus-global`: for every attribute specialist,
retain the existing flattened 8×6×8 feature map and concatenate its eight channel-wise global
means before the 24-wide projection. Everything else—training corpus, selected control seeds,
20,000 steps, batch size, optimizer, relation scorer, input resolution, and evaluator—remains
fixed. This adds 192 projection weights per head (768 total) and no new learned layer.

Keep the candidate only if it has fewer than 487 research board failures, or ties at 487 with
higher triple accuracy than 0.921326. Otherwise reject it and retain replicate 5. The winner then
receives research-only abstention calibration and the cycle's one sealed evaluation. The sealed
corpus remains evaluator-unopened while this rule is recorded.

### Board-v5 result (sealed corpus now exposed)

The `specialist-spatial-plus-global` candidate passed backend parity but was rejected by the locked
research rule: it produced 586 failures, board accuracy 0.856934, triple accuracy 0.911224, and
attribute accuracy 0.974116. The unchanged 41,918-parameter topology and replicate 5 checkpoint
therefore remained selected.

The guard was fitted only to replicate 5's 4,096 research records. Its frozen linear logit uses
`[score, score_margin, -attribute_entropy, attribute_margin]`; the threshold is one numerical epsilon
above the largest incorrect research logit. It accepted 1,016 research boards (24.80%) with zero
accepted errors. This is a finite empirical cutoff, not a calibrated probability.

The sealed digest was reverified as
`5caf2e34172ac3171c74426c874524968181644c5be404738f7dd26103fe8fac` immediately before its single
evaluation. On 10,000 sealed boards, the selected checkpoint achieved board accuracy 0.891500,
triple accuracy 0.923113, and attribute accuracy 0.980396: 1,085 raw failures. The frozen guard
accepted 2,462 boards (24.62%) with 4 accepted failures, for 0.998375 accepted accuracy. Those four
failures occurred under crop or shadow profiles. The board-v5 sealed corpus is now exposed and must
not tune coefficients, thresholds, architecture, or future candidate selection.
