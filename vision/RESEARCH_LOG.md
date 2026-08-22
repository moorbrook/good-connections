# Vision policy research log

This is the chronological evidence record for the local visual Set player. The governing protocol,
promotion gates, and terminology are in [`AUTORESEARCH.md`](AUTORESEARCH.md). Metrics below come from
the fixed `board-v3` research evaluator unless explicitly labeled otherwise.

## 2026-08-18 — Baseline and responsive-renderer diagnosis

The first compact FrankenTorch policy worked on its fixed training atlas but failed to establish
that it generalized to the responsive production renderer. A real browser screenshot-to-click
harness was added so a passing tensor evaluator could not hide crop, resampling, card-order, WASM,
or click-integration errors.

The learned runtime was also consolidated. Native evaluation and WASM now call the same
`vision-policy` implementation and load the same FTSV checkpoint. This removed a hand-written WASM
copy of the convolution, GELU, softmax, and checkpoint parser. The policy is named honestly: it is a
compact convolutional visual policy, not a transformer, VLM, general VLA, program synthesizer, or
neural architecture search system.

## 2026-08-18 — Multi-raster training

**Hypothesis.** Training across five source raster widths will cover responsive-renderer aliasing
without increasing the 40,414-parameter policy.

**Change.** Procedural atlases were generated at multiple raster widths. The topology and training
budget remained fixed.

**Result.** Kept as an improvement: 4 failures in 4,096 boards, triple accuracy 0.996277, attribute
accuracy 0.999443.

**Interpretation.** Raster diversity helped but did not meet the zero-failure research gate.

## 2026-08-18 — Width-balanced sampling

**Hypothesis.** Uniform file sampling overweights the 190 px reference bucket because it contains
more seeds. Sampling uniformly by theme and then raster bucket should remove that accidental bias.

**Change.** The training sampler now selects theme, source-raster bucket, and atlas within the bucket
uniformly. Model size and optimization budget remained fixed.

**Result.** Kept as an improvement: 2 failures in 4,096 boards, triple accuracy 0.996582, attribute
accuracy 0.999425.

**Counterexample.** Both remaining failures came from `signal-7002-w248`; card 77's empty fill was
misread as striped.

**Interpretation.** The data distribution, not just network capacity, was a material part of the
model. The remaining errors clustered around an unseen rasterization condition.

## 2026-08-18 — Global-mean feature branch

**Hypothesis.** Concatenating global-average convolution features with the spatial embedding will
add translation-insensitive fill evidence and eliminate the residual empty-versus-striped errors.

**Change.** A global-mean feature branch was added while retaining the local spatial path.

**Result.** Rejected and removed: 7 failures in 4,096 boards, despite higher triple accuracy
(0.997620) and attribute accuracy (0.999616).

**Interpretation.** Aggregate component metrics were misleading for the deployed action. The top
ranked board choice regressed, so the branch failed the primary objective and was reverted rather
than retained as optional code.

## 2026-08-18 — Dense raster coverage

**Hypothesis.** Adding 240 px and 260 px source-raster buckets will cover the aliasing neighborhood
around the unseen 248 px counterexample and eliminate the last failures without adding parameters.

**Controlled change.** Added 136 procedural atlases: four themes, seeds 0–16, and the two new raster
buckets. Training still samples uniformly by theme and raster bucket. The original 40,414-parameter
topology, seed 99005, 20,000-step budget, batch size 64, attribute-loss weight 4, and no-augmentation
setting are unchanged. The 248 px research-validation width remains unseen during training.

**Result.** Rejected: 5 failures in 4,096 boards, triple accuracy 0.994995, attribute accuracy
0.999217. Failures clustered in the `atelier-7004` and `nocturne-7004` atlases at widths 136, 176,
and 208. Count remained perfect; the wrong card attributes were color, shape, or fill.

**Interpretation.** More raster buckets did not monotonically improve the model. Under the fixed
20,000-step budget, expanding from five to seven uniformly sampled buckets also reduced expected
exposure per existing bucket, so “coverage increased” and “exposure was diluted” were coupled. The
result rejects this complete training recipe; it does not establish that nearby raster examples are
intrinsically harmful. Visual inspection confirmed that the failing cards are correctly rendered,
with errors concentrated in thin outline/color or small-shape evidence after downsampling.

The sealed final holdout was not evaluated. Its recorded aggregate digest remains
`1447840cb924741abc5f0a9ede5654a17d5fe9d697173f0eb8c6375cbc8ca16c`.

## 2026-08-18 — Attribute-specialist branches

**Hypothesis.** A single shared embedding creates avoidable interference between visual tasks:
counting, color estimation, shape recognition, and fill-frequency recognition. Sharing an
inexpensive first convolution but giving each attribute its own second convolution, projection, and
three-class head should reduce those errors at approximately the same parameter scale.

**Controlled change.** Keep the dense-raster corpus, fixed evaluator, seed, training budget, relation
scorer, and input resolution unchanged. Replace only the shared upper encoder with four specialist
branches. The relation remains learned; do not replace it with the symbolic modulo-3 rule. Compare
the result directly with the 5-failure dense-raster candidate before comparing it with the overall
2-failure incumbent.

The implemented network shares a 3→12, 5×5 stride-2 convolution. Each attribute then has its own
12→8, 3×3 stride-2 convolution, 384→24 projection, and 24→3 classifier. It has 41,918 trainable
parameters, 3.7% more than the 40,414-parameter shared model. This is an attribute-specialist
multi-branch network, not an ensemble: the branches predict complementary factors rather than
voting on the same target.

Each branch receives direct three-class attribute supervision. It also receives the set-validity
gradient through its softmax probabilities and the differentiable learned relation scorer. The
shared first convolution accumulates gradients from all four branches. There is no hard AND or XOR
gate in training or inference.

**Result.** Kept as a research improvement: 2 failures in 4,096 boards, triple accuracy 0.998352,
attribute accuracy 0.999725. This ties the best board-failure count while improving triple accuracy.
It does not pass the zero-failure promotion gate.

**Counterexamples.** Both wrong actions depend on a striped card predicted as empty:
`classic-7001-w136` card 52 and `signal-7002-w176` card 4. Count and color were perfect across the
attribute evaluator; shape accuracy was 0.999736 and fill accuracy was 0.999166.

**Interpretation.** Specialization removed the dense-raster model's color errors and most shape
errors, supporting the interference hypothesis, but thin fill texture remains the limiting visual
signal at small source rasters. The next candidate should target input fidelity or fill-frequency
representation rather than add generic branch capacity. The sealed final holdout was not evaluated.

**Throughput observation.** On a 15-core Mac, the trainer exposes 16 threads but sustains roughly
650–800% CPU during this run. The four specialist calls are mathematically independent but execute
sequentially because each mutates the same FrankenTorch session/autograd tape. If specialization is
retained, test an equivalent fused 12→32 convolution followed by four channel groups and grouped or
batched projections. Require output/gradient equivalence before measuring throughput; do not change
execution shape during the controlled architecture run.

This measurement is CPU-only: the current strict FrankenTorch session uses CPU `f64` tensors and
does not dispatch to Metal, Core ML, or the Apple Neural Engine. Any future accelerator export is a
second execution backend and must pass numerical, board-action, and browser/native integration
equivalence against the canonical FrankenTorch policy.

## Future hypotheses — teacher/student and pretraining

Do not mix these into an active controlled run.

**Procedural high-capacity teacher.** Train a larger, higher-resolution visual policy on the same
unlimited procedural labels, establish its error ceiling, and distill soft per-attribute outputs,
intermediate features, or cross-raster consistency into the small FrankenTorch student. The teacher
is training-only and never enters the browser bundle. Because the generator already supplies exact
hard labels, this experiment is justified only if the teacher contributes smoother invariances or
demonstrably better high-resolution perception—not merely another source of labels.

**Generic pretrained backbone.** A natural-image backbone may reduce sample requirements, but the
current synthetic geometry task has unlimited exact labels and a severe size/portability objective.
Evaluate pretraining only as a separately measured candidate, including conversion and WASM costs.
It is more promising for later robot vision, where real-image variation and scarce action labels
make learning the entire visual representation from scratch substantially harder.

**General VLM teacher.** Direct screenshot reasoning is useful as a feasibility baseline, not an
assumed oracle. Before distillation, measure grounding, attribute, calibration, and repeatability
errors. A stochastic model that occasionally names or points to the wrong card cannot be treated as
clean supervision merely because it is larger.

## Future diagnostic — representation audit

If behavioral failures persist, inspect learned geometry with controlled factorial tests rather than
interpreting raw high-dimensional weights or relying on a single 2D projection. For every attribute
head, measure confusion and calibration by theme/raster, invariance when the other three attributes
change, cross-raster consistency for the same card, activation and gradient distributions, singular
spectra, and similarity between branches. PCA or UMAP may help communicate a result, but promotion
must rest on exact behavioral and geometric measurements because a projection can conceal relevant
directions.

## 2026-08-18 — MPSGraph training accelerator

**Scope.** Accelerate architecture-search training only. FrankenTorch remains the independent
reference evaluator and current WASM inference runtime. An accelerated checkpoint must reproduce
its card logits and board actions after weight-layout conversion; MPSGraph is not being added to the
browser.

**Backend decision.** Use Apple's MPSGraph directly through the Rust `objc2` bindings rather than
adding Burn. This keeps the second implementation close to the platform graph compiler and avoids a
larger fast-moving framework layer.

**Smoke result.** A Rust `f32` graph ran on the Apple M5 Pro Metal device and used MPSGraph automatic
differentiation. For `mean([1,2,3,4]^2)`, it returned loss 7.5 and gradient `[0.5,1.0,1.5,2.0]`,
matching the analytic result.

**Implemented seam.** The procedural sampler and tensor dimensions now live in lightweight shared
Rust crates. The MPSGraph trainer does not compile or call FrankenTorch for training, but exports
through FrankenTorch's native state-dict serializer after transposing linear weights and flattening
convolution biases. The browser and reference evaluator still consume the canonical FrankenTorch
checkpoint.

**End-to-end benchmark.** The 41,918-parameter specialist policy trains at 287–303 complete steps
per second on the Apple M5 Pro. This timing includes fresh procedural sampling and augmentation,
HWC-`f64` to NCHW-`f32` conversion, CPU-to-device feeds, the full forward/backward graph, reading all
41,918 gradients, and CPU AdamW. The prior FrankenTorch CPU run took approximately 25 minutes for
20,000 steps (about 13.3 steps/s), so the measured acceleration is approximately 22–23×. At the
observed steady-state rate, the same 20,000-step budget projects to 66–70 seconds. Cached graph-only
forward/backward was about 2.2 ms, but that smaller number is not used for the speedup claim.

**Numerical parity.** After one MPSGraph AdamW step and checkpoint conversion, the identical fixed
batch produced loss 5.165518 in MPSGraph and 5.165518324 in FrankenTorch. The first eight shared
convolution-weight gradients agreed within roughly `2e-8`; intermediate activations and the final
set logit agreed to approximately `1e-6`. This validates convolution, exact-erf GELU, linear layout,
loss reduction, autodiff, and checkpoint conversion for the tested batch.

**Harness correction.** An initial comparison falsely appeared to show a large backend mismatch.
The MPSGraph probe used seed 99005 while the FrankenTorch probe accidentally reused the trainer's
default `0x5e7_2026`. Weight values matched, but the input pixel sums exposed that the examples were
different. The probe now creates its own explicit seed. No activation change was retained on the
basis of the invalid comparison.

**Lifetime correction.** The first 20,000-step attempt omitted a per-step Objective-C autorelease
pool. Temporary graph feeds and results accumulated until Metal ran out of memory at step 16,300.
That run produced no checkpoint and is invalid. Wrapping each complete MPSGraph step in an
`objc2` autorelease pool eliminated the warnings and memory failure. A subsequent 20,000-step
augmentation run completed in 68.215 seconds at 293.19 steps/s.

## 2026-08-18 — Accelerated specialist training results

**Augmentation variant.** The first complete accelerated run used procedural contrast, brightness,
and per-pixel noise augmentation. This differs from the incumbent's controlled no-augmentation
recipe and is recorded as a separate optimization trial. It produced 3 failures in 4,096 research
boards, triple accuracy 0.998596, and attribute accuracy 0.999723. It is rejected against the
2-failure incumbent. The failures spanned Nocturne, Classic, and Atelier and were not a uniform
improvement over the known thin-fill errors.

**Controlled no-augmentation candidate.** Added an explicit `--no-augment` option to the MPSGraph
trainer and reran the exact 20,000-step, batch-64, seed-99005, attribute-loss-weight-4 recipe. It
completed in 59.008 seconds at 338.94 steps/s, versus approximately 1,500 seconds for the previous
FrankenTorch CPU run: about 25.4× faster end to end. Procedural augmentation accounts for much of
the difference between this result and the 68.215-second augmented run.

On the locked research evaluator, the controlled candidate achieved 0 failures in 4,096 boards,
triple accuracy 0.999146, and attribute accuracy 0.999921. Per-attribute accuracy was count 1.0,
color 1.0, shape 1.0, and fill 0.999685. This is the first candidate to pass the research promotion
gate.

**One-time sealed holdout.** Before evaluation, the 48 files in
`final-multiraster-atlases` reproduced the recorded aggregate digest
`1447840cb924741abc5f0a9ede5654a17d5fe9d697173f0eb8c6375cbc8ca16c`. The candidate was then
evaluated exactly once on 10,000 fixed boards. It produced 7 failures, board accuracy 0.999300,
triple accuracy 0.998474, and attribute accuracy 0.999758. Count, color, and shape were perfect in
the attribute sample; fill accuracy was 0.999034. All seven board failures occurred in Nocturne
atlases and involved card 62's fill being predicted as value 1 rather than its actual value 2.

**Decision.** Do not promote this checkpoint to the browser model. The sealed result is recorded as
the final assessment of this candidate and must not be reused to select or tune a follow-up model.
A future search cycle requires a newly declared research question and, before another final claim,
a new sealed holdout whose construction is independent of these seven examples.

## Workflow observation

The deployed policy is Software 2.0: behavior learned into weights. The surrounding language-driven
research loop is Software 3.0 used to produce that Software 2.0 artifact. Procedural generation,
rule-based oracles, compilation, and browser checks remain ordinary Software 1.0. The useful system
is the combination, not a claim that one layer replaces the others.

## 2026-08-18 — Board-v4 independent cycle and MPS baseline

**Protocol reset.** The board-v3 sealed holdout was exposed by its one permitted evaluation, so its
examples and seven failures are excluded from further model selection. Before any board-v4 result,
the renderer produced a new research corpus of 256 atlases: seeds 9001–9008, widths 104, 120, 152,
192, 216, 272, 344, and 376, and all four themes. Its sorted per-file aggregate SHA-256 is
`f883b4c2f509fd49c4d831bf44e3ad809c6ad1c50f031a58555a4c77d73c0d68`. Representative atlases
covering all themes and the extreme widths were visually inspected.

A new sealed holdout was also rendered before training: 64 atlases from seeds 10001–10004, widths
112, 181, 299, and 367, and all four themes. Its aggregate SHA-256 is
`b5ba3578ee54eb3c0eebdbdd85cd7180ad34e3a5b10c8cea66614a1915450a52`. Representative atlases
from every theme and width were visually inspected. No evaluator had opened this corpus when the
research result below was produced.

**Baseline hypothesis.** The unchanged four-specialist policy may already generalize to the wider,
independently seeded board-v4 research distribution; measuring it first prevents later architecture
work from claiming gains without a baseline.

**Result.** `baseline-specialist-clean` trained with MPSGraph for 20,000 steps (1,280,000 sampled
triples) in 60.908 seconds, or 328.36 steps/second. On 4,096 fixed board-v4 research boards it
produced 0 failures, board accuracy 1.0, triple accuracy 0.999359, and attribute accuracy 0.999870.
Relative to the approximately 25-minute FrankenTorch CPU run, this is about 24.6× faster in the
training phase. The checkpoint passed the research gate and is eligible for its one sealed board-v4
evaluation; this entry was written before that evaluation.

**One-time sealed board-v4 result.** Immediately before evaluation, the 64 holdout files reproduced
the recorded aggregate digest. The eligible checkpoint was evaluated exactly once on 10,000 fixed
boards and 80,000 balanced triples. It produced 0 board failures, board accuracy 1.0, triple accuracy
0.999975, and attribute accuracy 1.0 for count, color, shape, and fill. Evaluation took 53.310
seconds.

**Decision.** Promote this exact checkpoint to the browser runtime. The finite claim is “0 failures
in 10,000 boards from the sealed board-v4 renderer distribution,” not universal proof of 100%
reliability. Do not use the now-exposed final-v2 corpus for later tuning.

**Browser promotion verification.** The promoted model and candidate checkpoint both have SHA-256
`2bb1925f14ff2bee008e2614339abc384e9fdbd49e76768ed0786a3d04d13dfe`. The rebuilt WASM passed
12/12 screenshot-to-click integration cases across four themes and 390, 600, and 900-pixel
viewports. A 1,000-iteration Node/WASM benchmark measured 2.416 ms per cached 12-card inference and
2.508 ms including pixel transfer. The generated WASM is 2.5 MiB and has SHA-256
`1f7df07298c982113d940b3d589d86a9b2bd9d7da0732800d2c11ede4879b4a8`.

For later robotic grasping work, preserve the same experimental structure but do not assume the toy
game's reliability transfers. Robot promotion gates must add calibrated uncertainty, distribution
shift tests, temporal and control latency, collision constraints, safe fallback behavior, and
hardware-in-the-loop evaluation before any autonomous action.

## 2026-08-18 — Board-v5 capture robustness, replications, and abstention

**Protocol reset and corpora.** Board-v5 evaluates capture-path shifts absent from board-v4. The
research and sealed corpora each contain 240 production-rendered atlases spanning four themes,
unseen seeds and widths, DPR 1/2, and mixtures of clean, blur, crop/zoom-in, shadow/scale-down, low
contrast, and JPEG quality-52 captures. Their locked sorted per-file digests are respectively
`fb43f82a14ff3507267adf7ed754136b0fdc55ec5a30222cfe62cf36095625ac` and
`5caf2e34172ac3171c74426c874524968181644c5be404738f7dd26103fe8fac`. Representative atlases from
all profiles and themes, both width extremes, and DPR 2 were visually inspected before evaluation.

**Reproducibility.** MPS training now has independent data and initialization seeds. A one-step
smoke produced byte-identical checkpoints for identical seed pairs and different hashes after only
the initialization seed changed. The former implicit seeds remain the defaults. Eight predeclared
20,000-step replications of the unchanged 41,918-parameter topology produced:

| Run | Board accuracy | Failures / 4,096 | Triple accuracy | Attribute accuracy | Train seconds |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.874268 | 515 | 0.924530 | 0.977661 | 57.237 |
| 2 | 0.845459 | 633 | 0.901337 | 0.970856 | 65.339 |
| 3 | 0.845703 | 632 | 0.895599 | 0.971133 | 69.095 |
| 4 | 0.871826 | 525 | 0.912659 | 0.977918 | 71.960 |
| 5 | 0.881104 | 487 | 0.921326 | 0.979599 | 71.902 |
| 6 | 0.856689 | 587 | 0.908173 | 0.975329 | 73.316 |
| 7 | 0.853027 | 602 | 0.906555 | 0.977918 | 65.607 |
| 8 | 0.879150 | 495 | 0.926544 | 0.978541 | 70.054 |

The locked selection rule chose run 5 by fewest failures. Run 1 exactly reproduced the formerly
promoted checkpoint hash, confirming that the lower robustness was caused by the harder evaluator,
not an accidental recipe change. Failures were stable across seeds: crop caused 253–322 per run and
shadow caused 179–279, while clean and blur caused at most two. This is a repeatable scale-shift
limit, not a seed lottery.

**Targeted architecture candidate.** The predeclared `specialist-spatial-plus-global` candidate
concatenated eight channel means to each specialist's flattened 384-wide spatial feature. It added
768 weights and no layer. MPSGraph and FrankenTorch agreed on the one-step probe loss (`5.351852`)
and gradients before the full run. The candidate then regressed to 586 failures, board accuracy
0.856934, triple accuracy 0.911224, and attribute accuracy 0.974116. It was rejected and removed.
Global means do not preserve the missing scale-normalized spatial evidence.

**Frozen abstention model.** Every board records relation score and runner-up margin plus normalized
selected-attribute entropy and weakest attribute probability margin. A four-feature logistic model
was fitted only on run 5's research records, and its threshold was fixed just above the largest
incorrect research logit. It accepted 1,016/4,096 research boards (24.80%) with zero accepted errors.
Repeated-frame choice agreement and pixel stability are separate mandatory gates.

**One-time sealed result.** The sealed digest was reverified immediately before its only evaluator
run. Raw results over 10,000 boards were 1,085 failures, board accuracy 0.891500, triple accuracy
0.923113, and attribute accuracy 0.980396. The frozen guard accepted 2,462 boards (24.62%) with four
errors: accepted accuracy 0.998375. All four accepted errors were in crop or shadow profiles. No
coefficient or threshold was changed afterward; board-v5 sealed data is now exposed and retired.

**Closed-loop browser result.** The extension takes two observations 90 ms apart, requires identical
actions and stable pixels, clears highlights before clicking, and then requires two stable captures
showing a material board change. An unchanged or unstable board stops autoplay. Across 12 responsive
cases (four themes by three viewports), all 12 proposed triples were valid; four passed the frozen
guard and completed click plus transition verification, while eight safely abstained. The embedded
checkpoint SHA-256 is `9f062411a540d5067bda8e1d13cf48256b8cde5a336083cb11287e51dab7293e`.
A 1,000-iteration benchmark measured 2.502 ms cached inference and 2.767 ms with pixel transfer.

**Next research question.** The priority is explicit scale robustness, not another generic width or
depth increase. Candidate inputs include scale-normalized card segmentation or a small multi-scale
specialist path. Any follow-up needs a newly generated sealed corpus; these board-v5 sealed examples
may diagnose the completed cycle but may not select or tune the next model.

## 2026-08-18 — Board-v6 scale-robustness architecture proposals

**Reason for a new cycle.** Board-v5 exposed a repeatable learned-representation failure under both
zoom-in/crop and scale-down/shadow profiles. Its sealed examples are retired from model selection.
Before any board-v6 evaluation, the next cycle fixes new research seeds 13001–13002, sealed seeds
14001–14002, disjoint source widths, the same production capture profiles, and an evaluator-unopened
sealed corpus. Before the first evaluation, both corpora were rendered at the locked 240-image
count and full-deck samples were visually inspected across every profile, theme, width extreme, and
DPR. The research digest is
`fe8bbd0f73643b20a36b93852d54a27c4f300e9b61f06ea9fbf36a3a865f34f8`; the still-unopened sealed
digest is `6a7509f981d401f4b0f02830965193c1772d2e6a01c78d516f7e950ecdf8b09a`.

**Control.** Evaluate the unchanged replicate-5 checkpoint on 4,096 new research boards. Candidate
training remains fixed at data seed 490009, initialization seed 3935559000370003845, 20,000 steps,
batch size 64, no augmentation, and the existing optimizer/loss recipe. Rank by research board
failures at 100% raw action coverage, triple accuracy, then parameter count. Guard scores and
abstention are excluded. Only a zero-failure research winner may see the new 10,000-board sealed
corpus.

**Experiment 1 — `specialist-spatial-pyramid`.** Preserve each specialist's full 8×6×8 local map
and concatenate a non-overlapping 2×2 average-pooled summary before projection. Hypothesis: coarse
quadrant evidence gives the learned projection enough context to tolerate scale changes while the
unchanged local map preserves stripe and outline detail. This is learned feature aggregation, not
attention or foreground segmentation.

**Experiment 2 — `specialist-dual-kernel`.** Replace each specialist's eight-channel 3×3 filter bank
with parallel four-channel 3×3 and four-channel 5×5 banks and concatenate their maps. Hypothesis:
the two receptive-field sizes let each attribute head retain fine fill evidence and wider
shape/count evidence under bidirectional scale shifts. This is a small multi-scale convolutional
branch, not a transformer or an automated architecture search system.

**Parity-probe correction.** An initial diagnostic compared MPSGraph's seed-490009 probe images to
FrankenTorch's fixed seed-99005 images, creating a false loss/gradient mismatch. No full training or
board-v6 candidate evaluation had run. Repeating a 5×5 diagnostic with seed 99005 on both sides
matched loss (5.102340 versus 5.102339833) and gradients, identifying the harness input—not the new
operator—as the cause. Experiment 1 remains the originally declared spatial-pyramid candidate.

**Raw control result.** On the locked board-v6 research corpus, the unchanged checkpoint produced
540 failures in 4,096 boards: raw board accuracy 0.868164, triple accuracy 0.894562, and attribute
accuracy 0.972188. Profile failures were clean 5, blur 0, crop 243, shadow 259, low contrast 32, and
JPEG 1. This confirms on new seeds that scale-shift profiles dominate the raw error budget.

**Experiment 1 result.** `specialist-spatial-pyramid` passed the corrected parity probe at 44,990
parameters (5.642668 versus 5.642668109 loss with matching gradients). Its 20,000-step run took
111.105 seconds at 180.01 steps/s. It regressed to 552 raw board failures, board accuracy 0.865234,
triple accuracy 0.900391, and attribute accuracy 0.969198. Low-contrast failures fell from 32 to 10,
but shadow rose from 259 to 277 and crop was effectively flat at 244. Reject and remove the branch.

**Experiment 2 result.** `specialist-dual-kernel` passed parity at the same 44,990 parameters
(5.180300 versus 5.180300408 loss with matching gradients). Training took 166.221 seconds at 120.32
steps/s. It achieved 517 raw board failures, board accuracy 0.873779, triple accuracy 0.910400, and
attribute accuracy 0.974388. Shadow improved from 259 to 229 failures, but crop worsened from 243 to
258. It wins the research ranking by 23 failures but remains far from the zero-failure promotion
gate.

**Decision.** Do not open board-v6 sealed data, fit an abstention guard, or deploy either candidate.
Restore the shipped single-kernel source so it continues to match the browser checkpoint. Preserve
the research artifacts and tracked `BOARD_V6_RESULTS.tsv`. The dual-kernel result is evidence for
multi-scale features, but one seed and a 0.56-point absolute raw gain are not enough to adopt a
slower policy or claim a general architectural law.

## 2026-08-19 — Board-v7 adversarial raw-perception corpus locked

Board-v7 is a fresh capture-shift protocol targeting at least 99% raw board accuracy at 100% coverage,
not guarded accuracy. The v6 audit identified five untested capture blind spots: severe defocus,
card-content tilt, partial occlusion, compressed colour gamut, and compound blur/crop/shadow/contrast
loss. The hidden atlas route now supports literal `defocus`, `tilt`, `occlusion`, `gamut`, and `compound`
profiles, plus source widths down to 72px so the locked 83px raster can probe sub-96px behavior. The
occlusion band is translucent and leaves symbol and stripe evidence above and below it; it is intended
to remain human-solvable rather than erase the card identity.

The research corpus is `board-v7-research-atlases`: seeds 15001–15003, all four themes, DPR 1 widths
83/157/293 with all 11 profiles, and DPR 2 widths 107/277 with clean/crop/shadow/tilt/occlusion. It
contains 516 atlases with sorted per-file aggregate SHA-256
`fea260de85f3f5221675162812cd3ed4201fd084d40bb47dcb89c4467d421edc`. The sealed corpus is the same
matrix with seeds 16001–16003 and digest
`46c56adb4aa2e8aeb4e4ab55f59b02670a056c8c9232402de5b0d802bc8e7dc6`. Both corpora passed the locked
filename, count, dimension, and visual checks; the sealed corpus was not passed to an evaluator.

The raw selection run is fixed at 1,024 triple batches and 8,192 boards, evaluated as three
sequential 172-atlas shards with exact per-seed allocation. The initial memory justification for
sharding was wrong: it used the 190x140 source-card dimensions, while `vision-contract` fixes the
stored model input at 30x21. This was caught before candidate selection was complete. The shard
allocation remains the locked protocol because it was fixed before the first result and balances
all three procedural seeds. The allocations are 342/341/341 triple batches and
2,731/2,731/2,730 boards, with explicit triple and board seeds recorded in
`BOARD_V7_PROTOCOL.md`; `tools/evaluate-board-v7-research-shards.sh` checks the corpus digest and
aggregates exactly 65,536 triples and 8,192 boards. Rank by raw board failures, then raw triple
accuracy, then parameter count. Only a candidate at or above 99% research accuracy (at most 81
failures in 8,192 boards) may receive the one sealed 10,000-board evaluation. Do not fit or apply
abstention during this cycle. Full protocol details are in `BOARD_V7_PROTOCOL.md`.

**Training corpus correction.** A first draft accidentally reused DPR2 width 121 from an older
sealed cycle. This was caught before training. Those 40 files were discarded and replaced with
width 139. The final board-v7 training corpus has 344 atlases, procedural seeds 17001–17002, DPR1
widths 97/211/349, DPR2 widths 139/251, and digest
`9e14889fccaadb7c326052884d7f13f58240fdc99439e6337c53048b98f21c64`. It has no seed or width
overlap with board-v7 research or sealed corpora.

**Architecture and augmentation search.** Raw control accuracy was 0.822632 (1,453/8,192
failures). Adaptive pooling regressed to 0.744141. A 44,254-parameter specialist with a dilated
residual convolution improved only after the training distribution matched the capture shifts:
profile-corpus training plus geometric, photometric, and horizontal-band augmentation reached
0.965332 (284 failures). The horizontal band specifically corrected the earlier edge-occlusion
augmentation mismatch. All reported policies make an unconditional highest-score action; no guard
or abstention score enters these results.

**Independent-replica ensemble search.** Two complementary checkpoints reached 0.987427 raw
accuracy at weights 0.55/0.45. Two further profile-trained replicas were weaker alone (0.948975 and
0.948120), but their failures were sufficiently different to help at low weight. The predeclared
four-checkpoint bracket produced 85 failures at weights 0.495/0.405/0.05/0.05, 83 at
0.4675/0.3825/0.075/0.075, and 76 at 0.44/0.36/0.10/0.10. The last result is 0.990723 raw accuracy
at 100% coverage, clearing the research gate by five failures. It was frozen as the sole research
winner. Complete candidate results are in `BOARD_V7_RESULTS.tsv`.

**One-time sealed assessment.** Before evaluation, the four weights and three sealed RNG/allocation
rows were fixed in `BOARD_V7_PROTOCOL.md`. Two preflight invocations produced no predictions: one
stopped because the corpus path was absent, and one stopped because the directory symlink was not
traversed. After the helper was corrected and the empty header-only output was moved intact to
`/tmp/good-connections-board-v7-sealed-preflight-incomplete`, the single scored sealed run produced
122 failures in 10,000 boards: raw accuracy 0.987800 and triple accuracy 0.986267. This misses the
99% raw goal. The research-to-sealed gap is evidence of seed-level generalization error and
research-selection overfit. Board-v7 sealed is now exposed and retired; it must not be used to tune
weights or select the next model. Board-v8 must lock fresh procedural research and sealed seeds
before further selection.

## 2026-08-19 — Board-v8 adversarial raw-perception protocol locked

Board-v8 fixes a new, evaluator-unopened research/sealed pair for the same
99% raw board-accuracy goal at 100% coverage. It keeps the v7 eleven-profile
matrix for controlled profile comparisons while moving to disjoint source
widths `79/167/323` at DPR 1 and `115/281` at DPR 2. The research seeds are
`18001–18003`; the sealed seeds are `19001–19003`. These seeds and widths were
checked against the inspected v5/v6/v7 corpus filenames, including v7
training, research, and sealed matrices. The occlusion profile remains a
translucent, human-solvable band rather than an information-erasure test.

Both v8 corpora contain 516 atlases, 172 per seed, with all four themes. The
research digest is
`4c5fcba428b7a48be8b1ab69ea6c9ac72395b2b59d1827d06fe3025cee94e62d`; the
sealed digest is
`9dfbbc84fc8e97c65f9070e43528be11c66073fa9e4b360ffabedd9077bfbe67`. Native
full-deck QA covered every theme, every new profile, both width extremes, both
DPRs, and both seed ranges. Every sample retained the 9x9 grid and intended
capture evidence. No v8 model evaluation has run; the sealed corpus remains
prediction-unopened.

The raw research allocation is three sequential 172-atlas shards: 342/341/341
triple batches and 2,731/2,731/2,730 boards, totaling 65,536 triples and 8,192
boards. The locked triple RNG seeds are `287145488695021–023` and board RNG
seeds are `294222009835245–247`. Four predeclared independent replicas must be
combined with uniform 0.25 weights; no research-tuned ensemble weighting or
abstention guard is permitted. Only an aggregate with at most 81 raw failures
may receive the one-time sealed 10,000-board assessment, allocated
3,334/3,333/3,333 across sealed seeds with board RNG seeds
`304222009835245–247`. Full details are in `BOARD_V8_PROTOCOL.md`.

**Training corpus lock.** Before v8 training, a separate 688-atlas procedural
corpus was rendered with seeds `20001–20004`. DPR-1 uses widths `89/193/347`
and all eleven profiles; DPR-2 uses widths `127/263` and the five-profile
subset `clean/crop/shadow/tilt/occlusion`. Its seeds and widths do not overlap
the locked v8 research or sealed corpora. The corpus verifier passed its
counts, filename contract, themes, profiles, DPRs, widths, and seeds. Native
inspection of harsh occlusion and compound atlases confirmed intact grids and
human-readable evidence. The pre-training digest is
`a8b5c19760547bc262a0946b2f15b31c15c39a5dc5feeb42124d69d61e5437e4`.

**Frozen v7 control on v8 research.** The unmodified four-checkpoint v7 winner
with its already frozen `0.44/0.36/0.10/0.10` weights scored 8,098/8,192 raw
boards: 98.8525391% accuracy with 94 failures. Triple accuracy was
64,646/65,536 (98.6419678%). This is a control measurement, not a selectable
v8 candidate, and it misses the v8 raw research gate by 13 failures. No guard
or abstention altered a prediction, and the v8 sealed corpus remained
evaluator-unopened.

**Four v8 replicas trained.** The four RNG pairs were selected from the unused
replica-seed list before v8 training began, although the exact values were not
committed to this log until after training. Each run used the locked 688-atlas
training corpus, 20,000 MPSGraph steps, batches of 64 triples, the same
44,254-parameter research policy, and the predeclared fixed uniform ensemble:

| Replica | Data seed | Initialization seed | Seconds | Checkpoint SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 99005 | 11400714819323198485 | 141.951 | `085071f86e53f422ba980eb1ee17f2436ef7db3ed31e8f96a7577d830bbfe900` |
| 2 | 190001 | 6364136223846793005 | 140.883 | `3b27f0dc61b12d92f6261c2279c238a403856eb47025c5cbab22b8dd4e413ea8` |
| 3 | 290003 | 1442695040888963407 | 141.310 | `101ab10e81103d7fcb9557c5d661346d6ad7cbaaa7e5f42f354081dfb4256312` |
| 4 | 390007 | 3202034522624059733 | 142.281 | `6aeb7c7b39b6f1455285bc6812165ac51d797c308560fcc15b0e664d4d121e1c` |

The first exported checkpoint passed a strict FrankenTorch state-dict load,
forward-loss, and backward-gradient probe. The locked research evaluator also
strict-loads all four checkpoints, so no permissive key adapter or alternate
inference implementation enters the score.

**Raw v8 research result.** The sole predeclared selectable combination, the
four new replicas at fixed `0.25/0.25/0.25/0.25` weights, scored 8,151/8,192
raw boards: 99.4995117% accuracy with 41 failures. Triple accuracy was
65,211/65,536 (99.5040894%). It clears the at-most-81-failure research gate by
40 failures and is frozen as the sole v8 research winner. The one-time sealed
run therefore becomes eligible. Before that run, sealed triple RNG seeds
`317145488695021–023` were added to `BOARD_V8_PROTOCOL.md`; the board seeds,
allocation, uniform weights, checkpoint hashes, and sealed corpus digest were
already fixed. No abstention or confidence guard enters either score.

**One-time sealed v8 result.** The frozen four-checkpoint winner received its
single permitted evaluation on the disjoint seeds `19001–19003`. It produced
19 failures in 10,000 raw boards: **99.8100000% accuracy at 100% coverage**.
The three fixed shards contributed 6, 12, and 1 failures. Triple accuracy was
65,372/65,536 (99.7497559%). This passes the 99% raw holdout goal by 81 boards.
The checkpoint hashes, `0.25/0.25/0.25/0.25` weights, corpus digest, and RNG
allocation all matched the pre-run lock. No abstention, confidence guard,
rerun, coefficient change, or sealed-data tuning was used. Board-v8 sealed is
now exposed and retired; this result is the final finite-distribution holdout
measurement for the cycle.

## 2026-08-19 — Board-v9 typed relations and No SET action

Board-v9 begins from a newly exposed limitation of the earlier benchmark: its
board sampler rejected every board without a valid SET. The new evaluator can
generate both classes exactly and treats `NoSet` as a proper raw game action,
not an abstention. Mixed evaluation alternates solvable and no-SET boards and
reports each conditional accuracy separately.

The research architecture now supervises `same`, `different`, and `invalid`
for each of count, colour, shape, and fill. The relation head emits three logits
per attribute; its softmax probabilities feed the SET head, preserving an
entirely differentiable training path. The final `argmax` names relations only
outside training. This raises the research policy from 44,254 to 44,448 learned
parameters. Legacy deployed and v8 research policies remain explicit separate
types and continue to reject incompatible state dictionaries strictly.

The data, policy, native trainer, and direct MPSGraph trainer passed their unit
tests and strict Clippy checks. A one-step fixed-batch backend parity probe
produced loss `6.396099` in MPSGraph and `6.396099628` after strict checkpoint
loading in FrankenTorch, with matching first-layer gradients. Architecture
development will use only exposed v8 data before fresh v9 corpus seeds and
gates are locked. Full design details are in `BOARD_V9_PLAN.md`.

**Typed single-model development baseline.** A 20,000-step MPSGraph run on the
exposed v8 training corpus reached 99.3286% typed-relation accuracy, but only
89.7461% mixed raw-board accuracy: 97.2168% on boards with a SET and 82.2754%
on no-SET boards. This is a rejected development result, not a v9 candidate.
The gap is an extreme-value failure: a no-SET action is correct only when all
220 invalid triples remain below the decision boundary. A post-hoc sweep of
the raw SET-logit threshold peaked at 95.8984% mixed accuracy, so static
calibration is also rejected. The next experiment directly penalizes the
log-sum-exp aggregate of negative logits in each grouped training batch.

**Negative log-sum-exp development sweep.** MPSGraph's exact maximum reduction
compiled but its gradient is not implemented on the tested Apple stack, so the
experiment uses a smooth log-sum-exp aggregate and names it accordingly. A
one-step checkpoint matched FrankenTorch at loss `9.607542154` and matching
first-layer gradients. At 10,000 steps, fixed-seed 2,048-board results were:

| Weight | Mixed | Has SET | No SET |
| ---: | ---: | ---: | ---: |
| 0.25 | 92.0898% | 96.9727% | 87.2070% |
| 1.00 | 94.2871% | 97.1680% | 91.4062% |
| 1.50 | 95.2148% | 97.3633% | 93.0664% |
| 2.00 | 95.9473% | 97.1680% | 94.7266% |
| 2.50 | **96.9238%** | 97.1680% | 96.6797% |
| 3.00 | 96.7285% | 96.1914% | 97.2656% |
| 4.00 | 89.2578% | 79.6875% | 98.8281% |

The balanced development choice is weight 2.5. On the larger identical 4,096
board stream, its 10,000-step checkpoint scored 96.2891%, while a 20,000-step
checkpoint scored 96.2402%; optimization depth did not remove the remaining
error. Their uniform ensemble reached 96.8018%.

**Rejected board-correlation sampler.** Grouping the negative half of every
batch as combinations from one real no-SET board and one atlas passed backend
parity, but at 10,000 steps it fell to 89.3555% mixed accuracy (94.6289% Has
SET, 84.0820% No SET) and reduced relation accuracy. One correlated board per
batch sacrificed too much negative and raster diversity, so this sampler
remains explicit and opt-in but is rejected as the next candidate.

**Independent ensemble and width search.** Four independent 10,000-step
replicas of the selected weight-2.5 objective scored 96.92%, 95.41%, 93.99%,
and 95.26% on the same 2,048 boards. Their fixed uniform ensemble reached
97.0703% overall, but was imbalanced at 95.2148% Has SET and 98.9258% No SET.
An exact sweep using newly explicit `expected_kind` and `strongest_is_set`
diagnostics peaked at 97.4121%, so the ensemble still has overlapping scores,
not merely a shifted threshold. Mixing one lower-loss checkpoint reached only
97.1191% after its best diagnostic threshold.

A distinct wide typed policy then doubled shared channels (12 to 24),
specialist channels (8 to 16), and specialist embedding width (24 to 48),
raising parameters from 44,448 to 173,680. Strict shape loading and one-step
MPSGraph/FrankenTorch parity passed. At weight 2.5 it improved attribute
accuracy to 99.6460% and relation accuracy to 99.1455%, but mixed raw-board
accuracy remained 96.2891% (97.6074% Has SET, 94.9707% No SET). Recalibrating
the training loss at weight 3.0 declined to 95.9717%. Width is therefore
rejected: the remaining bottleneck is the board-level action, not local visual
capacity alone.

**Board-summary head falsification.** The evaluator now records the expected
board kind, whether the strongest triple is actually a SET, deeper score
ranks, score mean and standard deviation, log-sum-exp, and threshold counts.
A deterministic PEP 723 analysis split the 2,048 uniform-ensemble boards in
half, fit only on the first 1,024, and evaluated on the disjoint second half.
A logistic board head using the rich summaries improved held-out raw action
accuracy from 97.4609% to 98.1445%; an eight-unit tanh MLP tied it at 98.1445%
rather than improving further. Summary-level board heads are therefore useful
but insufficient for the 99% goal. A future board head must consume richer
per-triple typed-relation evidence instead of only statistics of final SET
scores.

**Per-triple reader falsification.** Board evaluation can explicitly export
all 220 final scores, all 2,640 typed-relation logits, every triple target, and
the card attribute logits with `--export-board-evidence`; normal evaluations
do not pay the large artifact cost. `analyze_relation_reader.py` is a deterministic
PEP 723/NumPy probe; the first half of a 4,096-board stream is its training
split and the second half remains untouched. A shared triple MLP over the
existing score plus twelve relation probabilities reached only 96.7285% on
the held-out raw action. Giving the reader an explicit multiple-instance
board loss through differentiable log-mean-exp pooling peaked at 97.1680%.
A richer pooled board MLP reached 99.3652% on its training half but collapsed
to 96.5332% held out, so it is overfit and rejected. On the same test half,
even a perfect board-kind reader paired with the existing triple selector is
capped at 99.2188%; board classification alone has too little margin.

A score blend between the uniform four-model ensemble and the wide model was
also rejected. Selecting its coefficient and threshold on the training half
yielded 97.2656% on the held-out half. The richer relation evidence and model
diversity are therefore real but do not resolve the contextual card errors.

**Fill-loss search.** Fill is the dominant local error: the standard uniform
ensemble measured 98.5677% fill accuracy while count and colour were perfect
and shape was 99.9023%. Both native and MPSGraph trainers now expose a
normalized fill-loss multiplier. At multiplier 4, a one-step parity checkpoint
produced loss `6.323562` in MPSGraph and `6.323562224` in FrankenTorch with
matching shared gradients. The first exploratory run accidentally used the
MPSGraph default seeds and is explicitly excluded from comparison. Controlled
10,000-step runs reused baseline data seed `590011` and initialization seed
`3935559000370003845`:

| Fill multiplier | Fill accuracy | Mixed | Has SET | No SET |
| ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 98.5026% | **96.2891%** | 95.9961% | 96.5820% |
| 2 | 98.6328% | 96.0938% | 95.8008% | 96.3867% |
| 4 | **98.7956%** | 95.9961% | 96.1914% | 95.8008% |

Extra fill supervision improves the intended specialist but not the raw board
action, and the multiplier-4 checkpoint does not improve the uniform ensemble
when blended. This falsifies simple specialist loss reweighting. The next
model-search direction is a contextual card reader that can reconcile the
twelve shared card predictions before action scoring; isolated triple readers
cannot correct a visually plausible but globally inconsistent card error.

**Contextual card-reader search.** Board evidence now includes the procedural
target attributes for all twelve cards, which makes card correction measurable
without reading labels back from filenames. The PEP 723 probe
`analyze_contextual_reader.py` tested leave-one-out board summaries, a learned
hypergraph corrector over the 55 triples incident to each card, independent
corrector replicas, learned stacking of the four frozen vision models, typed
relation readers, score blends, and learned board heads. On a 4,096-board
development half plus a disjoint 4,096-board test half, the learned stacker and
hypergraph corrector reached 99.8739% card-attribute accuracy, including
99.5178% fill accuracy. Its eight-reader action ensemble nevertheless reached
only 98.4863% raw action accuracy (62 failures). A nonlinear board-existence
head over sorted score, relation, confidence, and entropy features was worse at
98.1445%. Both heads are rejected.

The useful architectural result is an exact differentiable relation layer.
For each attribute it computes the probability of `same` as the sum of three
classwise products and `different` as the sum over the six class permutations;
`invalid` is the remaining probability. Products and sums preserve gradients,
so this is a fixed neural layer rather than an argmax attribute decoder. A
direct product across the four valid-relation probabilities reached 98.8525%
on the first held-out stream. A small learned SET head over the exact relation
probabilities reached 98.9502% with its development threshold. Its test-only
threshold ceiling was 99.0479%, identifying calibration as part, but not all,
of the gap.

Threshold ties were then made explicit. Selecting the conservative edge of a
development tie improved the direct soft reader to 98.9746%. For one learned
exact-relation replica, taking the more conservative of thresholds fit on two
development halves reached 40 failures in 4,096 boards: 99.0234% overall,
99.4141% Has SET, and 98.6328% No SET. This is an exposed development crossing,
not a promoted result.

**Board-seed replication falsification.** The architecture, seeds, and
conservative split-threshold rule were frozen before a disjoint procedural
board stream with seed `2026081901`. The chosen exact-relation replica fell to
98.6572%; even its test-only threshold ceiling was 98.6816%. Training the
contextual reader on all 8,192 boards of the original stream and evaluating all
8,192 boards of this second stream produced a uniform four-head result of
99.0112% (81 failures). Because that ensemble choice was exposed to the second
stream, it was frozen and repeated with stream two as development and a third
stream, seed `2026081902`, as test. It failed at 98.7305% (104 failures).
Per-head development normalization and a board reader over exact-score
summaries did not improve it. The third stream's corrected card attributes were
99.8866% accurate, but fill remained the bottleneck at 99.5555%.

Contextual post-processing and the differentiable relation layer are retained
as research probes, but neither is integrated into the browser policy. The 99%
raw v9 action goal has not replicated. The next architecture search should
improve fill evidence or calibrated attribute uncertainty in the visual
specialist itself, then reuse the exact differentiable relation layer; further
heads over the current frozen evidence are not justified by these results.

**Fill-error localization.** `analyze_fill_errors.py` measured the frozen
uniform ensemble on 8,192 mixed boards (`2026081902`): 98.7579% fill accuracy,
or 1,221 errors in 98,304 card appearances. Of those, 1,215 were
striped-versus-open confusions. Solid fill was 99.9817% accurate, while striped
and open were 98.1874% and 98.1012%. The worst capture profile was `crop-w79`
at 7.2034%. Four hundred thirty-seven cards were wrong in at least three of the
four models, so averaging alone could not remove the shared low-resolution
failure.

Two targeted candidates were rejected. Adding 80 low-resolution training
atlases improved `crop-w79` but reduced matched aggregate fill accuracy from
98.4833% to 98.3632%. Replacing the shared 5x5 stride-2 convolution with a 3x3
stride-2 convolution passed Rust/MPSGraph parity but reduced fill accuracy to
97.7244% and raw board accuracy to 92.6270%. The rejected kernel topology was
removed rather than retained as a dormant implementation.

**Fill-only texture branch.** A new opt-in research policy keeps the baseline
encoder and specialists, then adds a raw-RGB fill residual:
3x3 stride 1, two 3x3 stride-2 convolutions, a projection, and a three-logit
classifier. Its logits are added only to the standard fill specialist, so the
branch cannot directly alter count, colour, or shape. The final searched width
is 8 convolution channels and a 24-unit projection, for 55,155 total policy
parameters versus 44,448 in the typed baseline. The browser-deployed policy
type and checkpoint contract remain unchanged.

The narrower 4/12 branch passed parity at 47,211 parameters and improved a
four-model uniform ensemble from 96.0693% to 96.8872% raw action and from
98.7579% to 99.0346% fill accuracy on the matched 8,192-board stream. Its
frozen contextual action ensemble reached only 98.7793%, so capacity was the
next predeclared search axis. The 8/24 branch passed the one-step parity gate:
MPSGraph probe loss `16.082127`, FrankenTorch loss `16.082128620`, and matching
sampled shared gradients. A single controlled replica reached 96.8628% raw
action and 99.0112% fill accuracy. Four locked replicas reached 97.2900% raw
action and 99.2544% fill accuracy. Applying the existing frozen contextual
recipe to its untouched 4,096-board half produced 39 failures: **99.0479%**
overall, 99.3652% Has SET, and 98.7305% No SET. This is an exposed architecture
selection result, not the replication.

**Disjoint 190xx replication.** The 8/24 topology, uniform model weights,
training seeds, and contextual recipe were frozen, then evaluated on the
disjoint `board-v8-sealed-atlases` render family with board seed `2026081903`.
Before contextual correction the ensemble reached 98.1812% raw action and
99.5880% fill accuracy. On the untouched 4,096-board half, the development-
selected hypergraph action ensemble reached **99.2188%** overall at 100%
coverage: 99.4629% Has SET and 98.9746% No SET, or 32 total failures. This is
the first honest replication of the v9 overall 99% raw-action goal. The No SET
conditional missed a separate 99% conditional gate by one error, so the
stricter three-way promotion gate remains open. A secondary frozen blend
reported 99.3896% overall, 99.6582% Has SET, and 99.1211% No SET, but it was not
the development-selected crossing and is not used as the primary claim.

The fill branch is retained as a real visual improvement. The contextual and
hypergraph readers remain clearly named NumPy research harnesses; they are not
yet the Rust browser policy. The next implementation milestone is to port the
frozen differentiable relation/correction path, then validate the complete
runtime policy without relaxing the conditional gate.

## 2026-08-19 — Fill-texture policy promoted to the browser runtime

The retained 8-channel/24-unit fill-texture topology is now honestly named
`DeployedVisualSetPolicy`. Native training, native evaluation, MPSGraph
training, and FrankenTorch/WASM inference select it by default. The temporary
`--fill-texture-branch` opt-in and its separate research wrapper were removed;
`--legacy-untyped` and `--wide-vision` remain explicit research alternatives.
The selected replica-1 checkpoint was copied byte-for-byte into the embedded
runtime input (`SHA-256 dc7819edafb75d80914a6f56d154a19d9e2e721a96f1dccfe2482a97f621717e`).

This deployment does **not** claim to port the four-checkpoint contextual or
hypergraph NumPy reader. The embedded single checkpoint scored 96.8628% raw on
the 180xx development stream and 97.4976% raw on the disjoint 190xx stream.
**Guard metric correction.** The first runtime report mistakenly filtered on
`strongest_is_set`, which is an oracle label describing whether the selected
triple is truly valid, rather than on the model's predicted `action`. That
leaked ground truth into the guard evaluation. The 99.9004%-at-49.0234% figure
is invalid and withdrawn. The scripts now select candidate clicks from the
actual `Set(...)` versus `NoSet` action. Refitting on 180xx predicted-SET
actions at the same 99.9% development criterion allowed 2,440/8,192 clicks,
with 2 errors. Frozen 190xx evaluation allowed 2,371 clicks: 2,363 correct and
8 wrong. On Has SET boards it clicked 2,369/4,096 (57.8369% coverage), with 6
errors (99.7467% accuracy); it also allowed 2 wrong clicks on 4,096 `NoSet`
boards. Guard results remain deployment diagnostics and are not the raw model
claim.

The release WASM was rebuilt with the promoted checkpoint. Native strict-load
tests, Rust unit tests, JavaScript tests, preprocessing and guard tests passed.
The live headless screenshot path produced 12/12 valid SET proposals and
clicked 7/12 guarded actions with verified board transitions. A
1,000-iteration Node/WASM sample measured 5.62 ms cached inference and 5.73 ms
including pixel transfer. The one-step backend parity check used the retained MPSGraph
attribute-loss weight of 4: MPSGraph loss `6.413500`, FrankenTorch loss
`6.413499775`, with the first eight shared gradients agreeing to roughly
`6e-8`. Running the native probe at its unrelated default weight of 0.35 is not
a valid parity comparison.

## 2026-08-19 — Unbounded generative evaluation and curriculum feedback

The fixed twelve-case extension test was generalized into an unbounded browser stream. Each case
uses fresh deterministic board and renderer seeds plus randomized theme, capture profile, viewport, and DPR, with an
exactly balanced `HasSet`/`NoSet` schedule. Inference remains pixel-only; the harness reads DOM card
IDs only after the action is frozen. Every mining failure is appended as a reproducible JSONL
curriculum record, and subsequent mining samples mix fresh cases with nearby conditions derived from
the accumulated failures. This is active curriculum discovery, not online weight updating: a later
training cycle consumes the regenerated cases.

The adaptive miner and scoring auditor are deliberately separate. `--audit` refuses curriculum
input and does not write failures, so its fixed root seed can support honest evaluation. Once an
audit is inspected or its failures enter the curriculum, that stream is retired as development
data.

The headless browser bridge was also corrected. It now captures the current live game tab on every
request through CDP; it no longer substitutes a pre-rendered unrelated board after a click. Thus a
passing transition assertion now observes the actual post-click page state.

## 2026-08-20 — Failure curriculum connected to both native trainers

The mining journal can now drive a real subsequent training cycle without materializing a generated
bitmap corpus. `tools/render-vision-curriculum.mjs` reconstructs each unique failure with the
production responsive board renderer, captures it in memory, and applies the extension's 30×21 RGB
area-resampling contract. It writes a versioned binary stream containing twelve normalized card
tensors plus the post-prediction board/action labels. The symbolic JSONL remains the only persistent
curriculum artifact; PNGs are neither created nor retained.

`set-vision-data` validates and decodes that stream, verifies that the board IDs agree with the
declared Has-SET/No-SET kind, and creates balanced failure-centered batches. Exact observed pixels
supply all available positive and negative triples. For a No-SET board, the exact board supplies the
correlated negative half while ordinary procedural training supplies the positive half. The wrongly
selected triple is placed first among negatives when applicable. Board IDs are supervision at
training time only; browser inference remains pixel-only.

Both the fast MPSGraph trainer and the FrankenTorch reference trainer accept `--curriculum-stdin`
and `--curriculum-rate`. The rate selects failure-centered optimizer batches; audit/eval-only mode
rejects curriculum input. An end-to-end smoke replayed the known `signal`/`blur` No-SET failure from
root seed `20260823`, iteration 26. MPSGraph completed a curriculum-only step in 8.57 ms after graph
compilation, and the reference trainer completed the same framed-input training path. These are
plumbing checks, not evidence that the resulting model improves generalization; promotion still
requires a fresh, feedback-free audit stream.

## 2026-08-20 — Active-v10 rate sweep precommitted

Before candidate training, the active-v10 protocol was fixed in `ACTIVE_V10_PROTOCOL.md`. A
fresh-only 256-board browser stream at root seed `2026082001` produced 16 raw failures and exactly
matched 93.75% Has-SET and No-SET accuracy. The errors span clean and altered captures, all four
themes, DPR 1 and 2, small and large cards, false SET selections, wrong triples on Has-SET boards,
and false No-SET actions. This stream and its 16 failures are development data.

The controlled candidates use curriculum rates `0/0.02/0.05/0.10/0.20` with identical topology,
base corpus, optimizer budget, data seed, and initialization seed. A separate curriculum RNG and an
always-generated base batch ensure that changing rate does not perturb the ordinary data sequence.
Selection uses 4,096 exposed research boards at 100% raw coverage. The selected checkpoint then gets
one 1,000-board feedback-free browser audit at the already fixed root seed `2026082099`; its
predeclared gate is 98.0% overall and 97.5% for each board kind. No audit result appears below this
entry because candidate training had not begun when the protocol was recorded.

**Full-batch result.** Rate zero reproduced the deployed checkpoint exactly
(`dc7819edafb75d80914a6f56d154a19d9e2e721a96f1dccfe2482a97f621717e`) and scored 117/4,096
failures. Full curriculum-batch rates `0.02/0.05/0.10/0.20` scored `127/153/151/157` failures. The
approach is rejected: every nonzero candidate regressed both raw board accuracy and triple accuracy.
Before additional training, the protocol was amended with four sparse within-batch candidates that
replace at most 2 or 4 of 64 base samples. This tests whether hard examples help when they cannot
monopolize an optimizer step; the audit remains unopened.

**Sparse result.** Replacing one example per label in 25%, 50%, and 100% of batches produced
`123/110/141` failures; replacing two per label in every batch produced 125. The predeclared winner
is `p1-r050`: 110/4,096 failures (**97.3145%**) versus the control's 117 (**97.1436%**). Has-SET
improved from 97.6074% to 97.7539%, No-SET from 96.6797% to 96.8750%, triple accuracy from 96.2524%
to 97.0032%, and attribute accuracy from 99.6780% to 99.7248%. Its checkpoint SHA-256 is
`b2e763d5bf3604734dc3fe491ba47d0ef47dc2bc53e23d1fdf35cb25ca146b2e`. This is a modest exposed
development improvement, not yet a robustness claim. The fixed browser audit remains unopened at
this point.

**Fresh browser audit result.** The selected `p1-r050` checkpoint passed the fixed extension smoke
test and then ran exactly once on all 1,000 predeclared, feedback-free boards at root seed
`2026082099`. It scored **95.2% raw accuracy** (48 failures), split into 96.6% Has-SET and 93.8%
No-SET. Guarded accuracy was 99.4169%, but at only 34.3% coverage and therefore does not alter the
raw result. The candidate misses every predeclared raw gate and is rejected. Most false proposals
violated exactly one true attribute, commonly shape or fill; the maximum over 220 triples amplifies
these locally plausible errors on No-SET boards. This audit is now exposed and retired. Its 48
failure records are development data for active-v11, whose candidate and fresh-audit rules were
locked in `ACTIVE_V11_PROTOCOL.md` before the next mining stream began.

## 2026-08-20 — Active-v11 targeted replay

The predeclared focused miner started with the 16 original development failures and 48 failures from
the now-retired active-v10 audit. On 768 boards at root seed `2026082101`, 202 were fresh and 566
were curriculum-derived, including 146 exact replays. The intentionally hard stream scored 62.5%
raw and added 288 failure records. The resulting journal has 352 records and 219 unique rendering
conditions after exact-condition deduplication, covering both board kinds and all ten capture
profiles. These figures measure curriculum hardness, not generalization.

The replay mixer now has an explicit `--targeted-curriculum` mode in both trainers. It preserves the
ordinary batch first, then replaces only labels implicated by the failure: a No-SET false positive
adds its selected hard negative; a Has-SET false negative adds an exact positive; a wrong proposal
on a Has-SET board adds both. The old symmetric mixer remains the default and is a controlled
candidate. Unit tests verify that targeted No-SET replay leaves every positive slot unchanged and
targeted Has-SET false-negative replay leaves every negative slot unchanged. The framed
browser-to-MPSGraph path also completed a real one-step smoke run.

**Targeted sweep result.** On the fixed 4,096-board exposed atlas evaluation, expanded symmetric
replay at rate 0.50 regressed to 137 failures. Targeted replay produced 131 failures at rate 0.50,
129 at rate 1.00 with one implicated example per label, and 115 at rate 1.00 with up to two. The
last candidate reached 97.1924% raw, split 97.8027% Has-SET and 96.5820% No-SET, with 99.7060%
attribute accuracy. Targeting therefore beats symmetric replay at comparable exposure, and a second
hard example restores some No-SET balance, but none beats the active-v10 reference's 110 atlas
failures. Per the precommit, the best three targeted checkpoints and unchanged reference advance to
the separate 512-board browser development comparison at root seed `2026082102`.

**Browser selection result.** The unchanged active-v10 reference scored 35/512 failures (93.1641%,
95.7031% Has-SET, 90.6250% No-SET). Targeted `r100-p2` scored 34 (93.3594%, 95.7031%, 91.0156%);
targeted `r100-p1` and `r050-p1` each scored 38. The two-example model therefore wins the locked
rule by one board, an intentionally modest development result rather than a claim of improvement.
Its checkpoint SHA-256 is `52036c615e1c6a695996d7d184611ea4f41e9e68cd06bdc659a50935465494d4`.
The extension rebuilt with that exact checkpoint and passed all 12 fixed raw proposals; six guarded
actions clicked and verified a live pixel transition.

**Fresh audit result.** The selected checkpoint then ran exactly once on all 1,000 predeclared
boards at root seed `2026082199`. It scored **94.7% raw** (53 failures), split 98.0% Has-SET and
91.4% No-SET. Guarded accuracy was 99.6226% at 26.5% coverage. Targeted replay therefore improved
recall while pushing too many locally plausible triples above the raw zero boundary; it worsened the
No-SET extreme-value problem and misses the gate decisively. This audit is exposed and retired.

## 2026-08-20 — Active-v12 aligned checkpoint soups

Active-v11's fresh audit exceeded its 20-failure overall allowance before completion, so
`ACTIVE_V12_PROTOCOL.md` locked the next experiment before any soup was constructed. A new strict
Rust utility linearly interpolates matching F64 tensors and rejects key, metadata, length, dtype, or
finite-value violations. The two fixed ingredients share topology, initialization, base-batch RNG,
and schedule: the active-v10 reference and active-v11 targeted `r100-p2`.

On the exposed 4,096-board atlas stream, primary weights `0.25/0.50/0.75` for the active-v11
checkpoint produced `101/98/103` failures. All beat both endpoints (110 and 115); the 50/50 soup
reached **97.6074% raw**, split 98.1934% Has-SET and 97.2168% No-SET. This is one 55,155-parameter
model with unchanged inference cost, not a runtime ensemble. The best two soups and both endpoints
advance to the predeclared browser development seed `2026082201`; no browser result was inspected
before this entry.

**Browser selection result.** On the identical 512 live-browser boards, the 50/50 soup scored
19 failures (**96.2891%**, 97.6563% Has-SET, 94.9219% No-SET). The 25%-primary soup, active-v10
endpoint, and active-v11 endpoint scored 23, 36, and 27 failures respectively. The midpoint nearly
halved the old reference's errors and improved both conditionals, so it wins the locked rule. Its
SHA-256 is `ca20943318755c037a9ac08faaeb3887ec06ae4ae27a3c7b2017e1cabfc3bbec`. The rebuilt
extension passed all 12 fixed raw proposals and transition-verified six guarded actions. The fresh
audit seed remains the predeclared `2026082299`.

**Fresh audit result.** The selected soup ran exactly once on all 1,000 predeclared boards at root
seed `2026082299` and scored **94.2% raw** (58 failures), split 96.8% Has-SET and 91.6% No-SET.
Guarded accuracy was 99.0385% at 31.2% coverage. Parameter averaging is retained as a real paired
development improvement, but it does not pass the fresh gate and the checkpoint is rejected as a
complete policy. The evidence supports adding an explicit learned board-existence action head;
`ACTIVE_V13_PROTOCOL.md` was locked before its all-board evidence stream.

## 2026-08-20 — Active-v13 board-existence head

The frozen 50/50 soup collected diagnostics from 1,024 fresh, balanced, in-memory browser renders
at root seed `2026082301`; no bitmaps were persisted. Its unchanged raw zero-threshold action scored
53 failures (**94.8242%**), split 95.7031% Has-SET and 93.9453% No-SET. The evidence contains the
strongest score, runner-up margin, selected-card entropy and margin, and four statistics over all
220 triple scores. Labels were attached only after frozen inference.

The first fit exposed a protocol error before any candidate was deployed: because generation
alternates board kind by iteration, the specified even/odd split contained 512 No-SET fitting
boards and 512 Has-SET validation boards. Its nonsensical report and head are retained under
`invalid-even-odd-*`. The protocol records the correction: within each board kind, records are
ranked by a fixed hash of board seed and divided into equal 256-board fitting and validation halves.
This assignment uses neither model output nor action correctness.

On that corrected split, validation selected the predeclared L2=0.1 logistic head. It scored
25/512 fitting failures (95.3125%; 95.3125% Has-SET, 94.9219% No-SET) and 22/512 validation
failures (95.7031%; exactly 95.7031% for each board kind). This is only a development candidate;
the locked, feedback-free 512-board browser comparison at root seed `2026082302` remains unopened
at this entry.

**Browser selection result.** On the identical 512 fresh browser boards, the unchanged raw
zero-threshold head scored 26 failures (94.9219%), split 97.2656% Has-SET and 92.5781% No-SET. The
fitted head scored 22 failures (**95.7031%**), split 96.8750% Has-SET and 94.5313% No-SET. It reduced
total errors by four and improved No-SET by five boards while losing one Has-SET board, exactly the
maximum conditional regression allowed by the locked rule. The fitted head is selected and the
extension has been rebuilt with it. The one fresh audit at root seed `2026082399` remains unopened
at this entry.

**Fresh audit result.** The selected head passed the fixed 12-case extension smoke, then ran exactly
once on all 1,000 predeclared boards at root seed `2026082399`. It scored **94.8% raw action
accuracy** (52 failures), split 95.8% Has-SET and 93.8% No-SET. Guarded accuracy was 99.0196% at
30.6% coverage and is not part of the raw claim. The 21 Has-SET and 31 No-SET failures spanned every
capture profile and all four themes. The head balanced the action classes but did not lower the
upstream visual/triple error floor, so it is rejected as a robustness solution. Its modest paired
development improvement may remain deployed while the next upstream model experiment runs; this
audit is exposed and retired.

## 2026-08-20 — Active-v14 full browser-distribution training

`ACTIVE_V14_PROTOCOL.md` locked this experiment before its training render. The browser renderer
and streaming evaluator now share one deterministic fresh-case generator. A generic
`RenderedBoardCollection` replaces the misleading Rust curriculum-only type name while preserving
the separate historical failure-replay methods. Its new balanced sampler independently draws a
board and triple for each batch slot; positive triples come from Has-SET boards and negative triples
from either kind. Unit tests verify label balance, multiple-board sampling, exact tensor dimensions,
and rejection when no positive source board exists.

One Chrome process rendered all 2,048 balanced boards at root seed `2026082401` through the
production React/SVG board and fanned the framed 30x21 RGB tensors directly into both MPSGraph
trainers. No screenshot, generated card bitmap, or tensor corpus was persisted. Real two-board
smokes completed an optimizer step through both MPSGraph and the FrankenTorch reference trainer.
The fixed 10,000-step candidates completed in 59.231 seconds (`browser-r050`, SHA-256
`ab6a673887f8cb1568d346930e029525b8a80e9cf4fd46baa05a2a63cce2e98f`) and 53.692 seconds
(`browser-r100`, SHA-256 `332380e3b6e1e0d8169b3759a60222f8ccc9e65d4dff0324ab9f96f42281b42c`).

Both checkpoints loaded strictly into the deployed FrankenTorch topology. On the exposed 4,096-board
atlas diagnostic, `browser-r050` scored 96.3623% raw (149 failures; 97.2168% Has-SET, 95.5078%
No-SET), while `browser-r100` scored 92.2363% (318; 96.1426% Has-SET, 88.3301% No-SET). The
live-only result demonstrates loss of atlas coverage. Per protocol these are regression diagnostics,
not selection results; both candidates and the fixed active-v12 soup reference advance to the
paired 512-board raw browser seed `2026082402`, which remains unopened at this entry.

**Browser selection result.** With a raw zero-threshold head on the identical 512 fresh browser
boards, the active-v12 soup reference scored 26 failures (94.9219%; 97.2656% Has-SET, 92.5781%
No-SET). `browser-r050` scored 25 (**95.1172%**; 96.8750% Has-SET, 93.3594% No-SET), gaining two
No-SET boards while losing exactly one Has-SET board and therefore remaining eligible. The
live-only `browser-r100` candidate was decisively worse at 42 failures (91.7969%; 96.4844% Has-SET,
87.1094% No-SET). The 50/50 candidate wins the locked rule by one error, a weak development result.
Its exact checkpoint is now embedded with the raw head. The fresh seed `2026082499` remains
unopened at this entry.

**Fresh audit result.** `browser-r050` passed the fixed extension smoke and then ran exactly once on
all 1,000 predeclared boards at root seed `2026082499`. It scored **94.8% raw** (52 failures), split
96.0% Has-SET and 93.6% No-SET. Guarded accuracy was 98.7302% at 31.5% coverage. It misses every raw
gate and is rejected. Seventeen of the 52 failures used the crop profile, versus two to seven for
each other profile; failures still covered all themes and every profile. Whole-batch live-domain
mixing is falsified: the one-board selection gain did not replicate, while live-only training
catastrophically forgot atlas coverage. The extension is restored to the active-v12 soup and
active-v13 fitted action head while the next upstream experiment is designed. The reusable result
is the deterministic, no-bitmap browser training stream—not either active-v14 checkpoint.

## 2026-08-20 — Active-v15 neural-attribute relation readers

The restored soup collected 1,024 balanced fresh browser records at root seed `2026082501`. WASM
retained all 144 post-softmax card-attribute probabilities and 220 learned triple scores only after
the frozen pixel forward pass; evidence mode then attached card IDs for scoring. No pixels were
persisted, audit mode still rejects evidence, and the production extension does not request the
arrays. The current fitted-head action scored 48 failures (95.3125%) over the full development
stream.

On the fixed hash-stratified split, current scored 25/512 fitting and 23/512 validation failures.
Exact continuous same/different composition improved fitting to 21 but tied validation at 23. The
25%-learned blend scored 20 fitting failures and also tied at 23 validation; the 50% and 75% blends
scored 24 validation failures. Discrete attribute argmax scored 27 validation failures and was badly
imbalanced (98.0469% Has-SET, 91.4063% No-SET). Per the locked simplicity tie-break, current remains
selected, so no alternate reader is ported and no browser seed or audit is opened. This confirms
that exact rule composition cannot recover information absent from the visual attributes.

The same post-inference evidence localizes the live visual bottleneck over 12,288 card appearances:
count 99.9430%, colour 99.7803%, **shape 98.4456%**, and fill 98.9176%. Crop is worst on both shape
(98.0710%) and fill (98.3025%); shadow and tilt shape are each 98.0583%. Shape—not fill—is now the
largest live-browser specialist error, unlike the earlier atlas-only diagnosis. The next model
search should change shape learning or evidence preservation rather than add another board-level
reader.

## 2026-08-20 — Active-v16 shape-loss search

`ACTIVE_V16_PROTOCOL.md` locked the hypothesis, candidates, seeds, and selection rule before any
training or browser generation. The attribute objective now supports independent shape and fill
multipliers while preserving its normalized scale; multiplier one is exactly the historical loss.
Both FrankenTorch and MPSGraph expose the same shape control. At shape multiplier 2, a one-step
parity probe produced loss `13.558712406` in FrankenTorch and `13.558712` in MPSGraph, with the
first eight parameter gradients agreeing to approximately 1e-7.

The two fixed 10,000-step MPSGraph trainings each completed in about 79.2 seconds. `shape-m2` has
SHA-256 `43e3fd8b654bb4c7ca191ecbd8c076b0798d895b24552582541efcc4f36481fa`;
`shape-m4` has SHA-256 `422657db2b593951f2602906493440433d93d946e785766c68cc37ef0131bc3c`.
Both loaded strictly through the FrankenTorch evaluator. On the exposed 4,096-board atlas,
`shape-m2` scored 96.8994% raw (97.3633% Has-SET, 96.4355% No-SET), 96.4172% triple accuracy,
and 99.6674% attribute accuracy. `shape-m4` scored 97.0459% raw (97.7051% Has-SET, 96.3867%
No-SET), 96.8140% triple accuracy, and 99.7152% attribute accuracy. Shape accuracy rose from the
fixed control's 99.7620% to 99.8027% and 99.8067%, respectively, but its control board accuracy was
slightly higher at 97.1436%. These atlas results are diagnostic only. The locked 512-board browser
seed `2026082601` remains unopened at this entry.

**Browser selection result.** On the identical 512 fresh browser boards with the raw zero-threshold
action, the fixed control scored 28 failures (94.5313%), split 97.2656% Has-SET and 91.7969%
No-SET. `shape-m2` scored 32 failures (93.7500%), split 96.0938% Has-SET and 91.4063% No-SET;
`shape-m4` also scored 32 failures, split 95.7031% Has-SET and 91.7969% No-SET. Both candidates
exceeded the permitted Has-SET regression and neither improved total failures, so the fixed control
wins. Per protocol, no fresh audit is opened. The extension has been restored byte-for-byte to the
active-v12 soup checkpoint and active-v13 fitted board head.

The hypothesis is rejected as an end-to-end policy improvement: isolated shape classification did
improve, but extra shape supervision perturbed triple calibration and reduced browser action
accuracy. Future searches must optimize or preserve the action surface directly; per-attribute
accuracy is a diagnostic, not a sufficient model-selection target.

## 2026-08-20 — Active-v17 pixel-grid multiview inference

`ACTIVE_V17_PROTOCOL.md` locked a sampling-phase hypothesis before implementation or evaluation.
The candidate keeps the deployed soup checkpoint, learned topology, and fitted board head frozen.
It interprets each fractional CSS card boundary four ways on the integer screenshot grid, applies
the existing 30x21 area resampler to each, runs the same model four times, averages neural triple
scores and attribute probabilities, and recomputes all board-head features. This is an honest
same-model test-time view ensemble, not extra training, symbolic rule solving, or a learned-model
ensemble. The historical crop is view one and remains the sole pixel buffer used by the temporal
guard.

Pure tests verify crop bounds, Rust-compatible triple order, evidence averaging, and recomputed
diagnostics. A two-board real Chrome/WASM smoke at seed `2026082700` scored both board kinds
correctly without persisting screenshots or tensors. The paired 512-board selection seed
`2026082701` remains unopened at this entry.

**Browser selection result.** With the frozen fitted head, baseline and four-view each scored
exactly 21/512 failures (**95.8984%**), split 97.2656% Has-SET and 94.5313% No-SET. More strongly,
the complete set of 21 failed iteration numbers was identical. Neural scores and uncertainty did
move on many failures, so the views were not byte-identical, but no action crossed into correctness.
Baseline wins the locked tie rule and the fresh audit remains unopened. Pixel-grid phase averaging
at the already-compressed 30x21 input does not recover the missing evidence; the research path
remains opt-in and production stays single-view.

## 2026-08-20 — Active-v18 aligned shape-checkpoint soups

`ACTIVE_V18_PROTOCOL.md` locked six strict parameter interpolations before construction. Each shape
endpoint shares its initialization, base-data sequence, and optimizer schedule with the fixed v16
control, so coordinate-wise interpolation is meaningful. Every generated checkpoint loaded through
the canonical FrankenTorch evaluator and remains a single 55,155-parameter model.

On the exposed 4,096-board atlas, shape-m2 primary weights 0.25/0.50/0.75 produced 125/157/145
failures; shape-m4 weights 0.25/0.50/0.75 produced 136/143/118. The locked top two are `m4-w075`
(97.1191%; 97.7539% Has-SET, 96.4844% No-SET; SHA-256
`3cf309bf43064e200d382c521a3561e111ac651fe74297c5b0f9d50624fb4ec7`) and `m2-w025`
(96.9482%; 97.3633% Has-SET, 96.5332% No-SET; SHA-256
`c1e9595e45536a9584e00ad10299c6441e4e2b9bad95346d0d471c06143cb30e`). The former nearly
recovers the control's 117 failures, but none beats it on this diagnostic. Per protocol both still
advance against the stronger current deployment on browser seed `2026082801`, which remains
unopened at this entry.

**Browser selection result.** With the identical frozen active-v13 board head, the prior deployed
checkpoint scored 30/512 failures (94.1406%; 94.5313% Has-SET, 93.7500% No-SET). `m4-w075`
scored 20 (**96.0938%**; 95.3125% Has-SET, 96.8750% No-SET), while `m2-w025` scored 23
(95.5078%; 95.3125% Has-SET, 95.7031% No-SET). Both soups improved both conditionals and were
eligible; `m4-w075` wins by three boards. The extension now embeds its exact checkpoint hash
`3cf309bf43064e200d382c521a3561e111ac651fe74297c5b0f9d50624fb4ec7` with the unchanged fitted
head. The fresh audit seed `2026082899` remains unopened at this entry.

**Integration-gate rejection.** Before opening that audit, the selected soup ran the fixed 12-case
Chrome screenshot/click smoke. On the known clean classic 390x800 case it proposed card IDs
55/36/19, which are not a SET; the other eleven raw proposals were valid and eight guarded actions
transition-verified. A browser-development gain cannot override a deterministic clean-case action
regression in the release gate. The soup is rejected for deployment, the active-v12 checkpoint and
fitted active-v13 head are restored, and audit seed `2026082899` remains unopened. This is another
warning that small fixed development comparisons can select brittle policies even when both class
conditionals improve.

## 2026-08-20 — Active-v19 large in-memory board-head data

`ACTIVE_V19_PROTOCOL.md` locks a 16,384-board evidence stream and richer regularized head search.
A new narrow Rust utility reads the existing versioned renderer stream, strict-loads the canonical
deployed FrankenTorch policy, scores complete boards, and emits post-forward neural evidence. Its
SET rule is used only to score the model's already-selected positions; it never selects a candidate.
The rendered collection exposes read-only board views rather than leaking private training storage.

All 28 `vision-data` tests (one benchmark ignored), the evaluator's two focused tests, compile-first
check, formatting, and strict Clippy pass. A real two-board Chrome→binary pipe→FrankenTorch smoke at
seed `2026082900` produced one Has-SET and one No-SET correct action, retained all 220 learned scores
per board, and spent 18 ms in native inference. Pixels existed only in memory and the pipe; the only
persistent smoke artifact is JSON neural evidence. The 16,384-board seed `2026082901` remains
unopened at this entry.
The initial large-run attempt exposed that `read_framed` buffered every board before inference,
which would require roughly 3 GB for 16,384 boards. It was stopped before producing evidence and
retained under `invalid-buffered-*`. A new `visit_framed` interface decodes and scores one board at
a time; its focused test and strict Clippy pass. Persistent page reuse and a 5 ms readiness poll cut
a 100-board benchmark from 21.0 to 10.9 seconds. Before opening the fixed seed, the protocol was
amended to execute the exact sequence as four ordered, concurrent 4,096-board shards.

All four shards completed in 467–473 seconds with exact global indices 0–16,383 and 4,096 evidence
rows each. Raw zero-threshold accuracy was 94.5068% overall: shard accuracies were 94.5068%,
93.8721%, 94.7998%, and 94.5068%. The model was consistently imbalanced toward Has-SET: shard
Has-SET accuracies ranged from 96.3379% to 97.6074%, while No-SET ranged from 91.4063% to 92.5781%.

On the fixed 8,192-board validation half, the deployed active-v13 head scored 362 failures
(**95.5811%**), split 96.5088% Has-SET and 94.6533% No-SET. The richer L2=0.01 head was best at
339 failures (**95.8618%**), split 96.3867% Has-SET and 95.3369% No-SET: it recovered 28 No-SET
boards but lost five Has-SET boards. The locked eligibility tolerance was four Has-SET regressions,
so it is ineligible by exactly one board. L2 0.1/1/10 scored 363/414/492 failures. The deployed head
remains selected; browser seed `2026082902` and audit seed `2026082999` remain unopened. Large-data
score-distribution modeling yields a real but small calibration improvement, still bounded by the
upstream visual/triple errors.

## 2026-08-20 — Active-v20 multi-board negative groups

`ACTIVE_V20_PROTOCOL.md` locked a controlled attempt to retain No-SET board correlation without
collapsing all negative training examples onto one board and one raster condition. The negative
half of each batch was divided into either four groups of eight triples or eight groups of four;
every group came from its own generated No-SET board and independently sampled atlas. The
log-sum-exp false-positive loss was computed separately per group and averaged.

The grouped sampler has focused layout and validation tests. Both native FrankenTorch and
MPSGraph implement identical grouped objectives. A four-group one-step parity probe produced final
loss `10.456101309` in FrankenTorch and `10.456100` in MPSGraph; the first eight shared-weight
gradients agreed to approximately 1e-7. The fixed 10,000-step MPSGraph candidates trained in 78.6
and 78.4 seconds. Their SHA-256 hashes are
`42daf7e6be138aa662e4dfeaa70b8e54cbf3d9bc6e0e9d0e28e71a2280566a7d` (four groups) and
`9324b8b2b075474ff64e53dcae51b6ba462113aa5161d39d192c25d1691f1dee` (eight groups).

Both candidates failed the exposed 4,096-board atlas gate decisively. The fixed control scored 117
failures (49 Has-SET, 68 No-SET) with 96.2524% triple accuracy. Four groups scored 302 failures
(50 Has-SET, 252 No-SET) despite 97.6929% triple accuracy. Eight groups scored 248 failures (38
Has-SET, 210 No-SET) despite 98.5046% triple accuracy. Neither browser selection nor audit seed was
opened, and the active-v12 deployment remains untouched.

This is a useful mechanistic rejection rather than a generic failure: shrinking each log-sum-exp
from 32 terms to 8 or 4 lowers its logit scale by approximately `ln(groups)`. The grouped models
learned markedly better local relations and preserved or improved Has-SET recognition, but the
weaker aggregate pressure allowed many No-SET false positives. A follow-up must restore the missing
log-count term before judging whether multi-board diversity itself helps.

## 2026-08-20 — Active-v21 count-corrected negative groups

`ACTIVE_V21_PROTOCOL.md` locked the unique `ln(group count)` correction implied by splitting the
control's 32-term negative log-sum-exp into smaller aggregates. The sampler, topology, seeds,
optimizer, and loss weight stayed fixed. A corrected four-group parity probe produced loss
`13.494941940` in FrankenTorch and `13.494942` in MPSGraph, with the first eight shared gradients
agreeing to approximately 1e-7. The 10,000-step four- and eight-group checkpoints trained in 84.8
and 85.4 seconds with SHA-256
`360a07306064ba9854ec23db301243f1c20e0813a802d87969af26e90f98a365` and
`b70729a57bb424698784a017a5cc0405b292aa69a8588642d5cfb846dd30386e`.

Both candidates again failed the exposed atlas gate. Four corrected groups scored 278 failures (46
Has-SET, 232 No-SET) and 97.5525% triple accuracy. Eight corrected groups scored 254 failures (36
Has-SET, 218 No-SET) and 98.2971% triple accuracy. The control remains 117 failures (49 Has-SET,
68 No-SET). No browser or audit seed was opened and deployment remains untouched.

The count correction does not rescue the distribution replacement. The grouped candidates retain
their strong local relation and Has-SET behavior but see too few broad hard negatives. The next
test should preserve the control's ordinary negative distribution and inject only a sparse number
of correlated No-SET-board triples.

## 2026-08-20 — Active-v22 sparse board-correlated injection

`ACTIVE_V22_PROTOCOL.md` preserved the control's ordinary 32-negative distribution and replaced
only four or eight negative positions with distinct triples from one generated No-SET board. The
original single 32-term log-sum-exp and weight 2.5 remained unchanged. Focused tests prove that
only the requested negative prefix is replaced and invalid counts are rejected. A four-injection
one-step parity probe produced loss `13.637276577` in FrankenTorch and `13.637276` in MPSGraph,
with the first eight shared gradients agreeing to approximately 1e-7.

The 10,000-step count-4 and count-8 candidates each trained in 88.2 seconds, with SHA-256
`6fe927db741cbedd645ce124aadc29adadf7f9add20156d36513283bf8d82e2c` and
`30fe1363edd618945ed023cb3f65df1769afca6981f284dd79c7cc94a8f0da2d`. On the exposed
4,096-board atlas, count-4 scored 156 failures (69 Has-SET, 87 No-SET), 96.1914% board accuracy,
and 96.0754% triple accuracy. Count-8 scored 170 failures (66 Has-SET, 104 No-SET), 95.8496% board
accuracy, and 95.9656% triple accuracy. Both trail the control's 117 failures (49 Has-SET, 68
No-SET) and 96.2524% triple accuracy. Neither browser nor audit seed was opened; the active-v12
deployment remains unchanged.

Sparse correlation avoids the catastrophic No-SET collapse of full replacement, but even four
injected samples perturb both conditionals enough to lose. The evidence across v20-v22 now rejects
No-SET-board correlation as a productive sampler change under this fixed batch and objective. The
next experiment should target the actual visual failure modes or increase hard-negative coverage
without changing the proven marginal negative distribution.

## 2026-08-20 — Active-v23 differentiable full-board fine-tuning

`ACTIVE_V23_PROTOCOL.md` moved training onto the actual board action surface without changing the
deployable topology. The policy encodes twelve cards once, differentiably gathers their learned
probabilities into all 220 triples, and applies the existing typed relation and SET heads. A focused
test proves that this path matches the existing inference scorer to 1e-10. Full-board backward
passes have finite loss and shared gradients. Because random boards produced almost no action
loss, the pre-run protocol amendment mines the hardest of eight generated boards while retaining
an exactly alternating Has-SET/No-SET schedule.

The native 1,000-step warm-started fine-tune completed in 70.1 seconds and saved locked snapshots.
Step 250 has SHA-256 `ffd9bf6111251d9fe4b25bf2267d03955127fc2462eacdecf0515b07aa459fc3`;
step 1,000 has SHA-256 `1ed84301a290639c9d215951bf768778962fd2a4ca0ec11b4f1d5368b114e923`.
On the exposed 4,096-board atlas, step 250 scored 136 failures (77 Has-SET, 59 No-SET) and
94.4580% triple accuracy. Step 1,000 scored 161 failures (90 Has-SET, 71 No-SET) and 94.0369%
triple accuracy. The control remains 117 failures (49 Has-SET, 68 No-SET) and 96.2524% triple
accuracy. No browser or audit seed was opened; deployment remains unchanged.

The full-board loss genuinely reduced No-SET false positives at 250 steps, but supervising only
the smooth extrema allowed the rest of the triple score surface to deteriorate and caused a larger
Has-SET regression. Full-board ranking needs a balanced dense per-triple auxiliary objective, not
more extremum-only steps.

## 2026-08-20 — Active-v24 balanced dense board supervision

`ACTIVE_V24_PROTOCOL.md` added a class-count-invariant dense SET-logit auxiliary to the v23 board
loss. Has-SET boards give the mean valid-triple BCE and mean invalid-triple BCE equal weight;
No-SET boards use mean invalid BCE. Focused arithmetic tests recover exactly `ln(2)` at zero logits
for both board kinds and prove finite gradients with opposite signs for positive and negative
actions.

The 1,000-step run completed in 69.4 seconds. Step 250 has SHA-256
`7a396994808abe1c31f7f8782419d73afba90c5bf38aeaa20a81d6b0077168cd`; step 1,000 has
SHA-256 `68f462d37b088dd3ec0213470589cf0fbd2acb012c0cce8e90af14b701ffab7a`. On the exposed
4,096-board atlas, step 250 scored 130 failures, exactly split 65 Has-SET and 65 No-SET, with
95.4773% triple accuracy. Step 1,000 scored 180 failures (78 Has-SET, 102 No-SET) with 94.8608%
triple accuracy. Neither clears the 117-failure control; no browser or audit seed was opened.

Dense balancing recovers seven Has-SET failures relative to v23 step 250 and preserves the No-SET
gain, but full-network updates still erode the already-strong visual representation. The next
controlled test should freeze the card encoder and attribute specialists and adapt only the typed
relation/SET decision head.

## 2026-08-20 — Active-v25 decision-head-only board fine-tuning

`ACTIVE_V25_PROTOCOL.md` kept the v24 balanced board and dense triple objectives but froze the
shared visual encoder, all four attribute specialists, and the fill-texture branch. Only the eight
typed relation/SET decision-head tensors were given to AdamW. A checkpoint-scope verifier proves
that all 44 frozen tensors remain bit-identical while all eight permitted tensors change.

The 1,000-step native run completed in 56.1 seconds. Step 250 has SHA-256
`3965caae79c1d119b4264eb61330ff8ec31480deb210b6e32ef7a51fb7dae77a`; step 1,000 has
SHA-256 `0584db5a9b5e1de2092e59fbd3a45825fb135f1443aba191b7133505b6a76749`.
On the exposed 4,096-board atlas, step 250 narrowly cleared the predeclared eligibility gate with
116 failures (53 Has-SET, 63 No-SET), versus the control's 117 (49 Has-SET, 68 No-SET). Step 1,000
scored 121 failures (52 Has-SET, 69 No-SET) and was rejected before browser evaluation.

The eligible step-250 checkpoint then faced the fixed deployed reference on the fresh paired
512-board browser seed `2026090401`. The reference scored 27 failures, with 95.3125% Has-SET and
94.1406% No-SET accuracy. Step 250 scored 32 failures, with 92.5781% Has-SET and 94.9219% No-SET
accuracy. It improved No-SET by two cases but lost seven Has-SET cases, failing both the total-error
and conditional-regression gates. The selector restored the active-v12 checkpoint and frozen
active-v13 board head. The audit seed `2026090499` was not opened.

Balanced dense supervision therefore addresses class-frequency imbalance but not the harder
failure-cost asymmetry: a small downward score shift helps negative boards while turning too many
positive boards into false `No SET` decisions. The next experiment should keep the visual encoder
frozen and train/calibrate an explicitly board-conditional decision layer with a constrained
Has-SET recall objective, rather than continuing unconstrained relation-head fine-tuning.

## 2026-08-21 — Active-v26 recall-constrained board-head calibration

`ACTIVE_V26_PROTOCOL.md` reused the already-exposed 16,384-board active-v19 neural evidence rather
than generating another fitting corpus. The richer L2=0.01 score-distribution head was fit on the
original first 8,192 rows, then its operating threshold was selected under a hard fitting
constraint: neither Has-SET nor No-SET failures could exceed the deployed active-v13 head. The
extension implementation consumes only the frozen FrankenTorch policy's 220 learned triple scores;
an exact runtime replay reproduces all fitting and validation decisions from the offline fitter.

On fitting data, the constrained head reduced failures from 413 to 379 (162 Has-SET, 217 No-SET)
versus the deployed head's 163/250 split. On the fixed 8,192-board validation half it reduced
failures from 362 to 343: 145 Has-SET and 198 No-SET versus 143/219. That 19-board net gain and
two-board Has-SET regression cleared the predeclared development tolerance. The serialized
candidate has SHA-256 `4ee99dc8d0026a64fbe4431bc46f0677a14aca54f39e472d4b1dff1518af3eb4`.

On the fresh paired 512-board browser seed `2026090501`, the active-v13 reference scored 21
failures (95.7031% Has-SET, 96.0938% No-SET). The recall-constrained head scored 20 failures with
identical 95.7031% Has-SET accuracy and 96.4844% No-SET accuracy. It therefore won the locked rule
by one No-SET board and passed the fixed integration smoke: 12/12 valid proposals and all six
guarded clicks transition-verified.

The one predeclared audit at seed `2026090599` rejected the candidate as a robustness solution. It
scored 95.7% raw accuracy with 43 failures, split 95.6% Has-SET (22 failures) and 95.8% No-SET (21
failures). Guarded accuracy was 99.0228% at 30.7% coverage. Of the Has-SET failures, ten were false
`No SET` decisions and twelve selected an invalid strongest triple. Failures covered every capture
profile; crop, gamut, and shadow were again largest. The active-v12 checkpoint and active-v13 head
were restored byte-for-byte.

The constrained head is a genuine and well-balanced calibration improvement, but its fresh gain is
far smaller than the remaining perception floor. Further board-head complexity is now deprioritized.
The next active-learning experiment should feed these retired audit failures back into a targeted
browser-rendered curriculum while updating the weak visual specialists and decision head without
moving the shared encoder or the already-strong count/colour specialists.

## 2026-08-21 — Active-v27 targeted browser-curriculum adaptation

`ACTIVE_V27_PROTOCOL.md` fed all 43 retired v26 audit failures back into training as active
curriculum. Each failure produced one exact replay and three deterministic local/new-board
appearances, mixed with 852 fresh boards for 1,024 total. Chrome rendered the production UI and
streamed framed 30x21 card tensors directly into FrankenTorch; no screenshots, bitmaps, or tensor
corpus were persisted. The complete render plus 1,000 native steps used the active-v12 warm start,
hardest-of-eight full-board mining, and the balanced v24 objective.

Only shape, fill, the fill-texture branch, and the typed relation/SET heads trained. Scope checks
prove all 34 permitted tensors changed and all 18 shared/count/colour tensors stayed bit-identical.
Training completed in 65.5 seconds. Step 250 has SHA-256
`77b46b40b9ae5fd4cd12606e6b78652199a5415bf9628b9e95169f13810bbf8f`; step 1,000 has
SHA-256 `439c1eedb7f36029358393dfba74d08d41f281dfacec92456446332223d8edbc`.

Against active-v12's exposed-atlas result of 98 failures (41 Has-SET, 57 No-SET), step 250 scored
97 (37/60) and cleared the locked gate. Step 1,000 overfit to 103 (59/44) and was rejected. The
first reference-only browser attempt later exposed a harness lifetime defect: after iteration 458,
Chrome left a screenshot promise unresolved and Node reported an unsettled top-level await. No
candidate ran. The partial log is preserved as invalid; a bounded capture promise and two ordered
fresh-browser shards reproduce identical predictions without the long-session stall.

On the complete 512-board seed `2026090602`, active-v12 scored 30 failures (9 Has-SET, 21 No-SET).
Step 250 scored 28 (11 Has-SET, 17 No-SET). Active curriculum therefore transferred a real four-case
No-SET gain, but added two Has-SET failures—one beyond the locked tolerance. It was rejected and no
audit seed was opened. This is evidence for targeted adaptation, but the update strength remains
too asymmetric for direct deployment.

## 2026-08-21 — Active-v28 partial active-curriculum adaptation

`ACTIVE_V28_PROTOCOL.md` interpolated the aligned v27 step-250 update with its exact active-v12
parent at fixed v27 weights 0.25/0.50/0.75. Frozen tensors remain unchanged under interpolation,
and strict scope checks constrain every difference to the same 34 permitted tensors. On the exposed
atlas, weights 0.25/0.50/0.75 scored 101/97/96 failures. The selected 0.75 blend scored 37 Has-SET
and 59 No-SET failures with SHA-256
`0dad6fce0bf195a4ff752d5b9a36add8452aaeca95d03a6478c585736e41ed96`.

On fresh paired browser seed `2026090701`, active-v12 scored 30 failures (11 Has-SET, 19 No-SET).
The 0.75 blend scored 26 and improved both conditionals: 8 Has-SET and 18 No-SET failures. It passed
the fixed 12-case integration smoke with all proposals valid and all six guarded clicks
transition-verified.

The one fixed 1,000-board audit at seed `2026090799` rejected the blend. Across four ordered
fresh-browser process shards it scored 95.8% raw accuracy with 42 failures, split 96.2% Has-SET (19)
and 95.4% No-SET (23). Guarded accuracy was 98.9726% at 29.2% coverage. The script restored the
active-v12 checkpoint and active-v13 head byte-for-byte.

Active curriculum plus partial interpolation is the strongest upstream development result in this
sequence and can improve both classes, but the effect is not stable enough to approach 98% raw on
unseen browser conditions. The next research step should stop tuning a single fixed mixture on
small paired sets and instead enlarge the independent generative validation budget or train a
representation with enough capacity/resolution to reduce the persistent shape/fill error floor.

## 2026-08-21 — Active-v29 doubled-resolution representation test

`ACTIVE_V29_PROTOCOL.md` locks a direct test of whether the `30x21` card tensor discards shape and
fill evidence before the learned policy can use it. The `60x42` candidate keeps the honest shared
convolution, four attribute specialists, fill-texture residual, typed relations, and learned SET
head. One additional learned stride-2 convolution in each specialist and texture path restores the
same `8x6` projection maps. The resulting topology has 58,075 parameters versus the deployed
55,155; it is a resolution/local-processing experiment, not a scaling-law claim.

Both dimensions are compile-time Rust features of the same contract, data loader, policy, MPSGraph
trainer, and WASM runtime. The deployed `30x21` path remains the default, incompatible checkpoints
strict-load fail, and the browser reads its dimensions from WASM. The renderer stream now accepts
explicit model dimensions so `60x42` browser tensors remain in memory and are never persisted.

The mandatory one-step fixed-batch parity gate passed before full training: MPSGraph loss
`14.509955` versus FrankenTorch `14.509956457`, with the first eight shared gradients agreeing
within `2.5e-8`. The initial diagnostic mismatch was a parity-harness error: only
`--validation-atlas-dir` was passed to the native probe, while probe mode reads `--atlas-dir`.
Passing the identical v8 training corpus to both implementations removed the discrepancy. No v29
training or evaluation seed was opened before this correction.

The fresh training corpus contains 1,408 atlases with SHA-256 corpus hash
`5a96f77dc9a42a06558edbbe4f527bc37d13b3cfd3b69a2396f8b2109560cc2d`; the disjoint 704-atlas
eligibility corpus has hash `ef52f067cadfbb15519b92189d3a776ba0087b831ddc5e6e1c13b7f698b79661`.
All eight matched replicas completed the locked 20,000-step schedule with exactly one positive and
one negative browser-rendered replacement per 64-example batch. The four `30x21` runs took
136.124, 136.124, 128.908, and 125.722 optimizer-seconds. The four `60x42` runs took 440.709,
423.606, 432.696, and 428.602 seconds: about 3.3 times the optimizer cost for four times the input
pixels. The 173,680-parameter low-resolution wide control took 122.998 seconds.

On the locked 4,096-board atlas eligibility set, the `30x21` uniform ensemble scored 11 failures
(10 Has-SET, 1 No-SET). The `60x42` uniform ensemble scored one failure (1 Has-SET, 0 No-SET),
99.7314% triple accuracy, and 99.9985% attribute accuracy. It therefore advanced. Individual
high-resolution replicas scored 8, 4, 17, and 14 failures, demonstrating material seed variance;
the uniform ensemble was not replaced with a post-selected subset. The wide low-resolution control
scored 86 failures and did not support a parameter-count explanation.

The paired 2,048-board production-browser result rejected the candidate decisively. The matched
`30x21` ensemble scored 97.6562% raw accuracy with 48 failures (36 Has-SET, 12 No-SET). The
`60x42` ensemble scored only 85.1074% with 305 failures, all false `No SET` actions on Has-SET
boards. The paired difference was -257 cases with bootstrap standard error 15.762. Live shape
accuracy was 98.4823%, below the locked 99.5% gate; fill improved to 99.6134%. No audit seed was
opened.

Completed neural evidence localizes most of the raw-action collapse to distribution shift through
the frozen active-v13 logistic board head rather than failure to identify a visual triple. On the
1,024 Has-SET boards, the high-resolution ensemble's strongest triple was valid 1,010 times and its
native zero-score action would have been correct 1,007 times, but the frozen head accepted only 722
boards and produced 719 correct actions. The old head's mean-score feature moved from -17.63 to
-22.83; its standardized weighted contribution fell from -0.38 to -5.09, overwhelming the
candidate's higher best score. This post-rejection diagnostic does not rescue the candidate: a
zero-score action was not the predeclared browser policy, the paired gain would be small, and the
shape gate still fails. The active-v12 checkpoint and active-v13 head were restored byte-for-byte.

The experiment therefore falsifies the simple claim that doubling the card tensor plus uniform
ensembling solves the live perception floor. It greatly improves the synthetic atlas and fill
classification, but barely changes live strongest-triple correctness (1,010 versus 1,009 Has-SET
boards for the matched control), does not fix live shape, and changes score-distribution geometry
enough to invalidate the inherited board head. Any follow-up must treat board-existence calibration
as topology-specific and must first produce a real live shape improvement on a development stream;
recalibrating this rejected model alone would be another board-head experiment, not evidence for
the resolution hypothesis.
