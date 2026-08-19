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

The raw selection run is fixed at 1,024 triple batches and 8,192 boards. Rank by raw board failures,
then raw triple accuracy, then parameter count. Only a candidate at or above 99% research accuracy
(at most 81 failures in 8,192 boards) may receive the one sealed 10,000-board evaluation. Do not fit
or apply abstention during this cycle. Full protocol details are in `BOARD_V7_PROTOCOL.md`.
