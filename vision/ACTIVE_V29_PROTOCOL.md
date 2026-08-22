# Active-v29 doubled-resolution representation test

Active-v28 improved the small paired browser set but failed its independent audit. Reweighting,
board-head fitting, threshold shifts, and targeted curriculum have not removed the persistent live
shape/fill floor. This protocol tests whether the fixed `30x21` card tensor destroys useful contour
and stripe information before the learned policy sees it.

This file is locked before implementing the high-resolution topology, rendering any v29 corpus,
training a candidate, or opening an evaluation seed. Results may reject the hypothesis but must not
change these candidates or gates.

## Input and topology candidates

The primary candidate consumes `60x42x3` RGB card tensors produced by the same area resampler as
the deployed `30x21x3` policy. It retains the shared convolution, four honest attribute
specialists, fill-texture residual, typed relation head, and learned SET head. Each specialist path
adds one learned stride-2 `3x3` convolution after its existing residual block; the fill-texture path
adds one learned stride-2 `3x3` convolution after its existing two downsamplers. Their final spatial
maps therefore remain `8x6`, matching the deployed projection inputs. This is a resolution and
local-processing intervention, not a transformer, VLM, VLA, symbolic solver, or neural architecture
search system.

The matched control is the unchanged deployed `30x21` topology retrained from scratch with the
identical data, loss, optimizer, step budget, and four seed pairs. A secondary capacity control is
the existing `30x21` wide typed-relation policy trained under the same budget; its exact parameter
count is reported rather than described as parameter matched if it differs from the candidate.

Before full training, the `60x42` implementation must pass one fixed-batch MPSGraph/FrankenTorch
forward-loss and first-shared-gradient parity probe. Checkpoints are strict topology-specific state
dictionaries; neither runtime nor tooling may reshape or silently adapt incompatible weights.

## Fresh training distribution

Render a new atlas corpus, never used for v29 selection, from procedural seeds `29001..29004`, all
four themes, widths `96,144,208,320`, all eleven capture profiles, and DPRs 1 and 2. Both resolution
arms read these identical source rasters and resample them only at the model boundary.

Also render 2,048 fresh production-browser boards in memory from root seed `2026090801`. Do not
persist screenshots or card tensors. Replace exactly one Has-SET and one No-SET example in every
64-example atlas batch with samples from this collection. This is two browser examples per batch,
not a whole-browser batch and not active failure replay.

Each arm trains four independent replicas for exactly 20,000 MPSGraph optimizer steps, batch size
64, AdamW, cosine learning-rate schedule, ordinary augmentation, attribute-loss weight 4, and the
weight-2.5 negative log-sum-exp objective. Do not apply shape/fill multipliers.

| Replica | Data seed | Initialization seed |
|---:|---:|---:|
| 1 | 890021 | 16045690984503098046 |
| 2 | 990029 | 10723151780598845931 |
| 3 | 1090033 | 14313749767032793493 |
| 4 | 1190039 | 11400714819323198485 |

## Fresh atlas eligibility

Render a separate eligibility corpus from seeds `29101..29102`, all four themes, widths
`104,152,216,344`, all eleven profiles, and both DPRs. Evaluate 4,096 balanced mixed boards with
triple seed `327145488795021` and board seed `324222009935245`.

Compare, without coefficient fitting:

1. each `30x21` replica;
2. their uniform four-model logit ensemble;
3. each `60x42` replica;
4. their uniform four-model logit ensemble; and
5. the secondary wide `30x21` control.

An arm is atlas-eligible only if its uniform ensemble has fewer failures than the matched `30x21`
uniform ensemble and neither conditional has more failures. Select the best eligible v29 action by
fewest failures, then triple accuracy, then lower runtime cost. Single-replica results remain
diagnostics and cannot replace the predeclared ensemble comparison.

## Paired browser selection

Compare the eligible `60x42` ensemble against the matched `30x21` ensemble on 2,048 paired fresh
browser boards at root seed `2026090891`, split into eight ordered 256-board shards in fresh Chrome
processes. The candidate advances only if:

- it has fewer total raw-action failures by at least two paired standard errors under a paired
  bootstrap over the fixed 2,048 cases (20,000 case-resampling replicates, deterministic bootstrap
  seed `2026090892`);
- neither Has-SET nor No-SET has more failures than the control; and
- live per-card shape and fill accuracy are each at least 99.5% on the same frozen evidence stream.

Guarded accuracy and coverage are diagnostics. Reject automatically if guarded accuracy improves
while raw accuracy does not, or if coverage falls below the matched control. Run the fixed extension
integration smoke after selection.

## One sealed audit

Only a selected candidate opens root seed `2026090999`: 2,000 feedback-free browser boards split
into eight ordered 250-board fresh-process shards. Promotion requires at least 98.0% overall raw
accuracy and at least 97.5% raw accuracy for both Has-SET and No-SET. Record guarded metrics but do
not use abstention to satisfy the raw gate. Record the development-to-audit gap.

Until every gate passes, the deployed active-v12 checkpoint and active-v13 board head remain
byte-identical. A rejected candidate is logged and retained only as a research artifact.

## Follow-up ablations

If the doubled-resolution ensemble clears the sealed audit, run two locked follow-ups before final
deployment: remove the two browser examples per batch to isolate sparse browser injection, and
compare the uniform four-model ensemble with its strongest single replica. If resolution clears the
audit but the ensemble does not materially improve it, deploy the single replica. Targeted v27
curriculum plus fixed 0.75 parent interpolation is deferred until these representation ablations are
complete.
