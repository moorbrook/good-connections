# Active-v14 full browser-distribution training protocol

Active-v13 showed that a scalar board-existence head can rebalance Has-SET and No-SET decisions but
does not remove the approximately five-percent live-browser action error. Failures span every
capture profile and theme. This protocol is locked before generating the new training corpus or
training a candidate.

## Hypothesis and data

The atlas-first training distribution is mismatched to the responsive browser distribution. Train
on ordinary browser boards rather than only replaying model-selected failures. Generate 2,048
balanced boards at root seed `2026082401` with the same deterministic fresh-case generator used by
the streaming evaluator: alternating Has-SET/No-SET kind and randomized theme, capture profile,
viewport, DPR, board seed, and visual seed. Rendering uses the production React/SVG board. Capture,
crop, and area-resample each card to the runtime's 30x21 RGB tensor entirely in memory and stream
the framed tensors to Rust; persist neither screenshots nor card bitmaps. Card IDs are training
labels attached after rendering, never runtime inputs.

The rendered-board collection must expose a balanced triple sampler. Positive triples come only
from Has-SET boards; negative triples may come from either kind. A batch samples multiple boards,
random triples, and randomized card order rather than allowing one board to monopolize a step.
Keep the historical failure-curriculum mixer as a separate, honestly named path.

## Fixed candidates

Use the deployed 55,155-parameter fill-texture topology, data seed `590011`, initialization seed
`3935559000370003845`, 10,000 MPSGraph steps, batch size 64, AdamW's existing schedule, typed
relations, and negative log-sum-exp weight 2.5. Train exactly two candidates from the shared initial
seed:

- `browser-r050`: choose a full balanced browser batch on 50% of steps and the existing grouped
  atlas batch on the remainder.
- `browser-r100`: use a balanced browser batch on every step.

Do not use failure curriculum, the active-v13 evidence labels, or its audit cases. Run the existing
4,096-board exposed atlas evaluation only as a regression diagnostic; it cannot select the winner.

## Browser selection and audit

Compare the unchanged active-v12 soup reference and both candidates with the raw zero-threshold
action head on 512 feedback-free browser boards at root seed `2026082402`. Select by fewest action
failures, then worst conditional accuracy, then prefer the greater atlas mixture (`browser-r050`,
then reference, then `browser-r100`) on an exact tie. A candidate is eligible only if neither
conditional accuracy is more than one board worse than the reference. This comparison selects an
upstream visual policy; do not refit a board head on it.

The winner then receives exactly one 1,000-board feedback-free audit at root seed `2026082499` with
the raw zero-threshold head. Promotion still requires at least 98.0% overall raw action accuracy and
at least 97.5% for each board kind at 100% coverage. If it fails, retire that audit and use its
failures only in a later, separately precommitted experiment.
