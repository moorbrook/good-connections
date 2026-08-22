# Active-v16 shape-supervision protocol

Active-v15 measured the frozen soup on 12,288 fresh browser card appearances: shape was the weakest
attribute at 98.4456%, followed by fill at 98.9176%; count and colour were above 99.7%. Exact and
blended relation readers did not beat the current policy on validation. This protocol is locked
before adding the shape-loss control or training its candidates.

## Controlled change

Keep the deployed 55,155-parameter fill-texture topology unchanged. Generalize the normalized
attribute loss so shape and fill can be weighted independently. With multipliers `s` and `f`, use
`(4 * mean_all + (s - 1) * shape_loss + (f - 1) * fill_loss) / (2 + s + f)`. Thus `s=f=1`
is byte-for-byte the old objective and the average attribute-loss scale stays fixed. Implement the
same formula in FrankenTorch and MPSGraph and require a one-step loss/first-gradient parity probe.

Train exactly two 10,000-step MPSGraph candidates on the existing board-v8 training atlases, using
data seed `590011`, initialization seed `3935559000370003845`, negative log-sum-exp weight 2.5,
fill multiplier 1, and shape multipliers 2 and 4. The fixed control is the matching rate-zero
checkpoint with SHA-256 `dc7819edafb75d80914a6f56d154a19d9e2e721a96f1dccfe2482a97f621717e`.
No browser evidence, active curriculum, or audit failures enter training.

## Selection and audit

Run all candidates on the exposed 4,096-board atlas diagnostic. Then compare the control and both
shape candidates with the raw zero-threshold action on 512 feedback-free browser boards at root
seed `2026082601`. Select by fewest action failures, then worst conditional accuracy, then lower
shape multiplier. A shape candidate is eligible only if neither conditional loses more than one of
256 boards versus control.

Only a selected shape candidate receives one 1,000-board feedback-free audit at root seed
`2026082699`. Promotion requires at least 98.0% overall and 97.5% for each board kind at 100% raw
coverage. If control wins browser selection, do not repeat its already-failed class of audits.
