# Active-v22 sparse board-correlated negative injection

Active-v20/v21 show that replacing all 32 ordinary negatives with No-SET-board triples improves
local relation accuracy but destroys broad No-SET rejection, even after restoring aggregate logit
scale. This protocol is locked before implementing the sparse sampler.

## Controlled change

Start from the exact active-v10 rate-zero batch and objective: 32 positive triples, 32 ordinary
seed-stratified negative triples, and one 32-term negative log-sum-exp with weight 2.5. Replace only
the first 4 or 8 positions of the negative half with distinct triples from one generated No-SET
board rendered through one independently sampled atlas. Do not add grouped losses or count
corrections. Keep topology 55,155, 10,000 steps, batch 64, board-v8 corpus, data seed `590011`, and
initialization seed `3935559000370003845` unchanged.

Require sampler boundary tests, CLI validation, and one-step FrankenTorch/MPSGraph loss and shared
gradient parity for an injection count of four.

## Selection and audit

On the exposed 4,096-board atlas, a candidate must beat the control's 117 total failures and stay
within four failures of each control conditional (at most 53 Has-SET and 72 No-SET failures). The
best eligible candidate advances to 512 feedback-free browser boards at seed `2026090101` against
the current active-v12 deployment and frozen active-v13 head. It must have fewer total failures and
lose no more than one of 256 boards in either conditional. Fixed integration smoke is mandatory.

Only a selected candidate receives the 1,000-board audit seed `2026090199`, under the unchanged
98.0% overall and 97.5% per-kind raw-action gates. Guarded accuracy is diagnostic only.
