# Active-v20 multi-board negative-group protocol

Active-v9's negative log-sum-exp loss aggregates unrelated negative triples, while its rejected
board-correlated sampler puts all 32 negative slots on one No-SET board and one raster condition.
Active-v19 confirms that No-SET false positives remain the stable large-sample weakness and that a
richer post-hoc head cannot fix them without losing Has-SET actions. This protocol is locked before
adding the group-count control or training candidates.

## Controlled change

Keep the deployed 55,155-parameter topology, 10,000 steps, batch size 64, board-v8 training corpus,
data seed `590011`, initialization seed `3935559000370003845`, normalized attribute loss, and
negative log-sum-exp weight 2.5. The positive half retains ordinary seed-stratified SET triples.
Partition the 32 negative slots into either 4 groups of 8 or 8 groups of 4. Each group samples a
separate generated No-SET board and one independently sampled atlas, then draws distinct triples
from that board. Compute log-sum-exp and a negative BCE target separately per group, then average.

The fixed control is active-v10 rate zero, SHA-256
`dc7819edafb75d80914a6f56d154a19d9e2e721a96f1dccfe2482a97f621717e`. Require unit tests for
group layout and argument validation plus one-step FrankenTorch/MPSGraph loss and shared-gradient
parity for four groups.

## Selection and audit

Evaluate control and both candidates on the exposed 4,096-board atlas stream. Advance the best
grouped candidate only if it has fewer failures than control and neither conditional loses more
than 4/2,048 boards. Then compare it with the current active-v12 deployment using the frozen fitted
active-v13 head on 512 feedback-free browser boards at seed `2026083001`. It must have fewer total
failures and lose no more than one of 256 boards in either conditional. Fixed integration smoke is
mandatory before any audit.

Only a selected grouped candidate receives one 1,000-board feedback-free audit at seed
`2026083099`, with the unchanged 98.0% overall and 97.5% per-kind raw-action gates. Guarded accuracy
is diagnostic only.
