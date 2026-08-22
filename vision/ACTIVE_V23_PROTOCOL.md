# Active-v23 differentiable full-board fine-tuning

The deployed policy already encodes twelve cards once during inference and scores all 220 triples
from their learned attribute probabilities. Training, however, presents isolated triples. The
v20-v22 failures show that sampled approximations to the board action surface distort the negative
distribution. This protocol is locked before exposing the differentiable board path or running a
candidate.

## Controlled change

Warm-start from the fixed active-v10 rate-zero checkpoint, SHA-256
`dc7819edafb75d80914a6f56d154a19d9e2e721a96f1dccfe2482a97f621717e`. For each optimizer step,
generate one twelve-card board, alternating Has-SET and No-SET. Encode its twelve rasterized cards
once, differentiably gather the learned card probabilities into all 220 triples, and reuse the
existing typed relation and SET heads.

Use a temperature-0.1 smooth maximum and margin 0.25. Every board penalizes the strongest invalid
triple above the zero action threshold. A Has-SET board also penalizes its strongest valid triple
below zero and requires the strongest valid triple to exceed the strongest invalid triple by the
margin. Retain normalized card-attribute cross entropy at weight 4 and typed-relation cross entropy
at weight 1 as auxiliary supervision over the complete board. The labels come only from the
procedural generator; inference remains pixel-only.

Run one deterministic 1,000-step fine-tune with data seed `590011`, learning rate `8e-5`, AdamW
weight decay `1e-4`, and label-preserving augmentation. Save locked snapshots at 250 and 1,000
steps. Require a forward-equivalence test between differentiable board scoring and the existing
inference scorer before training, plus finite loss/gradient checks.

**Pre-run amendment after the two-step safety smoke:** random training boards produced effectively
zero action loss, which would leave auxiliary losses to drift the policy. At each step, generate
eight fresh boards of the required class from the training corpus, score them without gradients
using the same smooth-threshold/ranking formula, and backpropagate through only the hardest board.
This keeps the Has-SET/No-SET schedule exactly balanced while targeting the rare action-surface
tail. The pool size eight is fixed before the 250/1,000-step run and is not swept.

## Selection and audit

Evaluate both snapshots and the control on the exposed 4,096-board atlas. A snapshot must beat 117
total failures and stay within four failures of each control conditional (at most 53 Has-SET and 72
No-SET failures). The best eligible snapshot advances to 512 feedback-free browser boards at seed
`2026090201` against the current active-v12 deployment with the frozen active-v13 head. It must
have fewer total failures and lose no more than one of 256 boards in either conditional. Fixed
integration smoke is mandatory.

Only a selected candidate receives the 1,000-board audit seed `2026090299`, under the unchanged
98.0% overall and 97.5% per-kind raw-action gates. Guarded accuracy is diagnostic only.
