# Board-v9: typed relations and a real No SET action

Board-v9 extends the raw action space. A policy must either select a valid
three-card SET or select `NoSet` when the board contains no valid triple. This
is not an abstention mechanism: every board has a scored game action and every
action counts at 100% coverage.

## Typed relation supervision

For each of the four attributes (count, colour, shape, and fill), the research
policy now predicts one of three relation classes:

1. `same`: all three cards have the same attribute value;
2. `different`: all three values are distinct; or
3. `invalid`: exactly two values are the same.

The relation head emits continuous logits. During training, a softmax produces
probabilities and cross-entropy provides gradients for all three classes. The
SET head consumes the twelve soft probabilities (four attributes times three
relation types), so no discrete operation interrupts backpropagation. Class
names are assigned with `argmax` only for evaluation and explanation.

The typed research policy has 44,448 learned parameters, 194 more than the v8
research policy. The shipped browser checkpoint remains bound to the separate
legacy deployed policy type.

## Raw board action

The policy enumerates every three-card action and finds its strongest SET
logit. A non-negative strongest logit selects that triple; if every triple is
negative, the raw action is `NoSet`. Zero is the supervised binary decision
boundary, not a fitted abstention threshold.

Mixed-board evaluation alternates exactly between boards generated with at
least one SET and boards generated with no SET. It reports overall raw board
accuracy plus the two conditional accuracies separately. A selectable v9
candidate must meet the eventual locked gate on all three measurements.

## Backend parity gate

The one-step fixed-batch MPSGraph checkpoint produced loss `6.396099`; strict
FrankenTorch loading and evaluation produced `6.396099628`. The first eight
shared-convolution gradients agreed to floating-point precision. Both backends
use explicit relation targets and the same differentiable typed topology.

## Research sequence

1. Use the exposed v8 corpora only for architecture development and determine
   whether ordinary triple supervision controls the maximum false-positive
   logit across 220 triples on a no-SET board.
2. If necessary, add board-distribution hard-negative training without using a
   research or sealed v9 corpus as training data.
3. Lock fresh, disjoint v9 training, research, and sealed procedural seeds and
   fixed balanced board allocations.
4. Select on raw research accuracy only, then run one sealed assessment.

