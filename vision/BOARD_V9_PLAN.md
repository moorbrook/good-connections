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

The first development baseline answered step 1 negatively. Despite 99.3286%
typed-relation accuracy, its mixed raw-board accuracy was 89.7461%, including
82.2754% on no-SET boards. The best post-hoc static threshold reached only
95.8984%. Step 2 therefore begins with a differentiable negative log-sum-exp
loss: a training batch groups its negative triples, and an auxiliary binary
loss controls their aggregate false-positive mass rather than only their mean.

The fixed-seed 10,000-step weight sweep peaked at 96.9238% mixed accuracy for
weight 2.5 on 2,048 exposed development boards. The same checkpoint scored
96.2891% on 4,096 boards; extending to 20,000 steps scored 96.2402%, and their
uniform ensemble scored 96.8018%. A follow-up sampler using combinations from
one actual no-SET board per negative half was rejected at 89.3555%. The next
development step tests independent replicas of the selected 2.5 objective,
because the v8 work showed that independent visual errors can be reduced by a
predeclared uniform ensemble.

Four independent replicas reached 97.0703% as a uniform ensemble, and an exact
threshold sweep reached only 97.4121%. A 173,680-parameter width candidate
improved local attribute and relation metrics but tied the standard model at
96.2891% mixed raw accuracy; a second loss weight was worse. Both routes are
rejected. The next architecture experiment must give `NoSet` an explicit
board-level learned output rather than deriving it solely from whether the
largest of 220 independent triple logits crosses zero.

A held-out board-head probe using rich statistics of all 220 final scores
reached 98.1445%; a nonlinear eight-unit MLP did not exceed the linear probe.
The follow-up exported all per-triple typed-relation evidence. A local shared
MLP reached 96.7285%, a differentiable multiple-instance reader reached
97.1680%, and a rich pooled MLP overfit from 99.3652% training to 96.5332%
held out. The current triple selector also caps a perfect board classifier at
99.2188% on that split. These routes are rejected rather than integrated.

Normalized fill-loss multipliers improved the bottleneck fill specialist from
98.5026% to 98.7956% but reduced mixed raw accuracy; simple specialist
reweighting is also rejected. The next architecture experiment must reason
contextually across the twelve shared cards before triple scoring, allowing
the model to reconcile a locally plausible attribute error with the rest of
the board. This is a learned contextual reader, not a symbolic SET solver.
