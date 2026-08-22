# Active-v24 board loss with balanced dense triple supervision

Active-v23 reduced No-SET failures from 68 to 59 at step 250 but increased Has-SET failures from 49
to 77 while triple accuracy fell by 1.79 points. The extremum/ranking loss improved the intended
tail but left 218 or more non-extreme actions per board weakly constrained. This protocol is locked
before adding dense supervision.

## Controlled change

Retain the exact v23 warm start, differentiable twelve-card/220-triple path, hardest-of-eight board
mining, alternating board classes, temperature 0.1, margin 0.25, attribute and typed-relation
auxiliaries, data seed `590011`, learning rate `8e-5`, and AdamW settings. Add one dense SET-logit
loss at weight 1. On Has-SET boards, average positive-target BCE and negative-target BCE separately
and give the two means equal weight. On No-SET boards, use the mean negative-target BCE. This makes
the loss independent of how many valid triples a generated board contains and prevents 220-action
class counts from overwhelming valid actions.

Run 1,000 steps and save snapshots at 250 and 1,000. Require focused arithmetic tests or finite
gradient checks before the locked run.

## Selection and audit

Use the unchanged 4,096-board atlas gate: fewer than 117 failures, at most 53 Has-SET failures, and
at most 72 No-SET failures. The best eligible snapshot advances to 512 feedback-free browser boards
at seed `2026090301` against active-v12 plus the frozen active-v13 head, with fewer total failures
and no more than one conditional loss per 256. Fixed integration smoke remains mandatory.

Only a selected candidate receives audit seed `2026090399` under the unchanged 98.0% overall and
97.5% per-kind raw-action gates. Guarded accuracy is diagnostic only.
