# Active-v13 board-existence head protocol

Active-v12's fresh audit exceeded 20 errors before completion and was strongly asymmetric: Has-SET
recall was high while No-SET rejection was poor. This protocol is locked before collecting any
all-board evidence for a learned board head.

## Development evidence

Keep the selected 50/50 soup checkpoint fixed. Run 1,024 fresh balanced browser boards at root seed
`2026082301` and record, after each frozen pixel-only inference, the board kind and the model's
strongest-triple diagnostics: score, runner-up margin, selected-card entropy and margin, all-triple
score mean, standard deviation, log-sum-exp, and non-negative fraction. The evidence journal stores
no pixels. Audit mode must reject evidence output.

The originally specified even/odd split is invalid because the browser generator deliberately
alternates board kind by iteration: it would put every No-SET board in one split and every Has-SET
board in the other. This was discovered by the first fit report, before any head was deployed or
browser selection was run. Preserve that report as `invalid-even-odd-*` evidence of the protocol
failure. Replace only the split rule with a deterministic stratified split: within each board kind,
rank records by `mix32(boardSeed ^ 0x4d595df4)` (then board seed and iteration as tie-breakers); the
first 256 form fitting and the last 256 form validation. This guarantees that each split contains
256 boards of each kind without using model outputs to assign records. Standardize every feature
using fitting-split mean and population standard deviation only.
Fit deterministic logistic board-existence heads for L2 values `0`, `0.001`, `0.01`, and `0.1`
using full-batch gradient descent: 5,000 steps, learning rate 0.05, zero initialization, and the mean
binary cross-entropy plus half-L2 penalty on weights but not bias. Also evaluate a score-threshold-
only baseline at every midpoint between adjacent fitting scores.

An action is correct when the head rejects a true No-SET board, or accepts a Has-SET board whose
strongest proposed triple is genuinely a SET. Select on validation action failures, then worst
conditional action accuracy, then prefer the threshold baseline over logistic and stronger L2 over
weaker L2. Do not select on board-kind classification alone.

## Browser selection and audit

Compare the selected head with the unchanged raw zero-threshold action on 512 feedback-free browser
boards at root seed `2026082302`. Keep the head only if it has fewer raw action failures and does not
reduce either conditional accuracy by more than one board. The chosen configuration then receives
exactly one 1,000-board feedback-free audit at root seed `2026082399`. The unchanged gate is 98.0%
raw overall and 97.5% for each board kind at 100% raw coverage.
