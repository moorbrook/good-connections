# Active-v18 aligned shape-checkpoint soup protocol

Active-v16 trained shape-loss endpoints from the same initialization, ordinary-data RNG stream,
optimizer schedule, and base batches as its fixed control. Both endpoints improved isolated shape
accuracy but worsened browser action accuracy. Active-v12 previously showed that aligned parameter
interpolation can preserve complementary behavior that either endpoint loses. This protocol is
locked before constructing or evaluating any v18 soup.

## Fixed ingredients and search

Use exactly these checkpoints:

- fixed control SHA-256 `dc7819edafb75d80914a6f56d154a19d9e2e721a96f1dccfe2482a97f621717e`;
- shape-m2 SHA-256 `43e3fd8b654bb4c7ca191ecbd8c076b0798d895b24552582541efcc4f36481fa`;
- shape-m4 SHA-256 `422657db2b593951f2602906493440433d93d946e785766c68cc37ef0131bc3c`.

For each shape endpoint, construct soups with shape-checkpoint primary weights 0.25, 0.50, and
0.75. The strict Rust blender must reject any key, shape, dtype, length, or finite-value mismatch.
This produces one 55,155-parameter checkpoint per candidate and does not create a runtime ensemble.

Evaluate all six soups on the exposed 4,096-board atlas diagnostic. Advance the two soups with the
fewest raw zero-threshold failures, breaking ties by worst conditional accuracy and then lower
shape multiplier and lower shape weight.

## Browser selection and audit

Compare those two soups against the current active-v12 deployed checkpoint on 512 feedback-free
browser boards at root seed `2026082801`. Use the same frozen active-v13 fitted board head for all
three models. Select by fewest failures, then worst conditional accuracy, then the current deployed
checkpoint. A soup is eligible only if neither conditional loses more than one of 256 boards versus
the current deployment.

Only a selected soup receives one 1,000-board feedback-free audit at root seed `2026082899`.
Promotion requires at least 98.0% overall and 97.5% for each board kind at 100% raw action coverage.
Guarded accuracy is diagnostic only. If the current deployment wins, do not open the audit seed.
