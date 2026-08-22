# Active-v19 large in-memory board-head protocol

Repeated 512-board browser comparisons vary by several accuracy points, and active-v18 selected a
candidate that immediately failed a fixed clean case. The next experiment scales development data
without persisting screenshots or tensors. This protocol is locked before adding the evaluator or
generating its stream.

## Fast pixel-only evidence stream

Add one FrankenTorch utility that reads the existing versioned rendered-board stream from stdin,
strict-loads one checkpoint, scores each twelve-card tensor board exactly once, and emits neural
board evidence. Ground-truth card IDs may be attached only after the frozen pixel forward pass and
are used only for scoring/fitting. The utility must not contain symbolic candidate selection or an
alternate neural implementation.

Generate exactly 16,384 balanced boards from the production React/SVG renderer at root seed
`2026082901`. A pre-seed throughput probe established that one renderer would take roughly 30
minutes, so execute the unchanged sequence as four concurrent ordered 4,096-board shards with
iteration offsets 0/4096/8192/12288. Pixels remain in each Chrome→pipe→Rust path; do not save
bitmaps or tensor corpora. Concatenate evidence in shard order. Use the first 8,192 boards as fitting
data and the remaining 8,192 as validation data. Because board kind alternates, both halves contain
exactly 4,096 boards of each kind.

## Head search

Keep the active-v12 checkpoint frozen. Compare the deployed active-v13 eight-feature logistic head
with regularized logistic heads over a predeclared richer permutation-invariant feature vector:
the existing eight features, the top 16 sorted triple scores, and fixed score quantiles at
0/5/10/25/50/75/90/95/100 percent. Quantiles use nearest rank `round(p * (n - 1))` on ascending
scores. Search L2 values 0.01/0.1/1/10 on fitting data only using standardized-feature logistic
IRLS, at most 25 iterations, 1e-9 coefficient tolerance, and an unregularized intercept. Select on
validation by fewest failures, then worst conditional accuracy, then fewer features, then stronger
regularization. A richer head is eligible only if neither conditional loses more than 4/4,096
boards versus the deployed head.

Only a richer validation winner advances with the deployed head to a 512-board paired screenshot
comparison at root seed `2026082902`; both use the unchanged checkpoint. A richer head must have
fewer failures and lose no more than one of 256 boards in either conditional. Only a selected richer
head receives one 1,000-board audit at root seed `2026082999`, with the unchanged 98.0% overall and
97.5% per-kind raw-action gates. Fixed integration smoke remains mandatory before audit.
