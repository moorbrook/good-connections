# Active-v26 recall-constrained board-head calibration

Active-v25 improved No-SET rejection but lost too much Has-SET recall. Active-v19 already showed
that a richer score-distribution head contains useful board-existence signal: its best fixed-zero
operating point removed 28 No-SET validation failures while adding five Has-SET failures, missing
the locked tolerance by one board. This protocol is locked before deriving a new threshold.

## Controlled change

Keep the deployed active-v12 FrankenTorch checkpoint frozen. Reuse the already-exposed active-v19
16,384-board neural-evidence corpus and its original ordered 8,192 fitting / 8,192 validation
split. Do not render new training data. Fit exactly one richer logistic head on the fitting half:
the active-v19 base eight board features, top 16 descending triple scores, and nine fixed score
quantiles, standardized on fitting data, with L2 `0.01`, unregularized intercept, IRLS capped at 25
iterations, and coefficient tolerance `1e-9`.

Unlike active-v19, do not force the decision threshold to zero. Enumerate every midpoint between
distinct fitting logits plus the two constant-action boundaries. A candidate predicts Has-SET at
`logit >= threshold`. Retain only thresholds whose fitting Has-SET and No-SET failure counts are no
worse than the deployed active-v13 head on the same fitting rows. Select the retained threshold by
fewest total failures, then best worst-conditional accuracy, then the lower threshold to favor
Has-SET recall. Absorb the selected threshold into the serialized intercept.

Evaluate that one locked candidate on the untouched-by-v26 active-v19 validation half. It is
eligible for browser testing only if it has fewer total failures than the deployed head and loses
no more than 4 of 4,096 boards in either conditional. The validation data is development evidence,
not an audit claim.

The richer runtime may read the policy's existing 220 learned triple scores to reproduce these
features. It must not use card IDs, symbolic SET rules, labels, or a second policy implementation.
The current deployed head remains unchanged unless the candidate wins the browser gate.

## Browser selection and audit

Compare the eligible candidate and deployed active-v13 head on the identical 512 fresh browser
boards at root seed `2026090501`, using the unchanged active-v12 checkpoint. The candidate must
have fewer total failures and lose no more than one of 256 boards in either conditional. Fixed
12-case extension smoke is mandatory after selection.

Only a selected candidate receives the one 1,000-board audit at root seed `2026090599`, under the
unchanged raw-action gates: at least 98.0% overall, 97.5% Has-SET, and 97.5% No-SET. Guarded
accuracy remains diagnostic only. A browser loss, smoke failure, or audit failure restores the
deployed active-v13 head; unopened downstream seeds remain sealed.
