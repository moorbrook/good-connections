# Active-v15 neural-attribute relation-reader protocol

Active-v14's selected distribution-aligned checkpoint failed its fresh audit at 94.8%; crop was the
largest failure profile. Before changing input resolution, measure whether the frozen soup's neural
attribute probabilities already contain a better action than its learned triple scorer. This
protocol is locked before collecting per-card probability evidence.

## Evidence and oracle boundary

Restore the active-v12 50/50 soup checkpoint and active-v13 fitted board head. Extend the WASM
diagnostic ABI to retain all 12x4x3 post-softmax neural attribute probabilities and all 220 learned
triple scores from the completed pixel-only forward pass. The production extension does not read
these arrays; the headless harness requests them only when `--evidence` is active. Card IDs and true
SET labels are attached by the harness after inference. Audit mode continues to reject evidence
output. Persist no pixels.

Collect 1,024 balanced fresh browser boards at root seed `2026082501`. Split each board kind by the
same fixed hash-ranked rule used in active-v13: first 256 fit, last 256 validate. All normalization
and thresholds use fitting records only.

## Fixed readers

Enumerate triples in lexicographic card-position order. For each attribute and triple, compute from
the neural probabilities:

- `same = sum_c p1[c] p2[c] p3[c]`
- `different = sum over the six distinct class permutations`
- `valid = same + different`

The exact-continuous triple score is the sum over four attributes of `log(max(valid, 1e-12))`.
Evaluate these predeclared readers:

1. `current`: the extension's existing learned strongest triple plus fitted active-v13 board head.
2. `exact-continuous`: choose the maximum exact-continuous triple and fit an action threshold at
   every midpoint between adjacent fitting board maxima.
3. `discrete-attributes`: argmax each card attribute, select the valid predicted SET with greatest
   exact-continuous score, or emit No-SET if none exists. It has no fitted threshold.
4. `blend-025`, `blend-050`, and `blend-075`: standardize every learned and exact triple score with
   one global fitting-only mean and population standard deviation per score family, blend using the
   stated learned-score weight, choose the maximum, and fit the action threshold on fitting only.

For each thresholded reader, select the threshold with the fewest fitting action failures, then the
best worst-conditional fitting accuracy, then the higher (more No-SET-conservative) threshold.

An action is correct only when it rejects a true No-SET board or accepts a Has-SET board and selects
a truly valid triple. Select by validation failures, then worst conditional accuracy, then prefer
`current`, `exact-continuous`, `blend-050`, `blend-025`, `blend-075`, and finally the discrete
reader. Emit a full report including fitting and validation metrics for every reader.

## Browser selection and audit

Only if a non-current reader wins validation, implement its action composition once in Rust under
`vision-policy`/`vision-wasm`; JavaScript may select a named Rust reader but must not duplicate its
math. Cross-check the Rust output against the evidence analyzer on fixed fixtures. Compare current
and selected readers on 512 feedback-free browser boards at root seed `2026082502`. Keep the new
reader only if it has fewer total failures and neither conditional loses more than one of 256
boards.

The chosen complete policy then receives exactly one 1,000-board feedback-free audit at root seed
`2026082599`. Promotion still requires 98.0% overall raw action accuracy and 97.5% for each board
kind at 100% coverage.
