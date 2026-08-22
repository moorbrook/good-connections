# Active-v27 targeted browser-curriculum specialist adaptation

The active-v26 audit is retired for claims but provides 43 concrete feedback examples: 22 Has-SET
and 21 No-SET failures spanning every capture profile. Twelve Has-SET failures selected an invalid
triple, ten emitted false `No SET`, and all 21 No-SET failures were false clicks. Prior whole-domain
browser mixing (active-v14) caused forgetting, while full-network board fine-tuning (active-v24)
damaged the visual representation. This protocol is locked before rendering or training v27.

## In-memory active curriculum

Turn each of the 43 retired failures into four deterministic browser appearances: one exact replay,
one same-board appearance with a new procedural visual seed, one nearby viewport appearance of the
same board, and one same-condition appearance with a new board and visual seed. Preserve board kind,
theme, capture profile, and DPR. Mix those 172 feedback-derived boards with 852 ordinary balanced
fresh boards generated at seed `2026090601`, for 1,024 rendered boards total. The renderer uses the
production React/SVG board and screenshot crop/resampler. Stream framed 30x21 card tensors directly
into the trainer, retain them only in training memory, and persist no screenshots, bitmaps, or
tensor corpus.

The trainer alternates Has-SET and No-SET steps and mines the hardest of eight boards of the required
kind from this collection. Keep the active-v24 board-extremum, balanced dense-triple, attribute, and
typed-relation objectives unchanged.

## Controlled parameter scope

Warm-start the exact deployed active-v12 checkpoint with SHA-256
`ca20943318755c037a9ac08faaeb3887ec06ae4ae27a3c7b2017e1cabfc3bbec`. Freeze the shared
convolution and the already-strong count and colour specialists. Train only shape specialist
`attribute_head_2`, fill specialist `attribute_head_3`, the fill-texture branch, and the eight
typed relation/SET decision-head tensors. Use data seed `590013`, learning rate `2e-5`, AdamW weight
decay `1e-4`, 1,000 steps, and snapshots at 250 and 1,000. A scope verifier must prove all frozen
tensors remain bit-identical and every changed tensor belongs to an allowed prefix.

## Selection and audit

The deployed active-v12 atlas baseline is fixed at 98/4,096 failures (41 Has-SET, 57 No-SET),
97.0642% triple accuracy, and 99.7243% attribute accuracy. A snapshot is eligible only with fewer
than 98 total failures, no more than 45 Has-SET failures, and no more than 61 No-SET failures on the
unchanged exposed atlas. Select by fewest failures, then triple accuracy, then earlier snapshot.

Only an atlas-eligible snapshot advances against the deployed checkpoint on 512 paired fresh
browser boards at seed `2026090602`, both using the frozen active-v13 head. It must have fewer total
failures and lose no more than one of 256 boards in either conditional. Fixed 12-case integration
smoke remains mandatory.

Only a selected candidate receives one 1,000-board audit at seed `2026090699`, under the unchanged
98.0% overall and 97.5% per-kind raw-action gates. Guarded accuracy is diagnostic only. Failure at
any downstream gate restores active-v12 plus active-v13.

## Pre-result browser infrastructure amendment

The first reference-only browser attempt reached iteration 458, then Chrome left a screenshot
promise unresolved and Node terminated with an unsettled top-level await. No candidate board was
run and no complete reference metric existed. Preserve that log as invalid. Add a bounded capture
promise and replay the exact predeclared 512-case sequence as two ordered 256-board shards starting
at iterations 0 and 256, each in a fresh browser process. Sum conditional counts before applying
the unchanged selection rule. This changes process lifetime only, not cases, model, head, or gate.
